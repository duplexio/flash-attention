"""Benchmark col_limit cross-attention vs materialized mask (PyTorch SDPA)."""

import torch
import torch.nn.functional as F
import time
import argparse

from flash_attn.cute.interface import flash_attn_varlen_func


def bench_fn(fn, warmup=10, rep=100):
    """Benchmark a function using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep  # ms


def make_col_limit(seqlen_q, seqlen_k, device, sparsity=0.5):
    """Generate col_limit where ~sparsity fraction of KV is visible on average."""
    # Linear ramp: row i sees i/seqlen_q * seqlen_k positions (like causal but for cross-attn)
    visible = torch.linspace(0, seqlen_k, seqlen_q, device=device).int()
    visible = (visible * sparsity * 2).clamp(0, seqlen_k).to(torch.int32)
    # Make monotonically non-decreasing
    visible = torch.sort(visible)[0]
    return visible


def materialize_and_sdpa(q_batch, k_batch, v_batch, col_limit_batch, seqlen_q, seqlen_k):
    """Reference: materialize full mask, use PyTorch SDPA."""
    # q_batch: (1, seqlen_q, nheads, d) -> (1, nheads, seqlen_q, d)
    q_t = q_batch.transpose(1, 2)
    k_t = k_batch.transpose(1, 2)
    v_t = v_batch.transpose(1, 2)

    # Build additive mask (1, 1, seqlen_q, seqlen_k)
    kv_idx = torch.arange(seqlen_k, device=q_batch.device).unsqueeze(0)
    visible = col_limit_batch.unsqueeze(1)
    bool_mask = kv_idx < visible  # (seqlen_q, seqlen_k)
    attn_mask = torch.where(bool_mask, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)
    attn_mask = attn_mask.to(q_batch.dtype)

    out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask)
    return out.transpose(1, 2)  # back to (1, seqlen_q, nheads, d)


def run_benchmark(seqlen_q, seqlen_k, nheads, d, sparsity, batch_size=1, backward=False):
    device = "cuda"
    dtype = torch.bfloat16

    total_q = seqlen_q * batch_size
    total_k = seqlen_k * batch_size

    q = torch.randn(total_q, nheads, d, dtype=dtype, device=device, requires_grad=backward)
    k = torch.randn(total_k, nheads, d, dtype=dtype, device=device, requires_grad=backward)
    v = torch.randn(total_k, nheads, d, dtype=dtype, device=device, requires_grad=backward)

    cu_q = torch.arange(0, total_q + 1, seqlen_q, dtype=torch.int32, device=device)
    cu_k = torch.arange(0, total_k + 1, seqlen_k, dtype=torch.int32, device=device)

    # Build release mask
    parts = []
    for _ in range(batch_size):
        parts.append(make_col_limit(seqlen_q, seqlen_k, device, sparsity))
    col_limit = torch.cat(parts)

    # For SDPA: need batched tensors
    q_batch = q.view(batch_size, seqlen_q, nheads, d)
    k_batch = k.view(batch_size, seqlen_k, nheads, d)
    v_batch = v.view(batch_size, seqlen_k, nheads, d)

    dout = torch.randn_like(q) if backward else None

    # --- FA4 with col_limit ---
    def fa4_fwd():
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
            causal=False, col_limit=col_limit,
        )

    def fa4_fwd_bwd():
        out, _ = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
            causal=False, col_limit=col_limit,
        )
        out.backward(dout)
        q.grad = k.grad = v.grad = None

    # --- PyTorch SDPA with materialized mask ---
    def sdpa_fwd():
        for b in range(batch_size):
            materialize_and_sdpa(
                q_batch[b:b+1], k_batch[b:b+1], v_batch[b:b+1],
                col_limit[b*seqlen_q:(b+1)*seqlen_q],
                seqlen_q, seqlen_k,
            )

    def sdpa_fwd_bwd():
        outs = []
        for b in range(batch_size):
            outs.append(materialize_and_sdpa(
                q_batch[b:b+1], k_batch[b:b+1], v_batch[b:b+1],
                col_limit[b*seqlen_q:(b+1)*seqlen_q],
                seqlen_q, seqlen_k,
            ))
        out_cat = torch.cat(outs, dim=0).reshape_as(q)
        out_cat.backward(dout)
        q.grad = k.grad = v.grad = None

    # --- FA4 without col_limit (full non-causal, upper bound) ---
    def fa4_no_mask_fwd():
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k,
            causal=False,
        )

    # Warmup and benchmark
    fa4_fwd()  # compile
    fa4_no_mask_fwd()  # compile
    sdpa_fwd()  # warmup

    mode = "fwd+bwd" if backward else "fwd"

    if backward:
        fa4_fwd_bwd()  # compile bwd
        t_fa4 = bench_fn(fa4_fwd_bwd)
        t_sdpa = bench_fn(sdpa_fwd_bwd)
        t_fa4_nomask = None  # skip for simplicity
    else:
        t_fa4 = bench_fn(fa4_fwd)
        t_sdpa = bench_fn(sdpa_fwd)
        t_fa4_nomask = bench_fn(fa4_no_mask_fwd)

    return t_fa4, t_sdpa, t_fa4_nomask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    configs = [
        # (seqlen_q, seqlen_k, nheads, d, sparsity, batch_size)
        (256, 128, 8, 128, 0.5, 1),
        (512, 256, 8, 128, 0.5, 1),
        (1024, 512, 8, 128, 0.5, 1),
        (2048, 1024, 8, 128, 0.5, 1),
        (1024, 512, 8, 128, 0.5, 4),
        (1024, 512, 8, 128, 0.1, 1),  # very sparse
        (1024, 512, 8, 128, 0.9, 1),  # almost full
    ]

    mode = "fwd+bwd" if args.backward else "fwd"
    print(f"\n{'='*90}")
    print(f"Release Mask Benchmark ({mode})")
    print(f"{'='*90}")
    print(f"{'Config':<40} {'FA4+RM':>10} {'SDPA+mask':>10} {'FA4 full':>10} {'Speedup':>10}")
    print(f"{'':<40} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'vs SDPA':>10}")
    print(f"{'-'*90}")

    for sq, sk, nh, d, sp, bs in configs:
        label = f"q={sq} k={sk} h={nh} d={d} sp={sp} b={bs}"
        try:
            t_fa4, t_sdpa, t_fa4_nomask = run_benchmark(sq, sk, nh, d, sp, bs, backward=args.backward)
            speedup = t_sdpa / t_fa4
            nomask_str = f"{t_fa4_nomask:.3f}" if t_fa4_nomask else "N/A"
            print(f"{label:<40} {t_fa4:>10.3f} {t_sdpa:>10.3f} {nomask_str:>10} {speedup:>9.1f}x")
        except Exception as e:
            print(f"{label:<40} ERROR: {type(e).__name__}")
            torch.cuda.synchronize()  # clear error state

    print(f"{'='*90}")


if __name__ == "__main__":
    main()
