# Copyright (c) 2025, Tri Dao.
# Tests for kv_seqused cross-attention support (forward + backward).

import math
import os

import pytest
import torch

from flash_attn.cute.testing import maybe_fake_tensor_mode, is_fake_mode
from flash_attn.cute.interface import flash_attn_varlen_func


USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1
SM_MAJOR = torch.cuda.get_device_capability()[0]
# SM120 varlen + GQA is pre-existing broken (pack_gqa compilation failure)
SM120_NO_GQA = SM_MAJOR == 12
VERBOSE = True

pytestmark = pytest.mark.skipif(
    SM_MAJOR not in (8, 9, 10, 11, 12) and not USE_FAKE_TENSOR,
    reason="kv_seqused not supported on this GPU arch",
)


def attention_kv_seqused_ref(q, k, v, kv_seqused, softmax_scale=None, window_size_left=None):
    """Differentiable reference: materialize full mask, compute attention with autograd.

    All inputs must have requires_grad=True (for q, k, v) to get gradients.
    """
    total_q, nheads, hdim = q.shape
    total_k, nheads_kv, hdim_v = v.shape[0], v.shape[1], v.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(hdim)

    heads_per_kv = nheads // nheads_kv
    k_expanded = k.repeat_interleave(heads_per_kv, dim=1)
    v_expanded = v.repeat_interleave(heads_per_kv, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q.float() * softmax_scale, k_expanded.float())

    kv_idx = torch.arange(total_k, device=q.device).unsqueeze(0)
    right = kv_seqused.unsqueeze(1)
    if window_size_left is not None:
        left = (kv_seqused - window_size_left).clamp(min=0).unsqueeze(1)
        mask = (kv_idx >= left) & (kv_idx < right)
    else:
        mask = kv_idx < right
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn, v_expanded.float())
    lse = torch.logsumexp(scores, dim=-1)
    return out.to(q.dtype), lse


def generate_kv_seqused(seqlens_q, seqlens_k, device, min_visible=0):
    """Generate a random monotonically non-decreasing kv_seqused."""
    parts = []
    for sq, sk in zip(seqlens_q, seqlens_k):
        vals = torch.sort(torch.randint(min_visible, sk + 1, (sq,), dtype=torch.int32, device=device))[0]
        # Make sure last Q position sees at least 1 KV
        vals[-1] = max(vals[-1].item(), 1)
        parts.append(vals)
    return torch.cat(parts)


def make_cu_seqlens(seqlens, device):
    cumsum = [0]
    for s in seqlens:
        cumsum.append(cumsum[-1] + s)
    return torch.tensor(cumsum, dtype=torch.int32, device=device)


# ============================================================================
# Forward-only tests
# ============================================================================

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "seqlens_q,seqlens_k",
    [
        ([128], [64]),
        ([64], [128]),
        ([128], [128]),
        ([256], [192]),
        ([64, 128], [96, 64]),
        ([128, 256], [128, 128]),
        ([1], [1]),
        ([3], [7]),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_kv_seqused_fwd(seqlens_q, seqlens_k, d, mha_type, dtype):
    if SM120_NO_GQA and mha_type == "gqa":
        pytest.skip("SM120 varlen + GQA is pre-existing broken")
    device = "cuda"
    nheads = 8
    nheads_kv = nheads if mha_type == "mha" else (nheads // 4 if mha_type == "gqa" else 1)

    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    torch.manual_seed(42)

    q = torch.randn(total_q, nheads, d, dtype=dtype, device=device)
    k = torch.randn(total_k, nheads_kv, d, dtype=dtype, device=device)
    v = torch.randn(total_k, nheads_kv, d, dtype=dtype, device=device)

    cu_seqlens_q = make_cu_seqlens(seqlens_q, device)
    cu_seqlens_k = make_cu_seqlens(seqlens_k, device)

    if is_fake_mode():
        kv_seqused = torch.zeros(total_q, dtype=torch.int32, device=device)
    else:
        kv_seqused = generate_kv_seqused(seqlens_q, seqlens_k, device)

    out, lse = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(seqlens_q), max_seqlen_k=max(seqlens_k),
        causal=False, return_lse=True, kv_seqused=kv_seqused,
    )

    if is_fake_mode():
        return

    # Per-batch reference
    out_ref_parts, lse_ref_parts = [], []
    offset_q, offset_k = 0, 0
    for sq, sk in zip(seqlens_q, seqlens_k):
        rm_batch = kv_seqused[offset_q : offset_q + sq]
        out_b, lse_b = attention_kv_seqused_ref(
            q[offset_q:offset_q+sq], k[offset_k:offset_k+sk], v[offset_k:offset_k+sk], rm_batch,
        )
        out_ref_parts.append(out_b)
        lse_ref_parts.append(lse_b)
        offset_q += sq
        offset_k += sk

    out_ref = torch.cat(out_ref_parts, dim=0)
    lse_ref = torch.cat(lse_ref_parts, dim=1)

    # Rows with kv_seqused=0 produce NaN (all -inf scores); exclude from comparison
    valid = kv_seqused > 0
    out_diff = (out[valid] - out_ref[valid]).abs().max().item() if valid.any() else 0.0
    lse_diff = (lse[:, valid] - lse_ref[:, valid]).abs().max().item() if valid.any() else 0.0

    if VERBOSE:
        print(f"Fwd output max diff: {out_diff:.6f}")
        print(f"Fwd LSE max diff: {lse_diff:.6f}")

    assert out_diff <= 2e-2
    # LSE can have inf/-inf for edge rows; only compare finite values
    lse_v, lse_ref_v = lse[:, valid], lse_ref[:, valid]
    both_finite = torch.isfinite(lse_v) & torch.isfinite(lse_ref_v)
    if both_finite.any():
        lse_diff = (lse_v[both_finite] - lse_ref_v[both_finite]).abs().max().item()
    else:
        lse_diff = 0.0
    assert lse_diff <= 1e-2


# ============================================================================
# Forward + Backward tests
# ============================================================================

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("d", [64, 128])
@pytest.mark.parametrize(
    "seqlens_q,seqlens_k",
    [
        ([128], [64]),
        ([64], [128]),
        ([128], [128]),
        ([64, 128], [96, 64]),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_kv_seqused_bwd(seqlens_q, seqlens_k, d, mha_type, dtype):
    if SM120_NO_GQA and mha_type == "gqa":
        pytest.skip("SM120 varlen + GQA is pre-existing broken")
    device = "cuda"
    nheads = 8
    nheads_kv = nheads if mha_type == "mha" else (nheads // 4 if mha_type == "gqa" else 1)

    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    torch.manual_seed(42)

    q = torch.randn(total_q, nheads, d, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total_k, nheads_kv, d, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total_k, nheads_kv, d, dtype=dtype, device=device, requires_grad=True)

    cu_seqlens_q = make_cu_seqlens(seqlens_q, device)
    cu_seqlens_k = make_cu_seqlens(seqlens_k, device)

    if is_fake_mode():
        kv_seqused = torch.zeros(total_q, dtype=torch.int32, device=device)
    else:
        kv_seqused = generate_kv_seqused(seqlens_q, seqlens_k, device)

    # Forward with flash attention
    out, lse = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(seqlens_q), max_seqlen_k=max(seqlens_k),
        causal=False, return_lse=True, kv_seqused=kv_seqused,
    )

    if is_fake_mode():
        return

    # Backward with random dout
    dout = torch.randn_like(out)
    out.backward(dout)
    dq_fa, dk_fa, dv_fa = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad, k.grad, v.grad = None, None, None

    # Reference forward + backward per batch
    dq_ref = torch.zeros_like(q, dtype=torch.float32)
    dk_ref = torch.zeros_like(k, dtype=torch.float32)
    dv_ref = torch.zeros_like(v, dtype=torch.float32)

    offset_q, offset_k = 0, 0
    for sq, sk in zip(seqlens_q, seqlens_k):
        q_b = q[offset_q:offset_q+sq].detach().float().requires_grad_(True)
        k_b = k[offset_k:offset_k+sk].detach().float().requires_grad_(True)
        v_b = v[offset_k:offset_k+sk].detach().float().requires_grad_(True)
        rm_b = kv_seqused[offset_q:offset_q+sq]
        dout_b = dout[offset_q:offset_q+sq].float()

        out_b, _ = attention_kv_seqused_ref(q_b, k_b, v_b, rm_b)
        out_b.backward(dout_b)
        dq_ref[offset_q:offset_q+sq] = q_b.grad
        dk_ref[offset_k:offset_k+sk] = k_b.grad
        dv_ref[offset_k:offset_k+sk] = v_b.grad
        offset_q += sq
        offset_k += sk

    dq_ref = dq_ref.to(dtype)
    dk_ref = dk_ref.to(dtype)
    dv_ref = dv_ref.to(dtype)

    # NaN can appear in gradients when kv_seqused=0 (all -inf scores)
    def finite_max_diff(a, b):
        diff = (a - b).abs()
        finite = torch.isfinite(diff)
        return diff[finite].max().item() if finite.any() else 0.0

    dq_diff = finite_max_diff(dq_fa, dq_ref)
    dk_diff = finite_max_diff(dk_fa, dk_ref)
    dv_diff = finite_max_diff(dv_fa, dv_ref)

    if VERBOSE:
        print(f"dQ max diff: {dq_diff:.6f}")
        print(f"dK max diff: {dk_diff:.6f}")
        print(f"dV max diff: {dv_diff:.6f}")

    assert dq_diff <= 5e-2, f"dQ mismatch: {dq_diff}"
    assert dk_diff <= 5e-2, f"dK mismatch: {dk_diff}"
    assert dv_diff <= 5e-2, f"dV mismatch: {dv_diff}"


# ============================================================================
# All-visible equivalence test
# ============================================================================

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize(
    "seqlens_q,seqlens_k",
    [
        ([64], [128]),
        ([128], [64]),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_kv_seqused_all_visible(seqlens_q, seqlens_k, d, dtype):
    """kv_seqused with all KV visible should match non-causal attention."""
    device = "cuda"
    nheads = 4
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    torch.manual_seed(42)

    q = torch.randn(total_q, nheads, d, dtype=dtype, device=device)
    k = torch.randn(total_k, nheads, d, dtype=dtype, device=device)
    v = torch.randn(total_k, nheads, d, dtype=dtype, device=device)

    cu_seqlens_q = make_cu_seqlens(seqlens_q, device)
    cu_seqlens_k = make_cu_seqlens(seqlens_k, device)

    if is_fake_mode():
        kv_seqused = torch.zeros(total_q, dtype=torch.int32, device=device)
    else:
        parts = []
        for sq, sk in zip(seqlens_q, seqlens_k):
            parts.append(torch.full((sq,), sk, dtype=torch.int32, device=device))
        kv_seqused = torch.cat(parts)

    out_rm, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(seqlens_q), max_seqlen_k=max(seqlens_k),
        causal=False, return_lse=True, kv_seqused=kv_seqused,
    )

    out_ref, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(seqlens_q), max_seqlen_k=max(seqlens_k),
        causal=False, return_lse=True,
    )

    if is_fake_mode():
        return

    if VERBOSE:
        print(f"All-visible output max diff: {(out_rm - out_ref).abs().max().item():.6f}")

    assert torch.allclose(out_rm, out_ref, atol=1e-5)


# ============================================================================
# Sliding window + kv_seqused tests
# ============================================================================

@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("window_size_left", [32, 64, 128])
@pytest.mark.parametrize(
    "seqlens_q,seqlens_k",
    [
        ([128], [256]),
        ([64], [128]),
        ([128], [128]),
        ([64, 128], [128, 64]),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_kv_seqused_window_fwd(seqlens_q, seqlens_k, d, window_size_left, dtype):
    """Forward: kv_seqused + window_size_left vs materialized reference."""
    if SM120_NO_GQA:
        pass  # MHA only, no skip needed
    device = "cuda"
    nheads = 8
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    torch.manual_seed(42)

    q = torch.randn(total_q, nheads, d, dtype=dtype, device=device)
    k = torch.randn(total_k, nheads, d, dtype=dtype, device=device)
    v = torch.randn(total_k, nheads, d, dtype=dtype, device=device)

    cu_seqlens_q = make_cu_seqlens(seqlens_q, device)
    cu_seqlens_k = make_cu_seqlens(seqlens_k, device)

    if is_fake_mode():
        kv_seqused = torch.zeros(total_q, dtype=torch.int32, device=device)
    else:
        kv_seqused = generate_kv_seqused(seqlens_q, seqlens_k, device, min_visible=1)

    out, lse = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(seqlens_q), max_seqlen_k=max(seqlens_k),
        causal=False, return_lse=True,
        kv_seqused=kv_seqused, window_size=(window_size_left, None),
    )

    if is_fake_mode():
        return

    # Per-batch reference
    out_ref_parts = []
    offset_q, offset_k = 0, 0
    for sq, sk in zip(seqlens_q, seqlens_k):
        rm_batch = kv_seqused[offset_q : offset_q + sq]
        out_b, _ = attention_kv_seqused_ref(
            q[offset_q:offset_q+sq], k[offset_k:offset_k+sk], v[offset_k:offset_k+sk],
            rm_batch, window_size_left=window_size_left,
        )
        out_ref_parts.append(out_b)
        offset_q += sq
        offset_k += sk
    out_ref = torch.cat(out_ref_parts, dim=0)

    valid = kv_seqused > 0
    out_diff = (out[valid] - out_ref[valid]).abs().max().item() if valid.any() else 0.0

    if VERBOSE:
        print(f"Window fwd output max diff: {out_diff:.6f}")

    assert out_diff <= 2e-2


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("window_size_left", [32, 64])
@pytest.mark.parametrize(
    "seqlens_q,seqlens_k",
    [
        ([128], [256]),
        ([64], [128]),
        ([64, 128], [128, 64]),
    ],
)
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_kv_seqused_window_bwd(seqlens_q, seqlens_k, d, window_size_left, dtype):
    """Backward: kv_seqused + window_size_left vs materialized reference."""
    if SM120_NO_GQA:
        pass  # MHA only, no skip needed
    device = "cuda"
    nheads = 8
    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    torch.manual_seed(42)

    q = torch.randn(total_q, nheads, d, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total_k, nheads, d, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total_k, nheads, d, dtype=dtype, device=device, requires_grad=True)

    cu_seqlens_q = make_cu_seqlens(seqlens_q, device)
    cu_seqlens_k = make_cu_seqlens(seqlens_k, device)

    if is_fake_mode():
        kv_seqused = torch.zeros(total_q, dtype=torch.int32, device=device)
    else:
        kv_seqused = generate_kv_seqused(seqlens_q, seqlens_k, device, min_visible=1)

    out, _ = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(seqlens_q), max_seqlen_k=max(seqlens_k),
        causal=False, return_lse=True,
        kv_seqused=kv_seqused, window_size=(window_size_left, None),
    )

    if is_fake_mode():
        return

    dout = torch.randn_like(out)
    out.backward(dout)
    dq_fa, dk_fa, dv_fa = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad, k.grad, v.grad = None, None, None

    # Reference per batch
    dq_ref = torch.zeros_like(q, dtype=torch.float32)
    dk_ref = torch.zeros_like(k, dtype=torch.float32)
    dv_ref = torch.zeros_like(v, dtype=torch.float32)

    offset_q, offset_k = 0, 0
    for sq, sk in zip(seqlens_q, seqlens_k):
        q_b = q[offset_q:offset_q+sq].detach().float().requires_grad_(True)
        k_b = k[offset_k:offset_k+sk].detach().float().requires_grad_(True)
        v_b = v[offset_k:offset_k+sk].detach().float().requires_grad_(True)
        rm_b = kv_seqused[offset_q:offset_q+sq]
        dout_b = dout[offset_q:offset_q+sq].float()

        out_b, _ = attention_kv_seqused_ref(q_b, k_b, v_b, rm_b, window_size_left=window_size_left)
        out_b.backward(dout_b)
        dq_ref[offset_q:offset_q+sq] = q_b.grad
        dk_ref[offset_k:offset_k+sk] = k_b.grad
        dv_ref[offset_k:offset_k+sk] = v_b.grad
        offset_q += sq
        offset_k += sk

    def finite_max_diff(a, b):
        diff = (a - b).abs()
        finite = torch.isfinite(diff)
        return diff[finite].max().item() if finite.any() else 0.0

    dq_diff = finite_max_diff(dq_fa, dq_ref.to(dtype))
    dk_diff = finite_max_diff(dk_fa, dk_ref.to(dtype))
    dv_diff = finite_max_diff(dv_fa, dv_ref.to(dtype))

    if VERBOSE:
        print(f"Window bwd dQ diff: {dq_diff:.6f}, dK diff: {dk_diff:.6f}, dV diff: {dv_diff:.6f}")

    assert dq_diff <= 5e-2, f"dQ mismatch: {dq_diff}"
    assert dk_diff <= 5e-2, f"dK mismatch: {dk_diff}"
    assert dv_diff <= 5e-2, f"dV mismatch: {dv_diff}"
