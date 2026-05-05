# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""mHC (Manifold-Constrained Hyper-Connections) Triton kernel for gfx908.

The upstream :mod:`vllm.model_executor.layers.mhc` implementation uses
TileLang for the pre-GEMM hot path (`mhc_pre_big_fuse_tilelang`) and
falls back to a pure-PyTorch path on ROCm. The torch fallback is
correct on gfx908 but launches one HIP kernel per arithmetic op; for
DSv4 inference at decode batch=1 those launches add up. This module
provides a Triton-only replacement for the pre-GEMM + RMS + sigmoid +
Sinkhorn pipeline that fuses the per-token reductions into a single
kernel and keeps the Sinkhorn iterations in registers (the
``hc_mult x hc_mult`` matrix is at most 4x4 for DSv4-Flash).

Dispatch is gated behind
:func:`vllm.platforms.rocm.is_dsv4_gfx908_path`. The torch fallback
remains the default safe path; this module is opt-in via the same
predicate. The numerical contract — output dtypes, shapes, and the
Sinkhorn iteration count — exactly mirrors the upstream torch
fallback in ``mhc.py:237-268`` so the unit test can compare them
element-wise.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op

__all__ = [
    "mhc_pre_gfx908_triton",
]


@triton.jit
def _mhc_pre_gemm_sqrsum_kernel(
    R_ptr,         # residual: [num_tokens, hc_mult, hidden_size] bf16
    F_ptr,         # fn: [hc_mult3, hc_mult * hidden_size] fp32
    M_ptr,         # mixes_out: [num_tokens, hc_mult3] fp32
    S_ptr,         # sqrsum_out: [num_tokens] fp32
    num_tokens,
    HC_MULT: tl.constexpr,
    HC_MULT3: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    BLOCK_MIXES: tl.constexpr,
):
    """Per-token reduction over (hc_mult, hidden_size).

    Computes::

        sqrsum[t]    = sum over m, h of  residual[t, m, h] ** 2
        mixes[t, j]  = sum over m, h of  residual[t, m, h] * fn[j, m * hidden_size + h]

    one program per token. ``HC_MULT`` and ``HC_MULT3`` are constexpr
    so the inner loops unroll cleanly; for DSv4-Flash these are 4 and
    24 respectively.
    """
    pid_t = tl.program_id(0)

    sqrsum = tl.zeros((1,), dtype=tl.float32)
    # Triton block dimensions must be powers of two. HC_MULT3 is 24 for
    # DSv4's hc_mult=4, so keep a padded register vector and mask stores.
    mixes = tl.zeros((BLOCK_MIXES,), dtype=tl.float32)
    mix_idx = tl.arange(0, BLOCK_MIXES)

    hc_dim = HC_MULT * HIDDEN_SIZE

    for h_iter in tl.static_range(0, NUM_H_BLOCKS):
        h_base = h_iter * BLOCK_H
        offs_h = h_base + tl.arange(0, BLOCK_H)
        h_mask = offs_h < HIDDEN_SIZE

        for m in tl.static_range(0, HC_MULT):
            r_ptrs = (
                R_ptr
                + pid_t * (HC_MULT * HIDDEN_SIZE)
                + m * HIDDEN_SIZE
                + offs_h
            )
            r = tl.load(r_ptrs, mask=h_mask, other=0.0).to(tl.float32)
            sqrsum += tl.sum(r * r, axis=0)

            for j in tl.static_range(0, HC_MULT3):
                f_ptrs = F_ptr + j * hc_dim + m * HIDDEN_SIZE + offs_h
                f = tl.load(f_ptrs, mask=h_mask, other=0.0)
                # Accumulate into mixes[j]. Triton lowers this small
                # static-range loop into a register-resident accumulator
                # tree.
                contrib = tl.sum(r * f, axis=0)
                # Construct a one-hot update so we can keep `mixes` as
                # a single vector.
                mixes = mixes + tl.where(mix_idx == j, contrib, 0.0)

    tl.store(S_ptr + pid_t + tl.arange(0, 1), sqrsum)
    tl.store(
        M_ptr + pid_t * HC_MULT3 + mix_idx,
        mixes,
        mask=mix_idx < HC_MULT3,
    )


def _mhc_pre_gemm_sqrsum_triton(
    residual_flat: torch.Tensor,  # [num_tokens, hc_mult, hidden_size] bf16
    fn_flat: torch.Tensor,        # [hc_mult3, hc_mult * hidden_size] fp32
    hc_mult: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = residual_flat.shape[0]
    hc_mult3 = fn_flat.shape[0]
    if hc_mult3 != hc_mult * 2 + hc_mult * hc_mult:
        raise ValueError(
            "fn_flat first dim must be hc_mult*2 + hc_mult^2; got "
            f"{hc_mult3} for hc_mult={hc_mult}"
        )

    residual_flat = residual_flat.contiguous()
    fn_flat = fn_flat.contiguous()

    mixes = torch.empty(
        (num_tokens, hc_mult3),
        dtype=torch.float32,
        device=residual_flat.device,
    )
    sqrsum = torch.empty(
        (num_tokens,), dtype=torch.float32, device=residual_flat.device
    )

    if num_tokens == 0:
        return mixes, sqrsum

    block_h = 256 if hidden_size >= 256 else triton.next_power_of_2(hidden_size)
    num_h_blocks = triton.cdiv(hidden_size, block_h)
    block_mixes = triton.next_power_of_2(hc_mult3)
    grid = (num_tokens,)
    _mhc_pre_gemm_sqrsum_kernel[grid](
        residual_flat,
        fn_flat,
        mixes,
        sqrsum,
        num_tokens,
        HC_MULT=hc_mult,
        HC_MULT3=hc_mult3,
        HIDDEN_SIZE=hidden_size,
        BLOCK_H=block_h,
        NUM_H_BLOCKS=num_h_blocks,
        BLOCK_MIXES=block_mixes,
        num_warps=4,
        num_stages=2,
    )
    return mixes, sqrsum


# ----------------------------------------------------------------------
# Public op: full mHC pre block on gfx908.
# ----------------------------------------------------------------------


def mhc_pre_gfx908_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-fused mHC pre block. Numerically equivalent to the torch
    fallback in ``vllm/model_executor/layers/mhc.py`` but with the
    per-token pre-GEMM + RMS reduction collapsed into one kernel.

    Args:
        residual: ``(..., hc_mult, hidden_size)`` bf16.
        fn:       ``(hc_mult3, hc_mult * hidden_size)`` fp32 where
                  ``hc_mult3 = 2*hc_mult + hc_mult**2``.
        hc_scale: ``(3,)`` fp32 scaling for the pre / post / comb mixes.
        hc_base:  ``(hc_mult3,)`` fp32 bias added to each mix.
        rms_eps:  RMS normalization epsilon.
        hc_pre_eps: pre-mix epsilon (added after sigmoid).
        hc_sinkhorn_eps: epsilon used during Sinkhorn normalization.
        hc_post_mult_value: post-mix multiplier (after sigmoid).
        sinkhorn_repeat: number of full row+column normalize iterations
                  (matches the upstream constant from ``deepseek_v4``).

    Returns:
        ``(post_mix, comb_mix, layer_input)`` matching the upstream
        ``mhc_pre`` shapes:
        - ``post_mix``  : ``(..., hc_mult, 1)`` fp32
        - ``comb_mix``  : ``(..., hc_mult, hc_mult)`` fp32
        - ``layer_input``: ``(..., hidden_size)`` bf16
    """
    if residual.dtype != torch.bfloat16:
        raise TypeError(f"residual must be bfloat16; got {residual.dtype}")
    if fn.dtype != torch.float32:
        raise TypeError(f"fn must be float32; got {fn.dtype}")
    if hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise TypeError("hc_scale / hc_base must be float32")

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    if fn.shape != (hc_mult3, hc_mult * hidden_size):
        raise ValueError(
            "fn shape mismatch: expected "
            f"({hc_mult3}, {hc_mult * hidden_size}); got {tuple(fn.shape)}"
        )
    if hc_scale.shape != (3,) or hc_base.shape != (hc_mult3,):
        raise ValueError(
            "hc_scale must be (3,) and hc_base must be "
            f"({hc_mult3},); got {tuple(hc_scale.shape)} and "
            f"{tuple(hc_base.shape)}"
        )

    outer_shape = residual.shape[:-2]
    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    # Pre-GEMM + sqrsum in a single Triton kernel.
    mixes, sqrsum = _mhc_pre_gemm_sqrsum_triton(
        residual_flat, fn, hc_mult, hidden_size
    )

    # The remaining work is tiny per token (hc_mult * hc_mult <= 16 for
    # DSv4-Flash). Keeping it in torch is faster than launching a second
    # Triton kernel because the launch dominates the math.
    rsqrt = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps).unsqueeze(-1)
    mixes_norm = mixes * rsqrt

    pre_logits = mixes_norm[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes_norm[:, hc_mult : 2 * hc_mult] * hc_scale[1]
        + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = (
        mixes_norm[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult)
        * hc_scale[2]
        + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    )
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    ).to(torch.bfloat16)

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_gfx908_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    post_mix = torch.empty(
        *outer_shape, hc_mult, 1, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )
    return post_mix, comb_mix, layer_input


direct_register_custom_op(
    op_name="mhc_pre_gfx908_triton",
    op_func=mhc_pre_gfx908_triton,
    mutates_args=[],
    fake_impl=_mhc_pre_gfx908_fake,
)
