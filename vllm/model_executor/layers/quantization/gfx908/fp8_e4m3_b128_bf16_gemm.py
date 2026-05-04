# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 E4M3 + E8M0 (block-128) -> bf16 dense GEMM for gfx908 (MI100).

DeepSeek-V4-Flash stores its dense weights (attention QKV, output
projection, router gate, and embedding/lm_head when quantized) as FP8
E4M3 (OCP semantics) plus per-128-element E8M0 block scales. gfx908 has
no FP8 MFMA tensor cores, so we keep the weights packed in VRAM and do
the dequantization as a software prologue inside this Triton kernel,
then issue the matrix multiply in bf16 (the natively supported MFMA
type on gfx908).

The kernel computes ``y = x @ w_fp8.T * scale + bias`` where ``scale`` is
broadcast per-128 elements along the inner dimension. ``w_fp8`` has the
OCP E4M3 byte layout — Triton's ``tl.float8_e4m3fn`` matches OCP, not
the FNUZ format used elsewhere on ROCm; we never reinterpret the bytes
through HIP C++ types so OCP semantics are preserved.

The companion scale decoder, :func:`decode_e8m0_block_scales_to_bf16`,
runs **once at load time** to convert the on-disk E8M0 (uint8) byte
``e`` into ``bf16(2 ** (e - 127))``. The bf16 scales are stored
co-located with the weight tensor and read once per K-block by the
kernel, so the per-MFMA-tile overhead is one bf16 broadcast multiply.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op

__all__ = [
    "decode_e8m0_block_scales_to_bf16",
    "fp8_e4m3_b128_bf16_gemm",
]

# Block size of the FP8 E8M0 grouping. DSv4 dense uses 128 along K.
_BLOCK_SCALE_GROUP: int = 128


# ----------------------------------------------------------------------
# Load-time scale decode: uint8 (E8M0) -> bf16
# ----------------------------------------------------------------------


def decode_e8m0_block_scales_to_bf16(
    e8m0_scales: torch.Tensor,
) -> torch.Tensor:
    """Decode E8M0 block scales (one byte each) to bf16 once at load time.

    Args:
        e8m0_scales: ``uint8`` tensor of arbitrary shape. Each byte ``e``
            represents the scale ``2 ** (e - 127)``. ``e == 0xFF`` is the
            spec NaN sentinel and is mapped to bf16 zero with a warning.

    Returns:
        ``bf16`` tensor of the same shape, ready to be co-located with
        the FP8 weight tensor.
    """
    if e8m0_scales.dtype != torch.uint8:
        raise TypeError(
            "decode_e8m0_block_scales_to_bf16 expects uint8 input, "
            f"got {e8m0_scales.dtype}"
        )

    # Promote to int32 to do the (e - 127) arithmetic without overflow,
    # then build the float value as 2 ** exp via ldexp on a tensor of
    # ones. ldexp accepts a float-tensor mantissa and an int-tensor
    # exponent and returns a float tensor with the right binary value.
    exp = e8m0_scales.to(torch.int32) - 127
    ones = torch.ones_like(exp, dtype=torch.float32)
    scales_fp32 = torch.ldexp(ones, exp)

    # E8M0 == 0xFF -> NaN per OCP. Real DSv4 weights should never trip
    # this; if they do, we substitute 0 and the model will produce a
    # deterministic dead-channel output rather than NaN-poisoning the
    # whole forward pass.
    nan_mask = e8m0_scales == 0xFF
    if nan_mask.any():
        scales_fp32 = torch.where(
            nan_mask, torch.zeros_like(scales_fp32), scales_fp32
        )

    return scales_fp32.to(torch.bfloat16)


# ----------------------------------------------------------------------
# Triton kernel
# ----------------------------------------------------------------------
# BLOCK_K is fixed at 128 to align with the E8M0 scale group: one scalar
# scale per [N, K=128] tile of W. Choosing a BLOCK_K that divides 128
# evenly is a hard requirement of the math; making it equal to the group
# size keeps the per-tile bookkeeping minimal.
_BLOCK_K_DEFAULT: int = 128


@triton.jit
def _fp8_b128_bf16_gemm_kernel(
    X_ptr,
    W_ptr,
    S_ptr,
    Y_ptr,
    B_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sg,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_GROUP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    num_k_iters = tl.cdiv(K, BLOCK_K)

    for k_iter in range(0, num_k_iters):
        k_base = k_iter * BLOCK_K
        mask_k = (k_base + offs_k) < K

        x_mask = mask_m[:, None] & mask_k[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]

        # Load X tile (bf16). Out-of-bounds elements zero out so they
        # contribute nothing to the dot product.
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load FP8 weight tile and convert to bf16. On gfx908 this is a
        # software cast lowered by Triton; the cast happens before the
        # MFMA so the dot product sees bf16 inputs.
        w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        w_bf16 = w_fp8.to(tl.bfloat16)

        # One bf16 scale per (BLOCK_N, BLOCK_K=SCALE_GROUP) tile.
        sg = k_iter * (BLOCK_K // SCALE_GROUP)
        s_ptrs = S_ptr + offs_n * stride_sn + sg * stride_sg
        scale = tl.load(s_ptrs, mask=mask_n, other=0.0).to(tl.bfloat16)

        w_bf16 = w_bf16 * scale[:, None]

        # MFMA in bf16; transpose W tile so its K dim aligns with X's K.
        acc += tl.dot(x, tl.trans(w_bf16), out_dtype=tl.float32)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=y_mask)


# Default tile sizes chosen for gfx908: 64-thread wavefronts, 64 KB LDS.
# BLOCK_M * BLOCK_N / num_warps must be a multiple of 64; (64, 128, 128)
# / 4 warps = 2048 elements per warp, which lowers cleanly to
# v_mfma_f32_16x16x16_bf16. num_stages=2 to fit in LDS without spills
# (PR #40860 already gates num_stages=3 off for ROCm; we follow suit).
_DEFAULT_BLOCK_M: int = 64
_DEFAULT_BLOCK_N: int = 128
_DEFAULT_NUM_WARPS: int = 4
_DEFAULT_NUM_STAGES: int = 2


def _fp8_b128_bf16_gemm_impl(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scales_bf16: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Concrete PyTorch entry point for the FP8 dense GEMM.

    Inputs follow the contract documented in :func:`fp8_e4m3_b128_bf16_gemm`.
    """
    if x.dim() == 2:
        x_2d = x
    else:
        x_2d = x.reshape(-1, x.shape[-1])

    M, K = x_2d.shape
    N, K_w = w_fp8.shape
    if K != K_w:
        raise ValueError(
            f"K mismatch: x has K={K}, w_fp8 has K={K_w}"
        )

    if w_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "w_fp8 must be torch.float8_e4m3fn (OCP E4M3 byte layout); "
            f"got {w_fp8.dtype}"
        )
    if x_2d.dtype != torch.bfloat16:
        raise TypeError(f"x must be bfloat16; got {x_2d.dtype}")
    if w_scales_bf16.dtype != torch.bfloat16:
        raise TypeError(
            "w_scales_bf16 must be bfloat16 (pre-decoded from E8M0); "
            f"got {w_scales_bf16.dtype}"
        )

    expected_scale_groups = (K + _BLOCK_SCALE_GROUP - 1) // _BLOCK_SCALE_GROUP
    if w_scales_bf16.shape != (N, expected_scale_groups):
        raise ValueError(
            f"w_scales_bf16 shape mismatch: expected ({N}, "
            f"{expected_scale_groups}); got {tuple(w_scales_bf16.shape)}"
        )

    x_2d = x_2d.contiguous()
    w_fp8 = w_fp8.contiguous()
    w_scales_bf16 = w_scales_bf16.contiguous()

    y = torch.empty((M, N), dtype=torch.bfloat16, device=x_2d.device)

    has_bias = bias is not None
    if has_bias:
        if bias.dtype != torch.bfloat16:
            raise TypeError(f"bias must be bfloat16; got {bias.dtype}")
        if bias.shape != (N,):
            raise ValueError(
                f"bias shape mismatch: expected ({N},); got {tuple(bias.shape)}"
            )
        bias = bias.contiguous()
        bias_ptr = bias
    else:
        # Triton needs a non-None pointer even when HAS_BIAS=False; pass
        # the output buffer's pointer (it is never read in that branch).
        bias_ptr = y

    grid = (
        triton.cdiv(M, _DEFAULT_BLOCK_M),
        triton.cdiv(N, _DEFAULT_BLOCK_N),
    )

    _fp8_b128_bf16_gemm_kernel[grid](
        x_2d,
        w_fp8,
        w_scales_bf16,
        y,
        bias_ptr,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        w_fp8.stride(0),
        w_fp8.stride(1),
        w_scales_bf16.stride(0),
        w_scales_bf16.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=_DEFAULT_BLOCK_M,
        BLOCK_N=_DEFAULT_BLOCK_N,
        BLOCK_K=_BLOCK_K_DEFAULT,
        SCALE_GROUP=_BLOCK_SCALE_GROUP,
        HAS_BIAS=has_bias,
        num_warps=_DEFAULT_NUM_WARPS,
        num_stages=_DEFAULT_NUM_STAGES,
    )

    if x.dim() != 2:
        y = y.reshape(*x.shape[:-1], N)
    return y


def _fp8_b128_bf16_gemm_fake(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scales_bf16: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    out_shape = (*x.shape[:-1], w_fp8.shape[0])
    return torch.empty(out_shape, dtype=torch.bfloat16, device=x.device)


# Public entry point. Dispatch sites should call this rather than the
# kernel directly so the ``torch.library`` fake impl participates in
# Dynamo / torch.compile shape inference.
fp8_e4m3_b128_bf16_gemm = _fp8_b128_bf16_gemm_impl


direct_register_custom_op(
    op_name="fp8_e4m3_b128_bf16_gemm_gfx908",
    op_func=_fp8_b128_bf16_gemm_impl,
    mutates_args=[],
    fake_impl=_fp8_b128_bf16_gemm_fake,
)
