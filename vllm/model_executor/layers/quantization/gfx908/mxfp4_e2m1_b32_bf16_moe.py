# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 E2M1 + E8M0 (block-32) -> bf16 fused MoE GEMM for gfx908.

DSv4-Flash stores the 256 routed-expert up/gate/down projections in
OCP MXFP4: each weight element is a 4-bit E2M1 value packed two-per-byte
in ``uint8``, with one E8M0 scale per 32 elements along the inner
(``K``) dimension. gfx908 has no FP4 / FP8 MFMA, so we keep the packed
weights in VRAM and dequantize inline before the bf16 MFMA dot.

The kernel here is the ``gemm_per_expert`` step that PR #40871 routes
through the ``triton_unfused`` MoE backend. We replace **only** the
inner GEMM; the surrounding ``select_experts -> permute ->
gemm_per_expert -> unpermute -> reduce_topk_weights`` topology stays
unchanged so vLLM's existing routing and capacity scheduling work as-is.
The dispatch site is gated behind
``vllm.platforms.rocm.is_dsv4_gfx908_path``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op

__all__ = [
    "mxfp4_e2m1_b32_bf16_fused_moe",
]

_BLOCK_SCALE_GROUP: int = 32  # E8M0 scale group for MXFP4

# E2M1 codepoint LUT — sign in bit 3, 2-bit exponent in bits 1-2,
# 1-bit mantissa in bit 0. The table mirrors the OCP MX spec.
_E2M1_LUT: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@triton.jit
def _decode_e2m1_nibble(nib):
    """Decode a 4-bit E2M1 nibble (0-15) into a bf16 value."""
    # Sign and magnitude reconstruction without a true LUT (Triton has
    # no LUT primitive; arithmetic decode lowers to a tree of selects
    # that the compiler keeps in registers).
    sign_bit = (nib >> 3) & 0x1
    exp = (nib >> 1) & 0x3
    mant = nib & 0x1

    # Subnormal at exp == 0: value = mant * 0.5 (== 0.0 or 0.5).
    # Normal otherwise: value = (1 + 0.5 * mant) * 2 ** (exp - 1).
    is_sub = exp == 0
    sub_val = mant.to(tl.float32) * 0.5
    norm_val = (1.0 + 0.5 * mant.to(tl.float32)) * tl.exp2(
        (exp - 1).to(tl.float32)
    )
    mag = tl.where(is_sub, sub_val, norm_val)
    sign = tl.where(sign_bit == 0, 1.0, -1.0)
    return (sign * mag).to(tl.bfloat16)


@triton.jit
def _mxfp4_b32_bf16_moe_per_expert_kernel(
    X_ptr,          # [num_tokens_e, K] bf16
    W_ptr,          # [N, K // 2] uint8 (packed E2M1 nibbles)
    S_ptr,          # [N, K // 32] bf16 (decoded E8M0 scales)
    Y_ptr,          # [num_tokens_e, N] bf16 output
    M,              # int32: tokens routed to this expert
    N,              # int32
    K,              # int32 (must be a multiple of 32; W_ptr stores K/2 bytes)
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wkb,     # stride along packed-byte K dim
    stride_sn,
    stride_sg,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,        # multiple of 32 along K (logical)
    SCALE_GROUP: tl.constexpr,    # = 32
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # K-dim offsets are in *logical elements*. The packed buffer uses
    # K/2 bytes; we compute the byte index as logical_k // 2.
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    num_k_iters = tl.cdiv(K, BLOCK_K)

    for k_iter in range(0, num_k_iters):
        k_base = k_iter * BLOCK_K
        logical_k = k_base + offs_k
        mask_k = logical_k < K

        # X tile: contiguous along K, bf16.
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + logical_k[None, :] * stride_xk
        x = tl.load(
            x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)

        # W packed bytes: byte_idx = logical_k // 2; nibble_lo when even,
        # nibble_hi when odd. We load bytes once and decode both nibbles
        # so the kernel issues exactly one byte load per two K elements.
        byte_idx = logical_k // 2
        is_hi = (logical_k & 1) == 1
        w_byte_ptrs = (
            W_ptr + offs_n[:, None] * stride_wn + byte_idx[None, :] * stride_wkb
        )
        # Each byte is shared between two adjacent K positions. We load
        # the same byte twice (once for the lo lane, once for the hi
        # lane) for clarity; the compiler CSE's the address arithmetic.
        w_bytes = tl.load(
            w_byte_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0
        )
        nib_lo = w_bytes & 0xF
        nib_hi = (w_bytes >> 4) & 0xF
        w_decoded_lo = _decode_e2m1_nibble(nib_lo)
        w_decoded_hi = _decode_e2m1_nibble(nib_hi)
        w_bf16 = tl.where(is_hi[None, :], w_decoded_hi, w_decoded_lo)

        # Scale: one bf16 per 32-element group. We pick BLOCK_K so it is
        # a multiple of 32 — there are BLOCK_K // 32 groups per K tile.
        # For correctness we allow BLOCK_K > 32 by indexing the scale
        # tensor at logical_k // 32 and broadcasting per-element.
        scale_idx = logical_k // SCALE_GROUP
        s_ptrs = S_ptr + offs_n[:, None] * stride_sn + scale_idx[None, :] * stride_sg
        scale = tl.load(
            s_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0
        ).to(tl.bfloat16)
        w_bf16 = w_bf16 * scale

        # Dot product in bf16 (MFMA on gfx908).
        acc += tl.dot(x, tl.trans(w_bf16), out_dtype=tl.float32)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc.to(tl.bfloat16), mask=y_mask)


# Default tile sizes. BLOCK_M=32 (per-expert token counts are small),
# BLOCK_N=64, BLOCK_K=64 (two scale groups per tile); 4 warps; 2 stages.
_DEFAULT_BLOCK_M: int = 32
_DEFAULT_BLOCK_N: int = 64
_DEFAULT_BLOCK_K: int = 64
_DEFAULT_NUM_WARPS: int = 4
_DEFAULT_NUM_STAGES: int = 2


def _mxfp4_per_expert_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales_bf16: torch.Tensor,
) -> torch.Tensor:
    """Run the per-expert MXFP4 -> bf16 GEMM.

    Args:
        x: ``[M, K]`` bf16, contiguous along K. M is the number of
            tokens routed to this expert (already gathered by the
            calling permute step).
        w_packed: ``[N, K // 2]`` uint8 of packed E2M1 nibbles
            (lo nibble = element ``2*i``, hi nibble = element
            ``2*i + 1``).
        w_scales_bf16: ``[N, K // 32]`` bf16 (decoded E8M0).

    Returns:
        ``[M, N]`` bf16.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must be bfloat16; got {x.dtype}")
    if w_packed.dtype != torch.uint8:
        raise TypeError(
            f"w_packed must be uint8 (packed nibbles); got {w_packed.dtype}"
        )
    if w_scales_bf16.dtype != torch.bfloat16:
        raise TypeError(
            "w_scales_bf16 must be bfloat16; "
            f"got {w_scales_bf16.dtype}"
        )

    M, K = x.shape
    N, packed_K_div2 = w_packed.shape
    if packed_K_div2 * 2 != K:
        raise ValueError(
            "w_packed second dim must equal K/2; "
            f"got K={K}, w_packed.shape[1]={packed_K_div2}"
        )
    expected_groups = (K + _BLOCK_SCALE_GROUP - 1) // _BLOCK_SCALE_GROUP
    if w_scales_bf16.shape != (N, expected_groups):
        raise ValueError(
            "w_scales_bf16 must be [N, K//32]; "
            f"expected ({N}, {expected_groups}); "
            f"got {tuple(w_scales_bf16.shape)}"
        )

    x = x.contiguous()
    w_packed = w_packed.contiguous()
    w_scales_bf16 = w_scales_bf16.contiguous()

    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if M == 0:
        return y

    grid = (
        triton.cdiv(M, _DEFAULT_BLOCK_M),
        triton.cdiv(N, _DEFAULT_BLOCK_N),
    )
    _mxfp4_b32_bf16_moe_per_expert_kernel[grid](
        x,
        w_packed,
        w_scales_bf16,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_packed.stride(0),
        w_packed.stride(1),
        w_scales_bf16.stride(0),
        w_scales_bf16.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=_DEFAULT_BLOCK_M,
        BLOCK_N=_DEFAULT_BLOCK_N,
        BLOCK_K=_DEFAULT_BLOCK_K,
        SCALE_GROUP=_BLOCK_SCALE_GROUP,
        num_warps=_DEFAULT_NUM_WARPS,
        num_stages=_DEFAULT_NUM_STAGES,
    )
    return y


# ----------------------------------------------------------------------
# Public entry point: full DSv4 expert FFN forward.
# ----------------------------------------------------------------------


def mxfp4_e2m1_b32_bf16_fused_moe(
    x: torch.Tensor,
    w_gate_packed: torch.Tensor,
    w_gate_scales: torch.Tensor,
    w_up_packed: torch.Tensor,
    w_up_scales: torch.Tensor,
    w_down_packed: torch.Tensor,
    w_down_scales: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    """Run a single expert FFN forward in MXFP4 -> bf16 dequant style.

    Implements ``y = (act(x @ W_gate.T) * (x @ W_up.T)) @ W_down.T`` for
    one expert. The caller is responsible for the ``select_experts ->
    permute`` step and the ``unpermute -> reduce_topk_weights``
    aggregation; this function is the inner-loop body that PR #40871's
    ``triton_unfused`` MoE backend dispatches per active expert.

    Args:
        x: ``[M, hidden]`` bf16. Already gathered to the tokens routed
            to this expert.
        w_*_packed: ``[N, hidden // 2]`` uint8 (or ``[hidden, N // 2]``
            for ``w_down`` — same per-expert weight layout DSv4 ships).
        w_*_scales: matching pre-decoded bf16 scale tensors,
            ``[..., K // 32]``.
        activation: gate activation function. DSv4 uses ``silu``;
            ``silu`` and ``gelu`` are accepted.
    """
    if activation not in ("silu", "gelu"):
        raise ValueError(
            f"activation must be 'silu' or 'gelu'; got {activation!r}"
        )
    if x.dim() != 2:
        raise ValueError(
            f"expected 2-D x [M, hidden]; got shape {tuple(x.shape)}"
        )

    gate = _mxfp4_per_expert_gemm(x, w_gate_packed, w_gate_scales)
    up = _mxfp4_per_expert_gemm(x, w_up_packed, w_up_scales)

    if activation == "silu":
        # SiLU(gate) * up. Done in fp32 internally to keep numerics
        # tight before the down-projection.
        gate_f32 = gate.to(torch.float32)
        gated = gate_f32 * torch.sigmoid(gate_f32)
        gated_bf16 = (gated * up.to(torch.float32)).to(torch.bfloat16)
    else:  # gelu
        gate_f32 = gate.to(torch.float32)
        gelu = (
            0.5 * gate_f32 * (1.0 + torch.tanh(
                0.7978845608028654
                * (gate_f32 + 0.044715 * gate_f32 ** 3)
            ))
        )
        gated_bf16 = (gelu * up.to(torch.float32)).to(torch.bfloat16)

    return _mxfp4_per_expert_gemm(gated_bf16, w_down_packed, w_down_scales)


def _mxfp4_fused_moe_fake(
    x: torch.Tensor,
    w_gate_packed: torch.Tensor,
    w_gate_scales: torch.Tensor,
    w_up_packed: torch.Tensor,
    w_up_scales: torch.Tensor,
    w_down_packed: torch.Tensor,
    w_down_scales: torch.Tensor,
    *,
    activation: str = "silu",
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], w_down_packed.shape[0]),
        dtype=torch.bfloat16,
        device=x.device,
    )


direct_register_custom_op(
    op_name="mxfp4_e2m1_b32_bf16_fused_moe_gfx908",
    op_func=mxfp4_e2m1_b32_bf16_fused_moe,
    mutates_args=[],
    fake_impl=_mxfp4_fused_moe_fake,
)
