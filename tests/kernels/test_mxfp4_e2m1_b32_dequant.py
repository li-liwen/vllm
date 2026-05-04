# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the gfx908 MXFP4 (E2M1 + E8M0 B32) -> bf16 fused MoE GEMM."""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

from vllm.model_executor.layers.quantization.gfx908.fp8_e4m3_b128_bf16_gemm import (  # noqa: E501
    decode_e8m0_block_scales_to_bf16,
)
from vllm.model_executor.layers.quantization.gfx908.mxfp4_e2m1_b32_bf16_moe import (  # noqa: E501
    _mxfp4_per_expert_gemm,
    mxfp4_e2m1_b32_bf16_fused_moe,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton MXFP4 MoE kernel requires a GPU device.",
)


# Reference E2M1 codepoint table (OCP MX spec).
_E2M1_VALUES: tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _reference_decode_packed(
    w_packed: torch.Tensor, K: int
) -> torch.Tensor:
    """Decode the on-disk packed-uint8 layout to a bf16 weight matrix.

    ``w_packed[n, b]`` stores two nibbles: low nibble is element
    ``2*b``, high nibble is element ``2*b + 1`` along K.
    """
    N, packed_K = w_packed.shape
    assert packed_K * 2 == K
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=w_packed.device)

    nib_lo = (w_packed & 0xF).long()
    nib_hi = ((w_packed >> 4) & 0xF).long()
    decoded = torch.empty(N, K, dtype=torch.float32, device=w_packed.device)
    decoded[:, 0::2] = lut[nib_lo]
    decoded[:, 1::2] = lut[nib_hi]
    return decoded


def _quantize_to_mxfp4(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a fp32 weight to (packed nibbles, bf16 scales, bf16 dequant).

    Block size 32 along K. Scales chosen as the smallest power-of-two
    that maps the per-32-block amax to <= 6.0 (the largest E2M1
    magnitude). Returns the packed bytes, the bf16 scales, and the bf16
    weight that the kernel should reproduce.
    """
    N, K = w.shape
    assert K % 32 == 0
    g = K // 32

    grouped = w.reshape(N, g, 32)
    amax = grouped.abs().amax(dim=-1).clamp(min=1e-8)
    raw = amax / 6.0
    log2_scale = torch.ceil(torch.log2(raw))
    scale = torch.exp2(log2_scale)
    e8m0_byte = (log2_scale.to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)

    scales_bf16 = decode_e8m0_block_scales_to_bf16(e8m0_byte)

    # Quantize: divide by scale, then round to nearest E2M1 codepoint.
    scaled = grouped / scale.unsqueeze(-1)
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=w.device)
    # For each element, find the nearest LUT codepoint by minimum |diff|.
    diffs = (scaled.unsqueeze(-1) - lut).abs()
    code = diffs.argmin(dim=-1).to(torch.uint8)  # [N, g, 32]

    code_flat = code.reshape(N, K)
    code_lo = code_flat[:, 0::2]
    code_hi = code_flat[:, 1::2]
    packed = (code_lo & 0xF) | ((code_hi & 0xF) << 4)
    packed = packed.to(torch.uint8)

    # Reconstruct the bf16 weight the kernel should reproduce.
    decoded = lut[code_flat.long()]
    dequant = (decoded.reshape(N, g, 32) * scale.unsqueeze(-1)).reshape(N, K)
    return packed, scales_bf16, dequant.to(torch.bfloat16)


@pytest.mark.parametrize("M", [1, 4, 32, 33])
@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("K", [64, 128, 256])
def test_mxfp4_per_expert_gemm_matches_dequant_reference(
    M: int, N: int, K: int
):
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0xBEEF)
    x = (torch.randn(M, K, generator=g) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )
    w = (torch.randn(N, K, generator=g) * 0.05).to(device=device)
    w_packed, scales_bf16, w_dequant = _quantize_to_mxfp4(w)

    out = _mxfp4_per_expert_gemm(x, w_packed, scales_bf16)
    ref = (
        x.to(torch.float32) @ w_dequant.to(torch.float32).T
    ).to(torch.bfloat16)

    rel_err = (out.float() - ref.float()).abs() / (
        ref.float().abs().clamp(min=1e-3)
    )
    assert rel_err.mean() <= 1e-2, (
        f"mean rel err {rel_err.mean().item():.3e} exceeds 1e-2"
    )
    assert rel_err.quantile(0.99) <= 1e-1, (
        f"p99 rel err {rel_err.quantile(0.99).item():.3e} exceeds 1e-1"
    )


def test_mxfp4_decode_lut_matches_reference():
    """Round-trip: pack one of every codepoint and check the kernel
    reproduces the exact LUT value (no rounding involved)."""
    device = torch.device("cuda")
    # Each row holds the 16 distinct codepoints as 8 packed bytes.
    code_per_row = torch.arange(16, dtype=torch.uint8, device=device)
    code_lo = code_per_row[0::2]
    code_hi = code_per_row[1::2]
    packed_row = (code_lo & 0xF) | ((code_hi & 0xF) << 4)
    # N=4 so the kernel has more than one row to dequantize.
    w_packed = packed_row.unsqueeze(0).repeat(4, 1).contiguous()
    K = 16
    # Identity scale so the kernel output is exactly the LUT values.
    scales_bf16 = torch.ones(
        (4, K // 32 + (1 if K % 32 else 0)), dtype=torch.bfloat16, device=device
    )

    # Use an x that picks out individual columns: identity-like on the
    # first M=K rows.
    M = K
    x = torch.eye(M, K, dtype=torch.bfloat16, device=device)

    out = _mxfp4_per_expert_gemm(x, w_packed, scales_bf16)
    expected = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)
    expected = expected.unsqueeze(0).repeat(4, 1).T.to(torch.bfloat16)  # [K, N=4]

    assert torch.allclose(out.float(), expected.float(), atol=0.0)


def test_mxfp4_fused_moe_silu_path():
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0xC0FFEE)
    M, hidden, inter = 6, 64, 64
    x = (torch.randn(M, hidden, generator=g) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )
    w_gate = (torch.randn(inter, hidden, generator=g) * 0.05).to(device=device)
    w_up = (torch.randn(inter, hidden, generator=g) * 0.05).to(device=device)
    w_down = (torch.randn(hidden, inter, generator=g) * 0.05).to(device=device)
    g_p, g_s, g_d = _quantize_to_mxfp4(w_gate)
    u_p, u_s, u_d = _quantize_to_mxfp4(w_up)
    d_p, d_s, d_d = _quantize_to_mxfp4(w_down)

    out = mxfp4_e2m1_b32_bf16_fused_moe(
        x, g_p, g_s, u_p, u_s, d_p, d_s, activation="silu"
    )

    g_act = (x.to(torch.float32) @ g_d.float().T)
    u_act = (x.to(torch.float32) @ u_d.float().T)
    silu = g_act * torch.sigmoid(g_act)
    ref = ((silu * u_act).to(torch.bfloat16).float() @ d_d.float().T).to(
        torch.bfloat16
    )

    rel_err = (out.float() - ref.float()).abs() / (
        ref.float().abs().clamp(min=1e-3)
    )
    assert rel_err.mean() <= 2e-2
