# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the gfx908 FP8 E4M3 + E8M0(B128) -> bf16 dense GEMM."""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

from vllm.model_executor.layers.quantization.gfx908.fp8_e4m3_b128_bf16_gemm import (  # noqa: E501
    decode_e8m0_block_scales_to_bf16,
    fp8_e4m3_b128_bf16_gemm,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton GEMM kernel requires a GPU device.",
)


# ----------------------------------------------------------------------
# E8M0 scale decoder
# ----------------------------------------------------------------------


def test_decode_e8m0_full_range():
    """Every E8M0 byte except the NaN sentinel decodes to the bf16 of
    ``2 ** (e - 127)``."""
    e = torch.arange(256, dtype=torch.uint8)
    decoded = decode_e8m0_block_scales_to_bf16(e)

    # Reference: ldexp(1.0, e - 127) in fp32, cast to bf16.
    exp = e.to(torch.int32) - 127
    ref = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp)
    # NaN sentinel -> 0
    ref[e == 0xFF] = 0.0
    ref_bf16 = ref.to(torch.bfloat16)

    assert torch.equal(decoded, ref_bf16)


def test_decode_e8m0_shape_preserved():
    e = torch.tensor([[120, 127, 134], [127, 127, 127]], dtype=torch.uint8)
    decoded = decode_e8m0_block_scales_to_bf16(e)
    assert decoded.shape == e.shape
    assert decoded.dtype == torch.bfloat16


def test_decode_e8m0_rejects_non_uint8():
    with pytest.raises(TypeError):
        decode_e8m0_block_scales_to_bf16(
            torch.zeros(4, dtype=torch.int32),
        )


# ----------------------------------------------------------------------
# FP8 GEMM correctness
# ----------------------------------------------------------------------


def _make_fp8_weight(
    n: int,
    k: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct (w_fp8, scales_bf16, w_dequant_bf16) consistent with the
    on-disk DSv4 dense-FP8 layout."""
    assert k % 128 == 0, "test helper only supports K%128==0"
    g = torch.Generator(device="cpu").manual_seed(seed)

    # Random fp32 reference, then quantize per-128-block.
    w_ref = torch.randn(n, k, generator=g, dtype=torch.float32) * 0.05

    n_groups = k // 128
    w_grouped = w_ref.reshape(n, n_groups, 128)

    # Pick a per-row, per-group scale that absorbs the row's amax for
    # that group; then dequantize back so the bf16 reference is what
    # the kernel must reproduce.
    amax = w_grouped.abs().amax(dim=-1).clamp(min=1e-8)
    # Target FP8 E4M3 max magnitude is 448. Choose scale so that the
    # largest absolute element in the group maps to ~448.
    raw_scales = amax / 448.0
    # Quantize the scale to a power of two so it round-trips cleanly
    # through E8M0.
    log2_scales = torch.ceil(torch.log2(raw_scales))
    pow2_scales = torch.exp2(log2_scales)
    e8m0_byte = (log2_scales.to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)

    # Re-derive the bf16 scale from the byte to mimic the load path.
    scales_bf16 = decode_e8m0_block_scales_to_bf16(e8m0_byte).to(device)

    # Quantize the weight to FP8 E4M3 using the chosen pow2 scale.
    w_scaled = w_grouped / pow2_scales.unsqueeze(-1)
    w_fp8 = w_scaled.reshape(n, k).to(torch.float8_e4m3fn).to(device)

    # Reference bf16 weight (what the kernel reconstructs internally):
    w_dequant = w_fp8.to(torch.float32).reshape(n, n_groups, 128) * pow2_scales.to(
        device
    ).unsqueeze(-1)
    w_dequant_bf16 = w_dequant.reshape(n, k).to(torch.bfloat16)

    return w_fp8, scales_bf16, w_dequant_bf16


@pytest.mark.parametrize("M", [1, 7, 64, 129])
@pytest.mark.parametrize("N", [128, 256])
@pytest.mark.parametrize("K", [128, 256, 512])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fp8_b128_bf16_gemm_matches_dequant_reference(
    M: int, N: int, K: int, with_bias: bool
):
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0xABCD)

    x = (torch.randn(M, K, generator=g, dtype=torch.float32) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )

    w_fp8, scales_bf16, w_dequant_bf16 = _make_fp8_weight(
        N, K, seed=0x1234, device=device
    )

    bias = None
    if with_bias:
        bias = (
            torch.randn(N, generator=g, dtype=torch.float32) * 0.01
        ).to(device=device, dtype=torch.bfloat16)

    out = fp8_e4m3_b128_bf16_gemm(x, w_fp8, scales_bf16, bias)
    ref = (x.to(torch.float32) @ w_dequant_bf16.to(torch.float32).T).to(
        torch.bfloat16
    )
    if bias is not None:
        ref = ref + bias

    rel_err = (out.float() - ref.float()).abs() / (
        ref.float().abs().clamp(min=1e-3)
    )
    # FP8 has ~3 sigfigs; bf16 has ~3; per-element error a few percent
    # is expected. Acceptance per agents.md T7.2: <= 5e-3 rel err.
    assert rel_err.mean() <= 5e-3, (
        f"mean rel err {rel_err.mean().item():.3e} exceeds 5e-3"
    )
    # The 99th-percentile element should also be well below 5%.
    assert rel_err.quantile(0.99) <= 5e-2, (
        f"p99 rel err {rel_err.quantile(0.99).item():.3e} exceeds 5e-2"
    )


def test_fp8_b128_bf16_gemm_handles_3d_input():
    """A leading batch dim must round-trip through the GEMM."""
    device = torch.device("cuda")
    K, N = 128, 128
    w_fp8, scales_bf16, _ = _make_fp8_weight(N, K, seed=0xFE, device=device)
    x = torch.randn(2, 3, K, dtype=torch.bfloat16, device=device)
    out = fp8_e4m3_b128_bf16_gemm(x, w_fp8, scales_bf16, None)
    assert out.shape == (2, 3, N)
    assert out.dtype == torch.bfloat16


def test_fp8_b128_bf16_gemm_rejects_wrong_dtypes():
    device = torch.device("cuda")
    K, N = 128, 128
    w_fp8, scales_bf16, _ = _make_fp8_weight(N, K, seed=0xCC, device=device)
    x_fp32 = torch.randn(4, K, dtype=torch.float32, device=device)
    with pytest.raises(TypeError):
        fp8_e4m3_b128_bf16_gemm(x_fp32, w_fp8, scales_bf16, None)
    with pytest.raises(TypeError):
        fp8_e4m3_b128_bf16_gemm(
            x_fp32.to(torch.bfloat16),
            w_fp8.to(torch.float32),  # wrong: must be float8_e4m3fn
            scales_bf16,
            None,
        )
