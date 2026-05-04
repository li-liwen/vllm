# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the gfx908 mHC Triton fused pre block.

The reference implementation is the upstream torch fallback at
``vllm/model_executor/layers/mhc.py`` (the ROCm branch of ``mhc_pre``).
The Triton port must match it bit-for-equivalent within bf16 noise.
"""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

from vllm.model_executor.layers.mhc_triton_gfx908 import (
    mhc_pre_gfx908_triton,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton mHC kernel requires a GPU device.",
)


def _torch_reference_mhc_pre(
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
    """Mirror of the ROCm branch in mhc.py:mhc_pre."""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    x = residual_flat.reshape(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
        + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = (
        mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[2]
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


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
@pytest.mark.parametrize("hidden_size", [256, 1024])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre_triton_matches_torch_reference(
    num_tokens: int, hidden_size: int, hc_mult: int
):
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0xD5C4)

    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    residual = (
        torch.randn(num_tokens, hc_mult, hidden_size, generator=g) * 0.1
    ).to(device=device, dtype=torch.bfloat16)
    fn = (
        torch.randn(hc_mult3, hc_mult * hidden_size, generator=g) * 0.05
    ).to(device=device, dtype=torch.float32)
    hc_scale = (torch.rand(3, generator=g) * 0.5 + 0.5).to(
        device=device, dtype=torch.float32
    )
    hc_base = (torch.randn(hc_mult3, generator=g) * 0.1).to(
        device=device, dtype=torch.float32
    )

    rms_eps = 1e-6
    hc_pre_eps = 1e-4
    hc_sinkhorn_eps = 1e-6
    hc_post_mult_value = 1.5
    sinkhorn_repeat = 3

    post_t, comb_t, lin_t = mhc_pre_gfx908_triton(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    post_r, comb_r, lin_r = _torch_reference_mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )

    # post_mix and comb_mix are fp32; they should match closely (within
    # accumulation order noise of the Triton vs torch GEMM).
    assert torch.allclose(post_t, post_r, atol=1e-3, rtol=1e-3), (
        f"post_mix differs: max_abs={((post_t - post_r).abs()).max().item():.3e}"
    )
    assert torch.allclose(comb_t, comb_r, atol=1e-3, rtol=1e-3), (
        f"comb_mix differs: max_abs={((comb_t - comb_r).abs()).max().item():.3e}"
    )
    # layer_input is bf16; allow ~1e-2 absolute slack.
    rel_err = (
        (lin_t.float() - lin_r.float()).abs()
        / lin_r.float().abs().clamp(min=1e-3)
    )
    assert rel_err.mean() <= 1e-2
    assert rel_err.quantile(0.99) <= 5e-2


def test_mhc_pre_triton_handles_empty():
    device = torch.device("cuda")
    hc_mult = 4
    hidden_size = 128
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    residual = torch.empty(
        0, hc_mult, hidden_size, dtype=torch.bfloat16, device=device
    )
    fn = torch.zeros(
        hc_mult3, hc_mult * hidden_size, dtype=torch.float32, device=device
    )
    hc_scale = torch.ones(3, dtype=torch.float32, device=device)
    hc_base = torch.zeros(hc_mult3, dtype=torch.float32, device=device)
    post, comb, lin = mhc_pre_gfx908_triton(
        residual, fn, hc_scale, hc_base, 1e-6, 1e-4, 1e-6, 1.5, 3
    )
    assert post.shape == (0, hc_mult, 1)
    assert comb.shape == (0, hc_mult, hc_mult)
    assert lin.shape == (0, hidden_size)
