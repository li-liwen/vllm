# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx908 (MI100 / CDNA1) DSv4 dequant kernels.

These kernels exist because gfx908 has no FP8 / FP4 MFMA tensor cores. We
keep the weights in their native low-precision form in VRAM, then perform
the dequantization as a software prologue inside each Triton kernel and
do the matrix multiply in bf16 on MFMA. Every kernel here is gated behind
:func:`vllm.platforms.rocm.is_dsv4_gfx908_path` at the dispatch site.
"""

from vllm.model_executor.layers.quantization.gfx908.fp8_e4m3_b128_bf16_gemm import (  # noqa: E501
    decode_e8m0_block_scales_to_bf16,
    fp8_e4m3_b128_bf16_gemm,
)
from vllm.model_executor.layers.quantization.gfx908.mxfp4_e2m1_b32_bf16_moe import (  # noqa: E501
    mxfp4_e2m1_b32_bf16_fused_moe,
)

__all__ = [
    "decode_e8m0_block_scales_to_bf16",
    "fp8_e4m3_b128_bf16_gemm",
    "mxfp4_e2m1_b32_bf16_fused_moe",
]
