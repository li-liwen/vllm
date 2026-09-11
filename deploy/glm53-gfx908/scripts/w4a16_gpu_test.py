#!/usr/bin/env python3
"""GPU test: AutoGPTQLinearMethod + TritonW4A16LinearKernel numerics on gfx908.

Builds a tiny GPTQ-packed linear layer (group 128, symmetric, uint4b8),
loads it through the vLLM machinery, and compares against dequantized
reference matmul.
"""
import torch

from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.platforms import current_platform


def main():
    assert current_platform.is_rocm()
    print("platform:", current_platform.get_device_capability())

    torch.manual_seed(0)
    K, N, G = 512, 256, 128
    device = "cuda"
    w = (torch.randn(N, K, dtype=torch.float32, device=device) * 0.05)

    # symmetric quantization, zero = 8 (uint4b8 convention)
    scales = w.abs().reshape(N, K // G, G).max(dim=2).values / 7.0  # [N, K//G]
    q = torch.round(w.reshape(N, K // G, G) / scales.unsqueeze(-1)).clamp(-8, 7)
    stored = (q + 8).to(torch.int32)

    # pack: qweight [K//8, N], nibble k%8 LSB-first
    flat = stored.reshape(N, K).t().contiguous()  # [K, N]
    qweight = torch.zeros(K // 8, N, dtype=torch.int32)
    for k in range(K):
        qweight[k // 8, :] |= flat[k, :] << ((k % 8) * 4)
    qzeros = torch.full((K // G, N // 8), 0x77777777, dtype=torch.int32)
    scales_h = scales.to(torch.float16)  # [N, K//G] then transposed by loader

    # Build the layer via the method
    method = AutoGPTQLinearMethod(
        AutoGPTQConfig(weight_bits=4, group_size=G, desc_act=False, is_sym=True,
                       lm_head_quantized=False, dynamic={}, full_config={})
    )
    # skip full Linear construction; exercise kernel directly
    from vllm.model_executor.layers.quantization.input_cache import InputCache
    from vllm.model_executor.kernels.linear.base import MPLinearLayerConfig
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel

    cfg = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=method.quant_config.quant_type,
        act_type=torch.bfloat16,
        group_size=G,
        zero_points=False,
    )
    kernel_type = choose_mp_linear_kernel(cfg)
    print("chosen kernel:", kernel_type.__name__)
    assert kernel_type.__name__ == "TritonW4A16LinearKernel", kernel_type.__name__

    class Layer(torch.nn.Module):
        pass

    layer = Layer()
    layer.register_parameter("qweight", torch.nn.Parameter(qweight, requires_grad=False))
    layer.qweight.input_dim = 0
    layer.qweight.output_dim = 1
    layer.qweight.packed_dim = 0
    layer.qweight.packed_factor = 8
    layer.register_parameter("scales", torch.nn.Parameter(scales_h, requires_grad=False))
    layer.scales.input_dim = 0
    layer.scales.output_dim = 1
    layer.register_parameter("qzeros", torch.nn.Parameter(qzeros, requires_grad=False))
    layer.qzeros.input_dim = 0
    layer.qzeros.output_dim = 1
    layer.qzeros.packed_dim = 1
    layer.qzeros.packed_factor = 8

    kernel = kernel_type(cfg, w_q_param_name="qweight", w_s_param_name="scales",
                         w_zp_param_name="qzeros")
    kernel.process_weights_after_loading(layer)

    x = torch.randn(64, K, dtype=torch.bfloat16, device=device)
    out = kernel.apply_weights(layer, x, None)

    ref = (x.float() @ ((q.float() * scales.unsqueeze(-1).float()).reshape(N, K).t()))
    diff = (out.float() - ref).abs()
    rel = diff.max().item() / ref.abs().max().item()
    print(f"max abs diff {diff.max().item():.5f}, rel {rel:.5f}")
    assert rel < 0.05, "dequantized matmul mismatch"
    print("PASS")


if __name__ == "__main__":
    main()
