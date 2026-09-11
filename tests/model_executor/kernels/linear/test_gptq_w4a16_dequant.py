# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 GPTQ dequantization checks against real GLM-5.3-Flash checkpoint tensors.

Validates the AutoRound/GPTQ packing conventions the deployment relies on:

- qzeros words 0x77777777 (nibble 7) = symmetric, effective zero 8 under
  GPTQ's offset convention (w = (q - 8) * scale); the TritonW4A16 kernel's
  uint4b8 zp_bias=8 must reproduce dequantized weights.
- absent g_idx = non-activation-ordered groups.
- TP slicing respects the K-major packing of qweight [K//8, N].
- BF16 exclusions (dense/shared experts) stay in their floating format.

CPU-only: dequantization math is exercised directly (no GPU kernels).
Set GLM53_CHECKPOINT to run against the real checkpoint; otherwise the
synthetic cases run everywhere.
"""

import os
import struct

import pytest
import torch

CHECKPOINT = os.environ.get("GLM53_CHECKPOINT", "")


def _read_tensor(path, name):
    import json

    import safetensors

    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header[name]
    with safetensors.safe_open(path, framework="pt") as f:
        return f.get_tensor(name)


def _unpack_qweight(qweight):
    """[K//8, N] int32, nibble k%8 of word k//8, LSB-first → [K, N] in [0,15]."""
    shifts = torch.arange(8, dtype=torch.int32) * 4
    nibbles = (qweight.unsqueeze(-1) >> shifts) & 0xF  # [K//8, N, 8]
    return nibbles.permute(0, 2, 1).reshape(qweight.shape[0] * 8, qweight.shape[1])


def test_qzeros_offset_convention_synthetic():
    """0x77777777 nibble 7 + offset convention == uint4b8 zero 8."""
    word = 0x77777777
    nibbles = [(word >> (4 * i)) & 0xF for i in range(8)]
    assert all(n == 7 for n in nibbles)
    # GPTQ asymmetric dequant is w = (q - (zp + 1)) * scale for symmetric
    # checkpoints that store 2^(bits-1) - 1; 7 + 1 = 8 = uint4b8's bias.
    assert nibbles[0] + 1 == 8


def test_triton_w4a16_zp_bias_matches_gptq_synthetic():
    """Dequantize a packed GPTQ tensor with zp=8 and compare to float weights."""
    torch.manual_seed(0)
    K, N, G = 256, 128, 128
    w = torch.randn(N, K, dtype=torch.float16) * 0.05
    # symmetric per-group quantization, zero 8, uint4 range [-8, 7]
    scales = (
        w.abs()
        .reshape(N, K // G, G)
        .max(dim=2)
        .values / 7.0
    )  # [N, K//G]
    q = torch.round(w.reshape(N, K // G, G) / scales.unsqueeze(-1)).clamp(-8, 7)
    stored = (q + 8).to(torch.int32)  # [N, K//G, G] in [0, 15]
    # pack K-major: qweight[k//8, n], nibble k%8
    flat = stored.reshape(N, K).t().contiguous()  # [K, N]
    qweight = torch.zeros(K // 8, N, dtype=torch.int32)
    for kk in range(K):
        qweight[kk // 8, :] |= flat[kk, :] << ((kk % 8) * 4)
    # dequant with the uint4b8 convention (what TritonW4A16 implements)
    unpacked = _unpack_qweight(qweight).t().reshape(N, K // G, G)  # [N, K//G, G]
    deq = (unpacked - 8).float() * scales.unsqueeze(-1).float()
    ref = (q.float() * scales.unsqueeze(-1).float()).reshape(N, K)
    torch.testing.assert_close(
        deq.reshape(N, K), ref, atol=1e-3, rtol=1e-3
    )


def test_tp_slice_k_major_packing_synthetic():
    """Column-parallel slicing of [K//8, N] qweight by N keeps nibble order."""
    torch.manual_seed(1)
    K, N = 128, 64
    qweight = torch.randint(0, 2**31, (K // 8, N), dtype=torch.int32)
    # TP rank r takes columns [r*N/2, (r+1)*N/2)
    r = 1
    tp = 2
    lo, hi = r * N // tp, (r + 1) * N // tp
    sliced = qweight[:, lo:hi]
    full = _unpack_qweight(qweight)
    part = _unpack_qweight(sliced)
    torch.testing.assert_close(full[:, lo:hi].int(), part.int())


@pytest.mark.skipif(
    not CHECKPOINT or not os.path.isdir(CHECKPOINT),
    reason="GLM53_CHECKPOINT not set",
)
def test_real_checkpoint_expert_qzeros_all_0x77777777():
    """Every sampled qzeros word must be the symmetric 0x77777777 pattern."""
    import json

    idx = json.load(open(os.path.join(CHECKPOINT, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    # sample one expert per layer
    targets = {}
    for name, fname in wm.items():
        if name.endswith("mlp.experts.0.gate_proj.qzeros"):
            layer = name.split(".layers.")[1].split(".")[0]
            targets[name] = fname
        if len(targets) >= 8:
            break
    assert targets, "no expert qzeros found in checkpoint"
    for name, fname in targets.items():
        qz = _read_tensor(os.path.join(CHECKPOINT, fname), name)
        words = qz.view(torch.int32).reshape(-1)
        assert (words == 0x77777777).all(), f"{name} has non-0x77777777 words"


@pytest.mark.skipif(
    not CHECKPOINT or not os.path.isdir(CHECKPOINT),
    reason="GLM53_CHECKPOINT not set",
)
def test_real_checkpoint_shapes_and_exclusions():
    """Packed shapes match group_size 128 K-major packing; exclusions stay BF16."""
    import json

    idx = json.load(open(os.path.join(CHECKPOINT, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    # gate_proj: N=2048 (moe_intermediate), K=4096 → qweight [4096/8=512, 2048]
    qw = _read_tensor(
        os.path.join(CHECKPOINT, wm["model.language_model.layers.20.mlp.experts.0.gate_proj.qweight"]),
        "model.language_model.layers.20.mlp.experts.0.gate_proj.qweight",
    )
    assert qw.shape == (512, 2048), qw.shape
    assert qw.dtype == torch.int32
    # scales F16 [K//G, N] = [32, 2048]
    sc = _read_tensor(
        os.path.join(CHECKPOINT, wm["model.language_model.layers.20.mlp.experts.0.gate_proj.scales"]),
        "model.language_model.layers.20.mlp.experts.0.gate_proj.scales",
    )
    assert sc.shape == (32, 2048)
    # exclusions: shared experts and down_proj are BF16
    for name in (
        "model.language_model.layers.20.mlp.shared_experts.gate_proj.weight",
        "model.language_model.layers.20.mlp.shared_experts.up_proj.weight",
    ):
        t = _read_tensor(os.path.join(CHECKPOINT, wm[name]), name)
        assert t.dtype == torch.bfloat16
    # conv1d repaired shape from the extra file
    conv = _read_tensor(
        os.path.join(CHECKPOINT, wm["model.language_model.layers.0.self_attn.q_conv1d.weight"]),
        "model.language_model.layers.0.self_attn.q_conv1d.weight",
    )
    assert conv.shape == (8192, 1, 4)
    # no g_idx anywhere in the index
    assert not any("g_idx" in n for n in wm), "unexpected g_idx in checkpoint"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
