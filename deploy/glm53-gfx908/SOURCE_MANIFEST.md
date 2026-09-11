# Source manifest — GLM-5.3-Flash W4A16 AutoRound on 8x MI100 (gfx908)

Base repository: li-liwen/vllm, commit 22258a26bc090bccf5473cf681bbe9bac41bd035 (main, immutable baseline).
Branch: feat/glm53-flash-gfx908

## Reference pins
| Reference | Pin | Use |
|---|---|---|
| btbtyler09/vllm-gfx908, mi100-optimized | 745883c4197e5bacd438f68de2ec509f5a890f50 | gfx908 platform guards, ROCm compat, graph handling, W4A16 dispatch, skinny GEMM. |
| Same repo, qwen38-flash-next | 808cd1633e8cf8165a232ae79e97de2b74df396a | Small-batch W4A16 kernels, BF16 GEMM tuning. Port selectively. |
| btbtyler09/mi100-llm-testing | d289d25c2140a0661f3e6c8138772db18450b4c0 | Build provenance, benchmark methodology, evidence. |
| larkinwc/vllm-gfx908 | 42d53e58525aaf63bcbf522c1ef0efd313eb86e7 | dtype handling cross-check, tuning infra, AITER findings, negative results. |
| promisezackr/glm53-flash-170hx-pp8 | 90ec72e9525e90be701e742c70a20c4154418307 | PP/mHC correctness, MTP loading, FP8 storage, sparse-attention fixes. |
| Mrzhiyao/glm53-a800-vllm | daeccb983ec84756cde7408b0e29161d492ea2c5 | KPool overrides, graph-safe tails, split-KV Triton sparse MLA. |
| PixelML/club-170hx | ab7000594aabd9a23b59bc619aa2d60ac0fb92a6 | Benchmark receipts, known failures. |
| btbtyler09/aiter-gfx908 | a454235de7100387b81f1bd56620c7744307f735 | AITER pin; clone under reference-repo if source changes needed. |

## Container image pin
btbtyler09/vllm-rocm-gfx908 @ sha256:03f325eb9fb40f21482972d30ade52ba1b47e2223dca66c71d1d3a057d0f0a67
(= installed v0.28.0rc7.dev-q38fn; digest authoritative)

## Checkpoint
/mnt/flash-inference/models/GLM-5.3-Flash-W4A16-AutoRound (NFS)
Verified local copy: /home/ubuntu/glm-5.3-flash/artifacts/checkpoint-local (outside Git)
~168.8 GiB indexed tensors, 34 shards + model_extra_conv.safetensors + model_extra_tensors.safetensors
AutoRound 0.15, GPTQ packing, symmetric INT4, group size 128.

## Host facts (recorded 2026-09-11)
- 8x AMD Instinct MI100 (gfx908), 8x ~31.984 GiB usable VRAM, driver 6.8.0-139-generic
- XGMI hives: GPUs 0-3 and 4-7; PCIe between hives
- 503 GiB host RAM, 893 GiB free local disk, 96 CPUs
- ROCm 7.2.4, PyTorch 2.12.0+git6bbd260, Triton 3.7.1+gitf0b55c07, Transformers 5.16.1 (inside pinned image)
- DSV4 container running (deepseek-v4-flash) — stop when GPU testing begins; snapshot in artifacts/dsv4-snapshot/
