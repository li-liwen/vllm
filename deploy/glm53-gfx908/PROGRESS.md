# Progress log — GLM-5.3-Flash W4A16 on 8x MI100

Branch: feat/glm53-flash-gfx908 (base 22258a26bc090bccf5473cf681bbe9bac41bd035)
Workspace: /home/ubuntu/glm-5.3-flash/vllm
Artifacts (outside Git): /home/ubuntu/glm-5.3-flash/artifacts/

## Phase 0 — branch and workspace (2026-09-11)
- [x] Created feat/glm53-flash-gfx908 from pinned base commit.
- [x] DSV4 launch config snapshotted to artifacts/dsv4-snapshot/rollback.md (API key redacted). Rollback = `docker start vllm-dsv4`.
- [x] GPU health recorded: artifacts/gpu-health-2026-09-11.md.
- [x] SOURCE_MANIFEST.md written with all pins.
- [~] Checkpoint local copy in progress: artifacts/checkpoint-local/ (rsync from NFS, ~168.8 GiB).

### Resume commands
```
cd /home/ubuntu/glm-5.3-flash/vllm
git checkout feat/glm53-flash-gfx908
# verify checkpoint copy finished:
rsync -an /mnt/flash-inference/models/GLM-5.3-Flash-W4A16-AutoRound/ /home/ubuntu/glm-5.3-flash/artifacts/checkpoint-local/ | tail -3   # empty output = complete
# then verify checksums (deploy/glm53-gfx908/audit_checkpoint.py)
```

## Phase 1 — pinned runtime build
- [x] Stage1 wheel build complete: glm53f-build:stage1 (46.8 GB) from vllm 0.28.0rc7.dev0+glm53.gfx908.
  - Base image digest sha256:03f325eb...; torch 2.12.0+git6bbd260 / triton 3.7.1 / transformers 5.16.1 untouched.
  - Deps audit: base image satisfies common+rocm reqs except tilelang/apache-tvm-ffi (added via requirements/glm53-pinned.txt).
  - /opt/prebuild_gfx908_exts.py "failures" are expected: those modules are Qwen4-fork-only, absent from our GLM branch.
- [x] Hardware probes PASSED 2026-09-11 (DSV4 stopped, 8 GPUs free):
  - matmul BF16/FP16 4096^3 vs chunked fp32 ref: max_diff 0.499/0.062 (sub-ULP) — PASS
  - Triton add kernel — PASS
  - RCCL all-reduce within both hives (0-3, 4-7) — PASS
  - 8-GPU RCCL all-reduce (cross-hive) + cross-hive send/recv — PASS
  - Graph capture+replay (with warmup), 8 concurrent single-GPU processes — PASS
  - p2p intra-hive PASS; cross-hive can_device_access_peer=False and direct torch
    cross-device copy SEGFAULTS (known MI100 issue; HSA_ENABLE_SVM/SDMA/TRANSFER
    knobs do not fix). PP transfers must use RCCL (verified working).
  - Graph capture without warmup faults in hipBLASLt workspace alloc; vLLM warms up
    before capture, so not a blocker. DISABLE_ADDMM_HIP_LT=1 set by default.
- Build: deploy/glm53-gfx908/scripts/build.sh

## Phase 2 — checkpoint loading
- [ ] Not started.

## Phase 2 — checkpoint loading (2026-09-11)
- [x] safetensors_use_index loader option + 8 CPU tests (all pass in container).
- [x] INC/auto-round W4A16 dispatch: AutoGPTQLinearMethod defers Marlin verify → TritonW4A16 on
      ROCm; INC scheme falls back to AutoGPTQ (kernel-choosing) when Marlin unavailable.
- [x] W4A16 dequant tests incl real checkpoint (5 pass): qzeros 0x77777777 = sym zero 8 ==
      uint4b8 bias; shapes match gs128; exclusions BF16; no g_idx.
- [x] INCConfig parses real quantization_config: bits 4, gs 128, sym, packing auto_round:auto_gptq;
      per-layer resolution: experts 4-bit, shared/conv1d/router/down_proj 16-bit.
- Remaining for Phase 2 gate: full model load on GPU (deferred to Phase 4 boot at 8K).

## Phase 3/4 prep findings (2026-09-11, CPU-side)
- KDA: main already dispatches ROCm → amd/ops/third_party/kda triton kernels.
- mHC: aiter mhc ops are torch-composed (arch-safe on gfx908); tilelang MHC enabled on non-gfx942
  ROCm; falls back to torch kernels otherwise.
- Indexer: ROCm routes to rocm_fp8_mqa_logits / rocm_fp8_paged_mqa_logits (aiter triton on
  gfx908 — no gluon special-casing — or torch fallback). 2D per-row seq_lens supported in both.
- MTP: glm5_next → Glm5NextMTPModel registration intact; draft parallel config uses pp=1
  (plan patch 0004 not needed). Draft embed/head loading ported (0005), shared head unquantized (0017).
- 8K boot profile now carries DSV4 stability envs (HSA_ENABLE_SVM=0, HSA_NO_SCRATCH_RECLAIM=1,
  HIP_FORCE_DEV_KERNARG=1, TORCH_BLAS_PREFER_HIPBLASLT=0).
- Build fix: base image's stale vllm (0.27.2) namespace-shadows the editable install — purged in
  Dockerfile (vllm.entrypoints.cli ModuleNotFoundError root cause).

### 1M memory estimate (TP4xPP2, refined with real tensor shapes)
- Server weights ~180 GiB (attn 17.9 + MoE 152 + draft 3.75 + embed/head 2.4 + vision ~1.9).
- Per rank at 1M: PP0 ≈ 31.3 GiB / 32 (0.7 headroom), PP1 ≈ 29.0 GiB / 32 (3.0 headroom),
  before activation/graph buffers. Tight on PP0 — activation peaks and graph pools may force
  the plan's fallbacks (reduced graphs → seqs 1 → chunk 1024 → TP2xPP4 partition 12,12,12,9).

## Phase 4 MILESTONE — full model boots TP4xPP2 (2026-09-11 19:58 UTC)
- [x] GLM-5.3-Flash W4A16 AutoRound loads and serves on 8x MI100 with TP4×PP2, 8K context, eager.
- [x] All 8 workers: Model loading took 21.7 GiB each, ~202 s (indexed loader active, 36 shards).
- [x] WNA16 MoE backend = TRITON chosen automatically on gfx908 (Marlin rejected).
- [x] init engine (profile, create kv cache, warmup) took 452 s; Application startup complete.
- [x] Memory profile at 8K: PP0 weights+non-torch 21.31 GiB, activation peak 1.27 GiB, KV 7.17 GiB;
      PP1 weights 22.84 GiB, KV 5.73 GiB. All under the 29.75 GiB utilization target.
- [x] Smoke tests: 17*23=391 correct; Chinese instruction coherent; step-by-step math correct;
      image understanding correct (red background + green rectangle identified).
- KV/mamba page alignment handled by main's auto block-size (attention block 1152, mamba padded).
- Remaining Phase 4 gate: MTP enablement + acceptance (next step).

### Boot fixes during bring-up
1. extra-config whitelist needed safetensors_use_index (default_loader.py).
2. index_file NameError in _get_weights_iterator → use SAFE_WEIGHTS_INDEX_NAME.
3. Base-image stale vllm + pip editable namespace stubs shadow submodule imports → purged in
   Dockerfile (both before install -e and after).
Image chain: glm53f-build:fix6 → glm53f:latest (commit-tagged rebuilds pending for reproducibility).

## Phase 4 MILESTONE — native MTP enabled (2026-09-11 22:41 UTC)
- [x] GLM-5.3-Flash serves with native MTP depth 1 on TP4xPP2. Application startup complete.
- [x] Draft loaded 28/30 params per rank; the 2 untouched are qzeros (expected: symmetric uint4b8
      ignores qzeros; checkpoint's 0x77777777 has no param home under MoeWNA16-symmetric).
- [x] Draft construction: MoE prefix 'model.layers.45.mlp.experts' (no mtp_block infix — that comes
      only from the module attribute path). INC block check needed a root remap from
      'model.layers.' onto the (mapper-rewritten) 'language_model.model.layers' block.
- [x] Streaming bursts: 106/106 multi-token bursts (MTP drafting active), coherent outputs.
- Boot fixes: INC parser root remap (config_parser.py); diagnosis logs added in
  inc.py/mtp.py/model.py/routed_experts.py (to be cleaned before final image).
