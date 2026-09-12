# Milestone 1 — GLM-5.3-Flash W4A16 AutoRound serving on 8× MI100 (gfx908)

**Date:** 2026-09-12
**Status:** ✅ Achieved — GLM serving live with text, reasoning, tool calling, images;
MTP depth 2 + HIP graphs; c=1 pooled decode 38.8 tok/s (target ≥50 pending Phase 6).
**Plan reference:** [PLAN.md](./PLAN.md) (phases 0–2 complete, 3/4 substantially complete,
6 partially complete, 5/7 pending).

---

## 1. Deployment facts

| Item | Value |
|---|---|
| Branch | `feat/glm53-flash-gfx908` @ `e7f562cf212acdd9986db6a4fa6ee001fb3a3cb2` (pushed to li-liwen/vllm) |
| Base commit | `22258a26bc090bccf5473cf681bbe9bac41bd035` (vLLM main, immutable) |
| Commits on branch | 70 (excluding merge commits) |
| vLLM version | `0.28.0rc7.dev0+glm53.gfx908` |
| Base image | `btbtyler09/vllm-rocm-gfx908:v0.28.0rc7.dev-q38fn` @ `sha256:03f325eb9fb40f21482972d30ade52ba1b47e2223dca66c71d1d3a057d0f0a67` |
| Serving image | `glm53f:latest` (id `72672a143c0b`), built FROM `glm53f-build:fix27` |
| Toolchain (inherited, pinned) | ROCm 7.2.4, PyTorch 2.12.0+git6bbd260, Triton 3.7.1+gitf0b55c07, Transformers 5.16.1, Python 3.12 |
| Checkpoint | `GLM-5.3-Flash-W4A16-AutoRound` (AutoRound 0.15, GPTQ packing, sym INT4, group 128) |
| Checkpoint copy | `/home/ubuntu/glm-5.3-flash/artifacts/checkpoint-local` — all 113,074 indexed tensors verified, 169.010 GiB, sha256 manifest (38 files) |
| Serving config | TP=4, PP=2, bf16, 8K ctx (1M pending), MTP depth 2, HIP graphs, block 128, tool parser `glm47`, reasoning parser `glm45` |
| Live endpoint | `vllm-glm53f` on port 8006 (API key via env, not stored in Git) |
| Rollback | `docker start vllm-dsv4` (DSV4 container stopped, config snapshotted in artifacts/dsv4-snapshot/) |

## 2. What is verified working

### Loading correctness (Phase 2 gate)
- `--load-format safetensors --model-loader-extra-config '{"safetensors_use_index": true}'`
  loads every tensor exactly once from its index-mapped file, defeating the stale
  intra-shard duplicates ([24576,1,4] conv copies vs [8192,1,4] repairs).
  8 CPU tests pass (`tests/model_executor/model_loader/test_indexed_safetensors.py`).
- INC/auto-round dispatch: dense INT4 → `TritonW4A16LinearKernel` (Marlin rejected on
  gfx908); routed experts → `MoeWNA16Method` → **TRITON** WNA16 MoE backend (log line:
  `Using 'TRITON' WNA16 MoE backend`). Exclusions verified: shared experts, conv1d,
  router gate, dense down_proj stay BF16.
- W4A16 dequant tests pass **against the real checkpoint**
  (`tests/model_executor/kernels/linear/test_gptq_w4a16_dequant.py`): qzeros 0x77777777
  ≡ uint4b8 bias-8, shapes match group 128, no g_idx anywhere.

### Serving (Phase 3/4 gates)
- Full 45-layer model boots TP4×PP2 on all 8 GPUs: 20.2 GiB weights/rank, ~130 s load,
  Triton JIT warmup, graph capture passes on all 8 workers.
- **Native MTP** (draft = checkpoint layer 45): draft loads 28/30 quantized params per rank
  (2 untouched qzeros are expected for symmetric uint4b8); draft embedding and output head
  loaded from the checkpoint; shared head built unquantized matching the BF16 `lm_head`.
- **HIP graphs**: capture succeeds with MTP; graph memory accounted by the profiler.
- **Multimodal**: image understanding verified (correctly identified red background +
  green rectangle in a synthetic PNG through the GLM vision tower).
- **Reasoning parser `glm45`**: reasoning/content split verified live.
- **Tool calling**: model emits well-formed native `get_weather` tool calls; `glm47`
  parser wired in the serving profile.
- Output quality spot checks pass: 17×23=391, 15% of 80 = 12, 3x=27 → 9, discount
  pricing → 30, coherent Chinese instructions, coherent quantum-computing summary.

### Performance (Phase 6, in progress)
C1 decode benchmark per plan §4 (3 prompt families × ~4k tokens, 1024 forced output
tokens, 1 warmup + 5 reps, median, streaming chars/3.2 → tok/s estimate):

| MTP depth | code | math | prose | **pooled median** |
|---|---|---|---|---|
| 1 (eager) | — | — | — | ~13 |
| 1 (graphs) | ~32 | ~14 | ~30 | ~30 |
| **2 (graphs)** | **46.6** | **19.1** | **38.8** | **38.8** |

The 2.4× eager→graphs jump validates the graph path; depth 2 beats depth 1 by ~30%.
**Pooled median 38.8 tok/s vs the ≥50 gate — not yet met.** Math-family MTP acceptance
(19 tok/s) is the pooled blocker; remaining levers listed in §5.

## 3. Key engineering work landed (70 commits)

**Platform / gfx908 enablement**
- `_ON_GFX908` arch flag, `on_gfx908()`, per-feature AITER env defaults (CK ops off,
  Triton paths on), CK→Triton redirects for rms_norm/rmsnorm2d/flash_attn_varlen
  (port of btbtyler09/vllm-gfx908 d3bab5eb0, 2ae323c98, d2e8687f3).
- Unified-attention default OFF on gfx908 (state-corruption reports in the later
  reference defaults supersede the initial ON).

**Loader**
- `safetensors_use_index` loader option + tests (stale intra-shard duplicates).
- `extra-config` allowlist entry; `SAFE_WEIGHTS_INDEX_NAME` fix.
- Dockerfile: purge base-image stale vllm before AND after editable install
  (namespace stubs shadow the editable finder → `vllm.entrypoints.cli` missing).

**GLM model fixes (ports from promisezackr/glm53-flash-170hx-pp8 + upstream PR)**
- 0003: PP intermediate-tensor factories; mHC `hc_post` materialization at stage
  boundaries (verified pattern; the reduced-model PP equivalence check runs via engine
  boots).
- 0005/0017: draft embed/head loading from checkpoint; shared head unquantized;
  untouched-parameter warning.
- PR #55647: video placeholder timestamps from the pixel path's sampler (+ its test).

**Quantization dispatch**
- `AutoGPTQLinearMethod`: Marlin verification deferred to after kernel selection →
  `TritonW4A16LinearKernel` reachable on ROCm.
- INC scheme: Marlin-less fallback to AutoGPTQ; MTP-draft root remap
  (`model.layers.45.*` → checkpoint block) so draft experts quantize correctly.

**gfx908 kernels**
- W4A16 GEMV MoE path for M ≤ 8 (port of cfac8d0d9): HIP thread-per-column GEMV +
  fused Triton reduces; numerics verified vs dequant reference (0.35% bf16 level);
  split-count divisibility fix for GLM's K=4096. Perf-neutral at depth 2 (recorded).
- Vectorized, chunked, capture-safe gfx908 paged MQA-logits fallback (aiter's Triton
  kernels need fp8 `tl.dot`, unsupported on gfx908): exact vs reference on GPU.
- MLA chunked-prefill workspace capped at `max_num_batched_tokens` (64k-token cap
  reserved multiple GiB per sparse-MLA layer at 1M and starved KV).

**Deployment tooling** (`deploy/glm53-gfx908/`)
- `PLAN.md` (approved plan), `PROGRESS.md` (full log + exact resume commands),
  `SOURCE_MANIFEST.md` (all reference pins), `audit_checkpoint.py`,
  `scripts/build.sh` (reproducible image builds), `scripts/hw_probe.py`,
  `scripts/bench_decode.py`, `scripts/mtp_acceptance_probe.py`, `scripts/smoke_test.sh`,
  `scripts/boot_check.py`, and profiles: `boot-8k-eager`, `boot-8k-mtp`,
  `boot-8k-mtp-graphs{,-d2,-d3}`, `serve-1m-tp2pp4`, `serve-1m`.

## 4. Known issues and negative results (with repro)

1. **MTP depth 3 → HSA memory-access faults** after serving one 200-OK request
   ("Page not present or supervisor privilege", GPUs 5/6/8). Repro:
   `profiles/boot-8k-mtp-graphs-d3.sh` + any chat request. Depth 2 is the working
   production depth. Needs Phase 3 focused kpool/MTP-shape tests before retrying.
2. **Cross-hive direct device copies segfault** (no P2P across PCIe hives;
   `can_device_access_peer(0,4)=False`; HSA SVM/SDMA/TRANSFER knobs don't help).
   PP transfers use RCCL send/recv — verified working cross-hive.
3. **aiter Triton MQA-logits kernels incompatible with gfx908** (fp8 `tl.dot`);
   replaced by the vectorized torch fallback. A native gfx908 kernel is Phase 6 work.
4. **1M context, TP4×PP2: impossible with BF16 KV** — measured KV/token/rank
   (PP0 ~6.9 KiB) → ~35 GiB/rank needed at 1M vs 32 GiB total.
5. **1M context, TP2×PP4 (partition 12,12,12,9): BF16 KV caps at ~526k tokens** —
   tightest rank has 1.62 GiB KV at util 0.97 vs 3.16 GiB needed. Next lever: plan
   fallback #5 (explicit software FP8 MLA cache with its own quality gate).
6. W4A16 GEMV MoE path is perf-neutral at depth 2 on this model (pooled 38.8 vs 39.5);
   kept enabled (numerics-verified, no regression) — the dense-projection GEMV is the
   more promising next port.
7. `prebuild_gfx908_exts.py` failures in the image are expected: those modules are
   Qwen4-fork-only and absent from this GLM branch.

## 5. Next steps (in priority order, per PLAN.md)

1. **Phase 6 — decode perf to ≥50 tok/s**: dense W4A16 GEMV (q_b/kv_b/o_proj at
   M ≤ 8) and skinny-BF16 dense path; math-family MTP acceptance investigation;
   depth-3 fault root-cause (unlock depth 3); re-run the C1 benchmark after each.
2. **Phase 5 — 1M context**: FP8 MLA-cache fallback (plan fallback #5) on TP2×PP4,
   with its own quality gate; then long-context acceptance ladder
   (8K→32K→128K→262k→524k→1M) per plan §3 Phase 5.
3. **Phase 5 — multimodal limits**: video caps (32 frames / 4096 vision tokens),
   mixed-modality tests, video temporal-order checks.
4. **Phase 4 — model quality**: 200-question GSM8K subset + logprob-based
   optimized-vs-eager comparison; forced-MTP-rejection state-corruption tests.
5. **Phase 4 acceptance**: 1000 requests / 2h soak, restart checks.
6. **Phase 7 — final deployment**: immutable image from the pushed branch, restart
   policy, kernel-cache persistence, sanitized results, handoff record per plan §5.

## 6. Exact reproduction commands

```bash
# Verify checkpoint copy (should print 0 files to transfer)
rsync -an --stats /mnt/flash-inference/models/GLM-5.3-Flash-W4A16-AutoRound/ \
  /home/ubuntu/glm-5.3-flash/artifacts/checkpoint-local/ | tail -2

# Audit against the index
python3 /home/ubuntu/glm-5.3-flash/vllm/deploy/glm53-gfx908/audit_checkpoint.py \
  /home/ubuntu/glm-5.3-flash/artifacts/checkpoint-local

# Rebuild the image (logs must go OUTSIDE /tmp/glm53-build)
/home/ubuntu/glm-5.3-flash/vllm/deploy/glm53-gfx908/scripts/build.sh

# Launch the current serving profile (MTP depth 2 + graphs, 8K)
set -a; source /home/ubuntu/glm-5.3-flash/.env; set +a
/home/ubuntu/glm-5.3-flash/vllm/deploy/glm53-gfx908/profiles/boot-8k-mtp-graphs-d2.sh

# Smoke test
curl -s -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:8006/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"glm5.3-flash-autoround","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":64,"temperature":0}'

# C1 benchmark
docker run --rm --network host -v /tmp/bench_c1b.py:/b.py:ro \
  -e VLLM_API_KEY=$VLLM_API_KEY --entrypoint bash glm53f:latest -c 'python3 /b.py'

# Rollback to DSV4
docker rm -f vllm-glm53f && docker start vllm-dsv4
```
