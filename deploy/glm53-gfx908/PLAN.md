# Deploy GLM-5.3-Flash AutoRound W4A16 on 8× MI100

## 1. Objective, decisions, and established facts

Deploy the exact checkpoint at `/mnt/flash-inference/models/GLM-5.3-Flash-W4A16-AutoRound` using a patched build of the user’s vLLM fork.

The approved scope is:

- Stop the existing `vllm-dsv4` container when GPU testing begins, preserve it for rollback, and leave GLM serving afterward.
- Support text, reasoning, tool calling, images, and video.
- Serve a **1,048,576-token total context window**, including prompt and generated output.
- Target **at least 50 decode tokens/second for one active request with approximately 4K prompt tokens and native MTP enabled**.
- Measure long-context performance separately; 50 tok/s at 1M context is not the acceptance target.
- Commit and push progress regularly to a new branch in `li-liwen/vllm`; leave `main` unchanged.

Performance is an acceptance gate to validate, not an established capability of this hardware/model combination.

### Findings from planning

| Item | Observed state | Consequence |
|---|---|---|
| Target repository | Clean `main`, commit `22258a26bc090bccf5473cf681bbe9bac41bd035` | Use this immutable starting point. |
| GPUs | Eight gfx908 devices, approximately 31.984 GiB usable VRAM each | Budget memory per rank, not only across the server. |
| Interconnect | XGMI within GPUs 0–3 and 4–7; PCIe between hives | Start with TP4×PP2, one TP group per hive. |
| Host memory/storage | Approximately 503 GiB RAM and 893 GiB free local disk; checkpoint is on NFS | Stage a verified local copy for repeated loading. |
| Model | `Glm5NextForConditionalGeneration`; 45 base layers: 34 KDA and 11 sparse MLA; one MTP layer | Preserve the hybrid cache manager and budget MTP separately. |
| Quantization | AutoRound 0.15, GPTQ packing, symmetric INT4, group size 128 | Use the existing INC/AutoRound configuration path with ROCm W4A16 dispatch. |
| Checkpoint contents | Approximately **168.805 GiB of indexed tensors** | Header-derived accounting is more reliable than the stale index size summary. |
| Repaired tensors | Index references `model_extra_conv.safetensors` and `model_extra_tensors.safetensors` | These files are required parts of the checkpoint. |
| Duplicate tensors | Original shards contain stale duplicates; some convolution tensors have `[24576,1,4]` versus indexed repairs `[8192,1,4]` | Loading must respect the index at tensor granularity. |
| Current GLM implementation | Already present on main, including AMD KDA and a Triton NoPE sparse-attention path | Extend current implementation instead of replacing it with an older model tree. |
| Pipeline parallelism | GLM’s intermediate-tensor factory is missing; source comments explicitly gate PP off | Port the mHC stage-boundary implementation. |
| MTP | Current loader drops ordinary embedding/head names before loading draft weights | Explicitly load the checkpoint embedding and output head for PP. |
| Runtime available locally | MI100 container with ROCm 7.2.4, PyTorch `2.12.0+git6bbd260`, Triton `3.7.1+gitf0b55c07`, Transformers 5.16.1 | Reuse this pinned toolchain and rebuild vLLM against it. |

The existing DSV4 notes describe earlier HSA/peer-mapping failures. Its current logs show successful request processing, so those notes are diagnostic history—not proof that GLM is currently blocked by the driver.

### Initial memory and placement model

Use **`VLLM_PP_LAYER_PARTITION=24,21`**:

- PP0: base layers 0–23, six sparse MLA layers, input embedding, vision work.
- PP1: base layers 24–44, five sparse MLA layers, output head, and the MTP layer.
- The MTP draft runs with TP4 and PP1 on the final pipeline stage.

At 1,048,576 tokens, a 512-element BF16 MLA cache requires approximately **1 GiB per sparse layer per TP rank**. Six sparse layers per stage therefore require approximately 6 GiB/rank. Compressed indexer keys add approximately 0.193 GiB/rank before alignment, tails, and other cache overhead.

Across the server, weights plus these caches total approximately **218.4 GiB**, before replicated parameters, KDA states, graph pools, activations, communication buffers, and workspaces. Capacity looks plausible, but real allocation measurements must confirm it.

TP8 with replicated BF16 MLA caches would exceed the available memory at 1M once weights are included; it is not the initial production candidate.

## 2. Reference sources and patch decisions

Record the following pins and decisions in a provenance manifest. For individual commits, preserve authorship with `git cherry-pick -x` when the commit applies cleanly and remains appropriate. Otherwise port the relevant changes manually and record source commits in the commit message.

| Reference | Pin | Use |
|---|---|---|
| `btbtyler09/vllm-gfx908`, `mi100-optimized` | `745883c4197e5bacd438f68de2ec509f5a890f50` | gfx908 platform guards, ROCm compatibility fixes, graph handling, W4A16 dispatch, skinny GEMM foundations. |
| Same repository, `qwen38-flash-next` | `808cd1633e8cf8165a232ae79e97de2b74df396a` | Newer small-batch W4A16 kernels, BF16 GEMM tuning, communication and graph improvements. Port selectively. |
| `btbtyler09/mi100-llm-testing` | `d289d25c2140a0661f3e6c8138772db18450b4c0` | Build provenance, benchmark methodology, performance and failure evidence. Qwen results are not GLM predictions. |
| `larkinwc/vllm-gfx908` | `42d53e58525aaf63bcbf522c1ef0efd313eb86e7` | Cross-check gfx908 dtype handling, tuning infrastructure, AITER findings, and recorded negative results. |
| `promisezackr/glm53-flash-170hx-pp8` | `90ec72e9525e90be701e742c70a20c4154418307` | PP/mHC correctness, MTP loading, software FP8 storage, sparse-attention optimization and addressing fixes. |
| `Mrzhiyao/glm53-a800-vllm` | `daeccb983ec84756cde7408b0e29161d492ea2c5` | Final KPool overrides, graph-safe tail handling, and split-KV Triton sparse MLA. Its tests use different weights and larger GPUs. |
| `PixelML/club-170hx` | `ab7000594aabd9a23b59bc619aa2d60ac0fb92a6` | Benchmark receipts and known failure cases. Its published 56.4 tok/s result uses AWQ with MTP, not the user’s exact AutoRound baseline. |
| `btbtyler09/aiter-gfx908` | `a454235de7100387b81f1bd56620c7744307f735` | Pinned AITER implementation already present in the available image; clone under `reference-repo` during implementation if source changes are needed. |

### Specific patch disposition

**MI100 foundation**

Inspect and port necessary changes from:

- `d3bab5eb0`: gfx908 AITER capability defaults.
- `d2e8687f3`: ROCm 7.2.4 rocBLAS/addmm fallback and version-dependent handling.
- `dd9a2b5ef`: graph/speculation compatibility; retain only applicable platform changes.
- `049792827a`, `7db67cd92e`: skinny GEMM support where absent from current main.

Current main already lists gfx908 in CMake. Do not duplicate existing support or spoof a newer GPU architecture.

**GLM PP/MTP**

Use PP8 patches:

- `0003`: intermediate-tensor factory and materialized mHC state at pipeline boundaries.
- `0005`: draft embedding/output-head loading and correct per-verification-row compressed context lengths.
- `0017`: unquantized MTP shared head matching the checkpoint’s BF16 output head.
- `0004`: audit its behavior; main already constructs draft parallel configuration with `pipeline_parallel_size=1`.
- `0007`: audit for remaining host synchronization; main now has PP broadcast helpers, so the old worker patch should not be applied wholesale.
- `0006`: consider fused draft-loop improvements only after basic MTP passes correctness and profiling identifies draft overhead.

**Sparse attention**

Use `0002`, the relevant helpers from `0001`, A800 final overrides, and the arithmetic fix from `0023`.

Keep main’s existing `ROCM_AITER_MLA_SPARSE` interface and NoPE Triton path. Bring in split-KV computation beneath that interface if it improves decode. Preserve existing 64-bit addressing fixes; do not import debugging code that silently clamps invalid indices.

**Performance candidates**

- `0b2406015`: small-M W4A16 GEMV and skinny dispatch.
- `cfac8d0d9`: HIP W4A16 MoE GEMV.
- `164f71d48`: BF16 split-K GEMM for small batches.
- `b7e26c096` and subsequent correctness fixes: optional XGMI push all-reduce.

GLM uses **sigmoid routing, top-8 experts, and SwiGLU limit 10**. Qwen-specific softmax/top-10 routing and unclamped fused activation code cannot be copied unchanged.

Do not enable W4A8, quantize previously BF16 weights to W8, or replace the checkpoint. Preserve the requested W4A16 computation contract. Exclude DFlash-specific patches from this deployment. Larkin’s recorded Marlin-repack regressions are reasons to avoid that path as the default.

**Video**

Port the missing timestamp/frame-sampling alignment fix from [vLLM PR #55647](https://github.com/vllm-project/vllm/pull/55647), pinned to its reviewed commit during implementation. Current main’s GLM processing class still inherits the incompatible timestamp path.

## 3. Implementation phases and validation gates

### Phase 0 — Establish the branch and reproducible workspace

1. Create `feat/glm53-flash-gfx908` from `22258a26bc090bccf5473cf681bbe9bac41bd035`.
2. Keep development in `/home/ubuntu/glm-5.3-flash/vllm`.
3. Add deployment material under `deploy/glm53-gfx908/`, including:
   - Approved plan and source manifest.
   - Build and launch scripts.
   - Deployment profiles.
   - Checkpoint audit and benchmark tools.
   - Progress log with exact resume commands.
4. Keep large artifacts outside the repository: local checkpoint copy, build caches, raw traces, and private logs.
5. Verify GitHub authentication and push only the new branch.
6. Snapshot DSV4’s launch configuration locally for rollback without publishing credentials.
7. Copy the complete checkpoint to local storage under the workspace’s ignored artifact area. Verify file checksums against the source; include both extra safetensors files.
8. Record topology, package versions, current power limits, and GPU health. Leave host driver and power settings unchanged.

**Gate:** reproducible manifest, verified checkpoint copy, clean branch, initial commit pushed. DSV4 can continue running during CPU-only preparation and builds.

### Phase 1 — Build a pinned gfx908 runtime

Use the locally available image as the initial base:

```text
btbtyler09/vllm-rocm-gfx908
@sha256:03f325eb9fb40f21482972d30ade52ba1b47e2223dca66c71d1d3a057d0f0a67
```

This corresponds to the installed `v0.28.0rc7.dev-q38fn` image. Its installed vLLM version differs from the tag’s apparent version; the digest and package manifest are authoritative.

- Rebuild vLLM Python, C++/HIP, and required Rust components from the new branch against the pinned runtime.
- Use Python 3.12, `uv`, and a `.venv` as required by the repository instructions.
- Pin inherited PyTorch/Triton/Transformers versions; prevent dependency resolution from replacing them with incompatible wheels.
- Compile only gfx908 targets.
- Disable inherited Qwen-specific quantization, routing, PLE, and fusion defaults.
- Enable only validated AITER operations. CK/FP8 matrix kernels must not become selected merely because AITER imports successfully.
- Prebuild adopted HIP extensions before graph capture.
- Give every image a source-commit label and preserve compiler/package manifests.

After the image builds, stop DSV4 and run small hardware probes:

- BF16/FP16 matrix multiplication and representative Triton kernels.
- Collectives within each four-GPU hive.
- PP transfers between hives.
- Eight-worker graph capture/replay with changing inputs.

Begin with RCCL collectives and custom all-reduce disabled. If HSA faults recur, reproduce with these small probes before retrying a full model load. Diagnose process visibility, peer access, communicator/stream counts, and graph use individually.

**Gate:** source-built runtime passes compute, communication, and graph probes without invalid memory access or nonfinite outputs. Commit and push.

### Phase 2 — Load the exact AutoRound checkpoint correctly

**Authoritative safetensors loading**

Add an opt-in default-loader setting:

```json
{"safetensors_use_index": true}
```

Expose it through the existing `--model-loader-extra-config` option and use `--load-format safetensors`.

When enabled:

- Read `model.safetensors.index.json`.
- Yield a tensor only from the file named by its `weight_map` entry.
- Ignore stale copies and unindexed extras.
- Fail on a missing mapped file or tensor.
- Preserve normal model-specific skipping of weights belonging to other PP ranks.
- Leave existing loader behavior unchanged when the option is disabled.

This avoids editing or repacking the user’s checkpoint.

**AutoRound/INC dispatch**

- Preserve `auto-round → INCConfig` detection and per-layer exclusion rules.
- Remove the Marlin-only restriction for supported ROCm INT4 linear configurations.
- Route dense INT4 layers to the existing Triton W4A16 implementation, with gfx908 tuning.
- Route quantized experts through the WNA16 Triton backend.
- Preserve group size 128 and sequential GPTQ packing.
- Handle the checkpoint’s stored zero-point convention correctly: sampled `qzeros` words are `0x77777777`, representing effective zero 8 under GPTQ’s offset convention.
- Handle absent `g_idx` as non-activation-ordered groups.
- Keep excluded attention/KDA projections, router, mHC parameters, vision weights, embedding, and output head in their intended floating-point formats.

Use BF16 as the model activation/storage dtype. Any operation-local FP16 optimization must preserve the public dtype contract and pass numerical tests.

**Gate:** actual checkpoint tensors pass packed-weight/dequantization tests; all required tensors load exactly once with expected shapes; no unexplained missing parameters. Commit and push.

### Phase 3 — Complete gfx908 attention and KDA correctness

**KPool and indexer**

- Implement software E4M3FN encoding/decoding where gfx908 cannot execute the required native conversion.
- Keep FP8 as a compact storage format; perform dot products using supported floating-point arithmetic with FP32 accumulation.
- Ensure query encoding, pooled-key encoding, scales, and decoding use the same FP8 format. Do not mix E4M3FN and FNUZ.
- Support both prefill and paged decode logits.
- Preserve exact per-row compressed context bounds during MTP verification; compressed bounds are not consecutive token lengths.
- Preserve incomplete-pool tails, causal masking, padded rows, and request isolation.
- Retain the allocated 2176-wide top-k buffer and its valid-tail semantics.

**Sparse MLA**

- Use main’s BF16, 512-wide NoPE Triton attention as the initial reference implementation.
- On gfx908, bypass unused AITER persistent-metadata allocation/JIT paths when execution will use Triton.
- Preserve full valid history/tail selection rather than truncating to suit an incompatible kernel.
- Apply 64-bit pointer arithmetic before multiplying slots/pages by strides.
- Bound indexer workspaces using query tiling and the existing logits budget. Start with a 128 MiB logits budget; do not allocate buffers proportional to `heads × full_prefill_chunk × 1M`.
- Make buffer writes, initialization, and temporary ownership safe for graph replay.

**KDA and mHC**

- Retain the current AMD KDA implementation.
- Fix any gfx908 compilation constraints revealed by focused tests.
- Validate convolution layout, recurrent-state dtype, chunk boundaries, speculative snapshots, and rejection rollback.
- Preserve mHC and router FP32 accumulation where required.
- Remove device-to-host synchronization from captured paths rather than hiding it behind broad eager fallbacks.

**Gate:** eager and graph-replayed kernel outputs match their references, including mixed requests and MTP verification shapes. Commit and push.

### Phase 4 — Enable TP4×PP2 and native MTP

**Pipeline handoff**

- Add and expose intermediate-tensor factories on the text and multimodal model wrappers.
- For mHC, allocate the actual residual-stream shape `[tokens, 4, 4096]`.
- Materialize deferred `hc_post` at the sending stage.
- Start the receiving stage with the materialized streams and cleared deferred state.
- Respect stage-local weight ownership and vision execution.
- Validate a reduced model’s PP output against an unsplit reference before loading the full model.

**MTP**

- Load layer 45 as the draft layer.
- Explicitly load the ordinary checkpoint embedding and output head into the draft when sharing across PP stages is unavailable.
- Construct the shared output head unquantized, matching the BF16 checkpoint tensor.
- Validate every required draft parameter; a single successfully loaded parameter must not make an incomplete layer appear valid.
- Preserve TP4 for the draft and PP1 draft configuration.
- Keep draft-token broadcasts and acceptance/state updates consistent across all eight workers.
- Test draft depths 1, 2, 3, and 5, including partial and complete rejection.

Boot the full model first at 8K context, eager, without MTP. Establish reference outputs, then enable graphs and MTP.

**Gate:** coherent full-model generation, correct PP handoff, loaded draft parameters, nonzero acceptance, and matching verification/state behavior. Commit and push.

### Phase 5 — Establish 1M capacity and multimodal serving

Start from:

```text
TP=4
PP=2
VLLM_PP_LAYER_PARTITION=24,21
dtype=bfloat16
kv_cache_dtype=auto
block_size=128
gpu_memory_utilization=0.93
max_num_seqs=4
max_num_batched_tokens=2048
prefix_caching=disabled initially
```

Block size 128 satisfies the model’s current `index_kpool × 32` alignment requirement; do not inherit generic MI100 block-size-32 defaults.

Increase tested context through:

```text
8K → 32K → 128K → 262,144 → 524,288 → 1,048,576
```

At each stage record per-rank weights, allocated cache, KDA state, workspace, graph memory, activation peaks, and remaining headroom.

If memory is insufficient, apply this order:

1. Correct accidental replication, oversized workspaces, and retained temporary buffers.
2. Reduce graph capture sizes, then active sequences from four to one, then prefill chunk size to 1024.
3. Test utilization up to 0.95 only with measured stable headroom.
4. Test **TP2×PP4 with partition `12,12,12,9`**, which reduces replicated sparse KV.
5. If BF16 KV still cannot satisfy 1M, implement an explicitly configured software FP8 MLA-cache path with its own quality and performance gate.

Do not silently lower the final context window or enable CPU weight offload.

**Multimodal path**

- Keep the vision encoder and multimodal wrapper enabled.
- Use a gfx908-compatible vision attention implementation, selected independently from text sparse MLA.
- Begin with one image and one video per request; cap sampled video frames at 32 and the image/video vision-token budget at 4096 per item.
- Apply these limits consistently to profiling, processor execution, and placeholder expansion.
- Port the video timestamp fix so placeholders use the same sampled frames and temporal padding as encoded pixels.
- Test short clips, clips with few frames, pre-sampled frames, mixed image/video requests, and request-specific resize/frame overrides.
- Use CPU media decoding initially; NVIDIA NVDEC is not applicable.
- Keep genuine multimodal profiling enabled so memory planning includes the vision workload.

**Gate:** successful real long-context requests, image understanding, video temporal-order understanding, and mixed-modality requests without worker failure. Commit and push.

### Phase 6 — Reach the decode-performance target

Profile the correct implementation before selecting optimizations. Measure time in:

- Expert GEMMs and routing.
- KDA/convolutions.
- mHC and dense projections.
- Indexer/top-k/sparse MLA.
- TP collectives and PP transfers.
- MTP draft, verification, sampling, and host dispatch.

Optimize in this order:

1. HIP graph replay for decode and MTP verification; prewarm all captured shapes.
2. Small-M W4A16 linear and MoE kernels.
3. GLM-compatible fused expert reduction/activation, preserving SwiGLU clamp and routing weights.
4. Skinny and small-batch BF16 projections.
5. Split-KV sparse decode adapted from the SM80 references.
6. XGMI custom all-reduce, confined to validated hive-local groups.
7. MTP depth and scheduler tuning.

For every candidate, retain a switch to the correctness baseline and accept it only after numerical and end-to-end checks.

Sweep:

- MTP off, 1, 2, 3, 5.
- TP4×PP2; TP2×PP4 if it remains a useful capacity/performance candidate.
- Prefill chunks 1024, 2048, 4096.
- Active sequence limits 1, 4, 8 where memory permits.
- Minimal decode graph capture first; larger captured shapes only when useful.

Run the headline benchmark using the **final 1M-capable service configuration**. Select the fastest passing configuration; prefer the simpler configuration when median performance differs by less than 3%.

If throughput remains below 50 tok/s, continue from the measured bottleneck and report the gap explicitly. Do not mark the performance requirement complete because the model merely starts or aggregate throughput exceeds 50.

**Gate:** measured C1 decode acceptance plus correctness and stability. Commit and push each independently validated optimization group.

### Phase 7 — Final deployment and handoff

- Build a final immutable image from the pushed branch.
- Launch `vllm-glm53f`, exposed on port **8006**, with served model name **`glm5.3-flash-autoround`**.
- Use the existing API-key configuration without copying its value into Git.
- Persist compiled-kernel caches and use a restart policy.
- Configure the validated native MTP depth, 1M context, multimodal limits, and memory settings.
- Enable automatic tool choice with `glm47` and reasoning parsing with `glm45`, following the [official GLM-5.3-Flash recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash).
- Leave DSV4 stopped and preserved for rollback.
- Push the final code, deployment instructions, sanitized results, and exact reproduction commands.

## 4. Test and acceptance specification

### Focused regression tests

Extend nearby existing suites rather than building a separate test framework:

| Area | Required scenarios |
|---|---|
| Safetensors loader | Stale duplicate before/after selected file; repaired convolution shape; extra MTP file; missing mapped tensor; default behavior unchanged. |
| AutoRound W4A16 | Actual packed tensors; zero-point offset; group boundaries; TP slicing; BF16 exclusions; dense/shared/routed experts; clamp behavior. |
| FP8 storage | Zero, subnormals, rounding boundaries, saturation, sign, encode/decode agreement, scale-format agreement. |
| KPool | Pool boundaries, incomplete tails, padded rows, mixed lengths, cache reuse, MTP per-row bounds, graph replay. |
| Sparse MLA | 512-wide NoPE; arbitrary valid top-k counts; masked rows; changed block tables; offsets beyond 32-bit products; prefill/decode/verification. |
| KDA/mHC | Chunked versus continuous execution; speculative rollback; PP-boundary equivalence; no stale graph state. |
| Multimodal | Placeholder/encoder count agreement, frame sampling, request overrides, mixed media, malformed-input handling. |
| Distributed | Hive-local collectives, cross-hive PP, graph replay with changing buffers, MTP broadcasts. |

Use existing reference comparisons and tolerances where available. Require exact agreement for packing, index selection where ties are controlled, masks, and cache addressing.

Run relevant repository lint/pre-commit checks on changed files and record commands/results.

### Model quality

- Establish an eager, MTP-off baseline using the exact indexed checkpoint and independently tested kernels.
- Compare optimized and speculative execution using fixed-prefix logprob tests and existing vLLM correctness helpers.
- Run a fixed 200-question GSM8K subset plus English/Chinese instruction, structured-output, and tool-call checks.
- Investigate any deterministic paired accuracy regression; do not excuse it as an optimization tradeoff.
- Exercise forced MTP rejection to detect corrupted recurrent or tail state.
- Check image content, OCR on a controlled image, video event ordering, and mixed text/media responses.

### Decode benchmark

Use three fixed prompt families—code, mathematical reasoning, and prose—with approximately 4096 tokenized input tokens each.

- One warmup and five measured repetitions per family.
- 1024 requested output tokens, fixed seeds and sampling settings.
- Forced output length only for throughput measurement; ordinary EOS behavior tested separately.
- Count tokens from returned token IDs and final usage, including reasoning tokens.
- Record streaming burst sizes so MTP bursts are not mistaken for single tokens.
- Decode rate: tokens emitted after the first token-bearing burst divided by elapsed time from that burst to the final token-bearing burst.
- Report median per family, pooled median, variation, TTFT, end-to-end rate, MTP acceptance, and accepted tokens per verification step.

**Pass:** pooled warm C1 median ≥50 decode tok/s with native MTP enabled on the final 1M-capable configuration. Publish all family results.

Also measure C2/C4/C8 aggregate throughput and latency; these are informational.

### Long-context and stability acceptance

- Actual tokenized prompts at 128K, 256K, 512K, and near the full limit.
- At least one near-limit request with **1,048,576 total prompt-plus-output budget**, reserving sufficient output tokens.
- Unique retrieval targets near 10%, 50%, and 90% of the long prompt, with uncached runs.
- Report retrieval correctness, TTFT, decode rate, peak memory, and preemption.
- Verify over-limit requests fail cleanly.
- Test mixed short/long traffic, cancellations, sequential cache reuse, and multimodal traffic.
- Complete at least **1000 requests and two hours of serving**, including checks after the soak and after a clean restart.

**Pass:** no GPU faults, worker deaths, unexplained nonfinite outputs, persistent state corruption, or memory growth across reuse. A configured context limit alone does not establish 1M support.

## 5. Progress, publication, and context-reset procedure

Every phase ends with:

1. Review the diff and required checks.
2. Commit a coherent unit, preserving source attribution.
3. Update the progress record with results, failures, current image/commit, and next command.
4. Push `feat/glm53-flash-gfx908` to the user’s GitHub fork.

Do not force-push, merge into main, or open an upstream PR. Keep credentials, model weights, compiled binaries, and private raw traces out of Git.

The implementation handoff must record:

- Current branch and last pushed commit.
- Exact base image digest and final image identifier.
- Dependency and source pins.
- Checkpoint checksums and authoritative index behavior.
- Last passing correctness, performance, context, and multimodal gates.
- Active launch configuration and rollback command.
- Known failures with reproduction commands.
- The next implementation step, including any incomplete benchmark.

After a context reset, read the approved plan and progress record, verify Git/container state, and resume from the last completed gate. Do not repeat repository discovery or replace the pinned baseline with a moving upstream branch.

Completion means the code is pushed, GLM remains running, all requested modalities and 1M context are validated, and the measured performance result is reported against the agreed ≥50 tok/s criterion.

