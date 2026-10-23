# DeepSeek V4.1 Flash SM80 deployment archive

This branch preserves the local Turbo/vLLM backport deployment, including its
DeepInfer-inspired mHC fusion. Use this directory's Dockerfile and Compose file.
The vLLM checkout at the repository root is a history anchor; **building that
checkout does not recreate this deployment**. Mainstream PP/DSpark support is
developed separately on `feat/dsv41-sm80-pp2-dspark`.

The scope is NVIDIA CUDA capability **8.0 only**, intended for A100/A800-class
devices. Physical measurements here used **eight CMP 170HX 64 GiB GPUs**, not
A100 hardware. The launcher rejects other capabilities and GPU counts. This is
a deployment archive, not a portable upstream SM80 support contribution.

## Reproduction

The Dockerfile pins the vendor runtime and Ubuntu image by digest. Its installer
checks every original file hash before applying 31 exact source payloads. The
`.source` suffix preserves deployed bytes without reformatting archived code.
Original license notices remain inside each payload. All 2,657 deployed Python
file hashes and native-library hashes are retained for comparison. This recipe
uses the vendor's compiled extensions; it does not rebuild current upstream.

Run from this directory on a prepared eight-GPU SM80 host, with Docker's existing
NVIDIA runtime. No driver, firmware, clock or host configuration changes belong
to this procedure. Download model weights separately under their own license.

```bash
docker build -t deepseek-v41-sm80:deployment-snapshot-20261023 .
export MODEL_PATH=/absolute/path/to/DeepSeek-V4.1-Flash-1017
export CACHE_PATH=/absolute/path/to/workspace-cache
mkdir -p "$CACHE_PATH"
# Create a private .env containing VLLM_API_KEY; do not commit it.
docker compose up -d
docker compose ps
```

Use `sudo` where required by the host's Docker permissions. The supplied Compose
file injects authentication from `.env`. Its authenticated health check discovers
the served model `deepseek-flash` on port 8098. The build context excludes `.env`,
results, model files and runtime caches.

## Captured configuration and limits

The source image is
`sha256:710434d84732383acc873a092c4b9b287ed9124acf88602455b9f90d6b641628`.
It combines the pinned v0.13.0 backport, Turbo patches, local PP embedding/auxiliary
ownership and input/cache fixes, and the mHC post/pre producer. TP2, PP2 with
`20,20`, DP2/static EP4, CPU Engram, `fp8_ds_mla`, prefix caching, CUDA graphs and
DSpark **five** draft tokens are preserved. Do not change draft count or enable
eager execution. `VLLM_DEEPINFER_SM80_MHC=0` selects the original mHC path at startup.

The captured **latest** Compose settings include a 472 GiB RAM cap, 32 GiB shared
memory, 16,384-token prefill batches, a 1,048,576-token context, and an image limit
of 50 with an 8,192-token image budget. These settings differ from the historical
text-only qualification. Large-image qualification was incomplete and had failed
capacity/API-worker runs; this archive makes no claim of multimodal qualification.
An image being accepted does not establish reliable fine-text OCR.

The container had exited with code 1 when inspected on October 23, after an API
worker died. Its OOM flag was false; this does not establish the cause of the
worker failure. The same container was restarted with its settings unchanged.
See [VALIDATION.md](VALIDATION.md) for observed checks and outstanding work.

## Attribution and evidence

- [Turbo](https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo/tree/fae324ae62ac5cef31b7d38f5d369618e1cae1fa)
  supplies the existing Ampere deployment foundation. Its A800 results are not
  measurements of this CMP deployment. See `TURBO-NOTICE` and `TURBO-LICENSE`.
- [DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9)
  supplied the mHC producer idea. Its engine, collectives and scheduler were not
  ported. See `DEEPINFER-LICENSE` and `NOTICE`.
- Historical matched deployment-backport measurements were C1 **65.26 tok/s**
  median and C8 **231.86 tok/s** aggregate median, with 23 GPU microbenchmark
  cases, four long retrieval probes through 700,073 actual input tokens, and
  eight concurrent retrieval passes. These are **CMP results**, not comparisons
  with current upstream. Raw evidence and the completed original report remain
  in the workspace; `provenance.json` records the report hash.

AI assistance was used to assemble this archive. No upstream PR is opened by
publishing this branch. Human review and test execution remain required before
the separate upstream contribution can be submitted.
