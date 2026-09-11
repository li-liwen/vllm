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
- [ ] Hardware probes (needs DSV4 stopped): deploy/glm53-gfx908/scripts/hw_probe.py
- Build: deploy/glm53-gfx908/scripts/build.sh

## Phase 2 — checkpoint loading
- [ ] Not started.
