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
- [ ] Not started. DSV4 may keep running during CPU-only build.

## Phase 2 — checkpoint loading
- [ ] Not started.
