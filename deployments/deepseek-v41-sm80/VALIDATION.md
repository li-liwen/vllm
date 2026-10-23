# Archive validation

Capture and checks performed on October 23, 2026, with AI assistance.

| Check | Result |
| --- | --- |
| Installed Python source changed in container writable layer | None |
| Overlay against pinned vendor source | 31 files; original and resulting SHA-256 recorded |
| Rebuilt image's Python hashes versus captured deployment | **2,657 / 2,657 matched** |
| Rebuilt image's native-file hashes versus captured deployment | **19 / 19 matched** |
| Overlay payload integrity and Python parsing | Passed during image build |
| Repository pre-commit and DCO checks | Passed, including Ruff, mypy, Markdown, SPDX, import and accelerator API checks |
| Staged credential/private-key/large-payload scan | Passed; 47 files inspected before the initial push |
| Image build with network disabled | Failed at apt package download; retained as failed attempt |
| Image build with normal package-download access | Passed |
| GPU serving from newly built archive image | Not run; production image remains unchanged |
| Existing production container restarted unchanged | **Healthy**, authenticated discovery 200; missing/invalid credentials 401/401 |
| A100/A800 hardware validation by this archive work | Not run |
| Hopper, Blackwell, AMD validation | Outside this deployment branch's scope |

Reproduction of the archive build:

```bash
docker build -t deepseek-v41-sm80:deployment-snapshot-20261023 .
docker run --rm --runtime runc --network none \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  --entrypoint /opt/deepseek-sm80/.venv/bin/python \
  deepseek-v41-sm80:deployment-snapshot-20261023 \
  /opt/deepseek-sm80/snapshot/install.py --verify-only
```

The workspace also compared every entry in `runtime-python-sha256.json` and
`runtime-native-sha256.txt` against the rebuilt image in a CPU-only container.
This demonstrates source/native-file identity; it does not constitute a fresh
full-model benchmark of the rebuilt image.

Rebuilt archive image ID:
`sha256:b4aba2221ebcff3e72ba54876b98fb1d4ce6b03b466d648267d9b29aea506089`.

Historical qualification belongs to the original image and text-serving
configuration: three warmed rounds, C1 median 65.26 tok/s, C8 aggregate median
231.86 tok/s, four paired retrieval probes through 700,073 input tokens, and
8/8 concurrent retrieval. It does not cover the larger Cartesian retrieval
sweep, full GSM8K evaluation, five-round upstream comparisons, or latest vision
settings. Those remain separate work. The original report and raw artifacts
were preserved without modification; the report SHA is in `provenance.json`.

The latest production container was found stopped after an API worker failed.
Its OOM flag was false. It was restarted unchanged; authenticated model discovery
returned `deepseek-flash`, and Docker reported healthy after the API servers
started at approximately 02:33:20 UTC. The initial 30-minute health checkpoint
did not pass; final readiness took about 33 minutes. This was one manual start
of the same stopped container, with no retry or deployment configuration change.
The prior overnight exit remains a limitation of the captured latest profile.
See the safe records under `evidence/`.
