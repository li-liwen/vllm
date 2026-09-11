#!/usr/bin/env python3
"""Check vLLM boot logs for common gfx908/MI100 failure signatures."""
import re
import sys

FAILURES = [
    ("HSA fault", r"HSA_STATUS_ERROR|Segmentation fault|Memory access fault"),
    ("invalid memory access", r"invalid (memory access|device side)"),
    ("hipBLASLt capture", r"operation not permitted when stream is capturing"),
    ("worker death", r"Worker exiting|EngineCore failed|core dumped"),
    ("NaN outputs", r"nan|NaN in (output|logits)"),
    ("oom", r"out of memory|OutOfMemory"),
    ("pp mismatch", r"size mismatch for|Missing key|Unexpected key"),
    ("quantization", r"Marlin|not supported.*bits"),
]

def main(logfile):
    text = open(logfile, errors="replace").read()
    hits = []
    for name, pat in FAILURES:
        for m in re.finditer(pat, text, re.IGNORECASE):
            line_no = text[: m.start()].count("\n") + 1
            line = text.splitlines()[line_no - 1][:200]
            hits.append(f"{name}: L{line_no}: {line}")
    if hits:
        print("SUSPECT SIGNATURES:")
        for h in hits[:30]:
            print(" ", h)
        sys.exit(1)
    print("no failure signatures found")
    # extract key boot markers
    for pat in (
        r"Loading safetensors.*",
        r"Using \w+ for AutoGPTQLinearMethod.*",
        r"Using '(\w+)' WNA16 MoE backend.*",
        r"Model loading took.*",
        r"Maximum concurrency.*",
        r"GPU KV cache size.*",
    ):
        for m in re.finditer(pat, text):
            print("BOOT:", m.group(0)[:160])

if __name__ == "__main__":
    main(sys.argv[1])
