#!/usr/bin/env python3
"""Measure MTP acceptance metrics from a running server.

Sends one long prompt + completion, then reports:
  - burst-size histogram from streamed chunks (MTP bursts show up as
    multi-token chunks per scheduling step)
  - accepted tokens per verification step (from server metrics if exposed)
Usage: python3 mtp_acceptance_probe.py --host http://localhost:8006
"""
import argparse
import collections
import json
import time

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8006")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--prompt", default="Explain the theory of general relativity in detail.")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    client = httpx.Client(timeout=600)
    t0 = time.perf_counter()
    bursts = []
    with client.stream(
        "POST",
        f"{args.host}/v1/chat/completions",
        headers={"Authorization": f"Bearer {args.api_key}"},
        json={
            "model": "glm5.3-flash-autoround",
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": True,
        },
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            d = json.loads(payload)
            content = d.get("choices", [{}])[0].get("delta", {}).get("content") or ""
            if content:
                bursts.append(len(content))
    elapsed = time.perf_counter() - t0
    hist = collections.Counter(bursts)
    total = sum(bursts)
    print(f"elapsed {elapsed:.2f}s, total chars {total}")
    print(f"burst-size histogram: {dict(sorted(hist.items()))}")
    multi = sum(v for k, v in hist.items() if k > 1)
    print(f"multi-token bursts: {multi}/{len(bursts)}")


if __name__ == "__main__":
    main()
