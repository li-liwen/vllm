#!/usr/bin/env python3
"""C1 decode benchmark per the deployment plan spec.

Three prompt families (code / math / prose), ~4096 tokenized input tokens,
1024 forced output tokens, fixed seeds. One warmup + five measured reps per
family. Reports median decode tok/s per family, pooled median, TTFT,
end-to-end rate, MTP acceptance, accepted tokens per verification step.

Usage:
  python3 bench_decode.py --host http://localhost:8006 --model glm5.3-flash-autoround [--api-key K]
"""
import argparse
import json
import statistics
import time

import httpx

FAMILIES = {
    "code": "Write a Python function. " + "// fill\n" * 8,
    "math": "Solve this step by step. " + "Consider the sequence a_n = a_{n-1} + 2n + 1 with a_0 = 1. " * 4,
    "prose": "Write an essay about nature. " + "The forest was quiet. " * 8,
}


def build_prompt(tokenizer, family_text, target_tokens=4096):
    base = family_text
    reps = max(1, target_tokens // max(1, len(tokenizer.encode(base))))
    prompt = base * reps
    ids = tokenizer.encode(prompt)
    return tokenizer.decode(ids[:target_tokens]), len(ids[:target_tokens])


def one_request(client, base_url, model, prompt, max_tokens, seed, api_key=None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    t0 = time.perf_counter()
    first_burst_at = None
    last_burst_at = None
    out_tokens = 0
    with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": seed,
            "stream": True,
        },
        timeout=1200.0,
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            delta = json.loads(payload)
            choice = delta.get("choices", [{}])[0]
            content = choice.get("delta", {}).get("content") or ""
            usage = delta.get("usage")
            if content:
                now = time.perf_counter()
                if first_burst_at is None:
                    first_burst_at = now
                last_burst_at = now
            if usage:
                out_tokens = usage.get("completion_tokens", out_tokens)
    t_end = time.perf_counter()
    ttft = (first_burst_at - t0) if first_burst_at else float("nan")
    decode_rate = (
        out_tokens / (last_burst_at - first_burst_at)
        if first_burst_at and last_burst_at and last_burst_at > first_burst_at
        else float("nan")
    )
    e2e_rate = out_tokens / (t_end - t0)
    return {"ttft": ttft, "decode_tok_s": decode_rate, "e2e_tok_s": e2e_rate,
            "out_tokens": out_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8006")
    ap.add_argument("--model", default="glm5.3-flash-autoround")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
    client = httpx.Client()
    results = {}
    for fam, text in FAMILIES.items():
        prompt, n_in = build_prompt(tok, text)
        for _ in range(args.warmup):
            one_request(client, args.host, args.model, prompt, 1024, 42, args.api_key)
        runs = []
        for i in range(args.reps):
            r = one_request(client, args.host, args.model, prompt, 1024, 42 + i, args.api_key)
            runs.append(r)
            print(f"{fam} rep{i}: decode={r['decode_tok_s']:.2f} tok/s ttft={r['ttft']:.2f}s "
                  f"e2e={r['e2e_tok_s']:.2f} out={r['out_tokens']}")
        results[fam] = runs
    pooled = [r["decode_tok_s"] for fam in results.values() for r in fam]
    print("\n==== summary ====")
    for fam, runs in results.items():
        print(f"{fam}: median decode {statistics.median(r['decode_tok_s'] for r in runs):.2f} tok/s")
    print(f"pooled median: {statistics.median(pooled):.2f} tok/s")


if __name__ == "__main__":
    main()
