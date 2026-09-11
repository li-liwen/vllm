#!/usr/bin/env python3
"""Audit the GLM-5.3-Flash AutoRound checkpoint against its safetensors index.

Verifies:
  - every weight_map entry exists on disk and the tensor is present in that file
  - file checksums (sha256) against a manifest, or generate one with --write-manifest
  - the two repair files (model_extra_conv.safetensors, model_extra_tensors.safetensors) exist
  - shape/dtype inventory for packed qweight/qzero/g_idx tensors

Usage:
  python audit_checkpoint.py <checkpoint_dir> [--verify-manifest manifest.sha256]
  python audit_checkpoint.py <checkpoint_dir> --write-manifest manifest.sha256
"""
import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path

REPAIR_FILES = {"model_extra_conv.safetensors", "model_extra_tensors.safetensors"}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safetensors_shapes(path: Path) -> dict:
    """Return {tensor_name: (dtype, shape)} without loading tensor data."""
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {k: (v["dtype"], v["shape"]) for k, v in header.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--write-manifest", type=Path, default=None)
    ap.add_argument("--verify-manifest", type=Path, default=None)
    args = ap.parse_args()

    ckpt = args.checkpoint
    index_path = ckpt / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"FATAL: no index at {index_path}", file=sys.stderr)
        return 1
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    print(f"index entries: {len(weight_map)}")
    files = sorted(set(weight_map.values()))
    print(f"indexed files: {len(files)}")
    missing_files = [f for f in files if not (ckpt / f).exists()]
    if missing_files:
        print(f"FATAL: missing mapped files: {missing_files}", file=sys.stderr)
        return 1
    for rf in sorted(REPAIR_FILES):
        print(f"repair file {'OK' if (ckpt / rf).exists() else 'MISSING'}: {rf}")

    # Load per-file tensor names lazily only for files we need to spot-check,
    # unless the index is small enough to verify fully.
    by_file = {}
    for name, fname in weight_map.items():
        by_file.setdefault(fname, []).append(name)

    errors = []
    total_bytes = 0
    qtensor_kinds = Counter()
    dtype_counts = Counter()
    for fname, names in sorted(by_file.items()):
        p = ckpt / fname
        total_bytes += p.stat().st_size
        avail = safetensors_shapes(p)
        for name in names:
            if name not in avail:
                errors.append(f"{fname}: tensor '{name}' in index but absent from file")
        dtype_counts.update(avail.get(n, ("ABSENT",))[0] for n in names)
        for name in names:
            if name.endswith(("qweight", "qzeros", "g_idx")):
                qtensor_kinds[name.split(".")[-1]] += 1

    if errors:
        print(f"FATAL: {len(errors)} index/file mismatches:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        return 1

    print(f"all {len(weight_map)} indexed tensors present in their mapped files")
    print(f"on-disk bytes (indexed files): {total_bytes / 2**30:.3f} GiB")
    print(f"quant tensor counts: {dict(qtensor_kinds)}")
    print(f"dtype mix of indexed tensors: {dict(dtype_counts)}")

    # Checkpoint config sanity
    cfg = ckpt / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        print(f"arch: {c.get('architectures')}, quant method: "
              f"{c.get('quantization_config', {}).get('quant_method')}, "
              f"bits: {c.get('quantization_config', {}).get('bits')}, "
              f"group_size: {c.get('quantization_config', {}).get('group_size')}")

    if args.write_manifest:
        out = args.write_manifest
        with out.open("w") as f:
            for fname in sorted(files) + sorted(REPAIR_FILES):
                p = ckpt / fname
                if not p.exists():
                    continue
                print(f"hashing {fname} ...", file=sys.stderr)
                f.write(f"{sha256_file(p)}  {fname}\n")
        print(f"manifest written: {out}")
        return 0

    if args.verify_manifest:
        bad = 0
        for line in args.verify_manifest.read_text().splitlines():
            want, fname = line.split()
            print(f"verifying {fname} ...", file=sys.stderr)
            got = sha256_file(ckpt / fname)
            if got != want:
                print(f"MISMATCH: {fname}", file=sys.stderr)
                bad += 1
        print("checksums OK" if bad == 0 else f"{bad} checksum mismatches")
        return 1 if bad else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
