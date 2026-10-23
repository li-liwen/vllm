# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the exact deployment overlay on its pinned vendor package."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packages", type=Path, default=Path("/usr/local/lib/python3.12/dist-packages")
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "patches/manifest.json").read_text())
    for item in manifest["files"]:
        relative = Path(item["target"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid target path")
        payload = root / "patches/files" / (str(relative) + ".source")
        if digest(payload) != item["after"]:
            raise ValueError(f"Payload changed: {relative}")
        ast.parse(payload.read_bytes(), filename=str(relative))
        expected = (
            {item["after"]} if args.verify_only else {item["before"], item["after"]}
        )
        if digest(args.packages / relative) not in expected:
            raise ValueError(f"Unexpected runtime source: {relative}")
    if not args.verify_only:
        for item in manifest["files"]:
            target = args.packages / item["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                (root / "patches/files" / (item["target"] + ".source")).read_bytes()
            )
    print(f"Verified {len(manifest['files'])} deployment source files")


if __name__ == "__main__":
    main()
