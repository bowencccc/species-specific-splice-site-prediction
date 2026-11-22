#!/usr/bin/env python3
"""Verify external benchmark artifacts against the committed manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO_ROOT / "artifacts" / "external_artifacts.tsv",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--size-only", action="store_true")
    args = parser.parse_args()

    failed = False
    with args.manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            path = args.root / row["artifact_path"]
            if not path.is_file():
                print(f"MISSING  {row['artifact_path']}")
                failed = True
                continue
            if path.stat().st_size != int(row["bytes"]):
                print(f"SIZE     {row['artifact_path']}")
                failed = True
                continue
            if not args.size_only and digest(path) != row["sha256"]:
                print(f"SHA256   {row['artifact_path']}")
                failed = True
                continue
            print(f"OK       {row['artifact_path']}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
