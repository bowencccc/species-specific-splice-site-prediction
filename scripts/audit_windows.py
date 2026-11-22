#!/usr/bin/env python3
"""Validate prepared NPZ windows against the reported split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def chrom_from_name(path: Path) -> str:
    match = re.match(r"(.+?)_(?:plus|minus)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse chromosome from {path.name}")
    chrom = match.group(1)
    return chrom if chrom.startswith("chr") else "chr" + chrom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True)
    parser.add_argument(
        "--species-config", type=Path,
        default=REPO_ROOT / "configs" / "species.json",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--check-all", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.species_config.read_text())
    if args.species not in config:
        raise ValueError(f"Unknown species {args.species!r}")
    cfg = config[args.species]
    root = args.data_root or REPO_ROOT / cfg["windows"]
    files = sorted(root.glob("*/*.npz"))
    by_chrom = {}
    for path in files:
        by_chrom.setdefault(chrom_from_name(path), []).append(path)

    expected_manifest = json.loads((
        REPO_ROOT / "artifacts" / "checkpoint_metadata" / args.species /
        "split_manifest.json"
    ).read_text())
    failed = False
    selected = []
    for split in ("train", "val", "test"):
        chromosomes = cfg[split]
        paths = [path for chrom in chromosomes for path in by_chrom.get(chrom, [])]
        expected = expected_manifest["splits"][split]["window_count"]
        status = "OK" if len(paths) == expected else "MISMATCH"
        print(f"{status:8s} {split:5s} observed={len(paths):,} expected={expected:,}")
        failed |= len(paths) != expected
        selected.extend(paths if args.check_all else paths[:1])

    for path in selected:
        with np.load(path) as data:
            if set(("sequence", "labels")) - set(data.files):
                raise ValueError(f"Missing arrays in {path}: {data.files}")
            if data["sequence"].shape != (10_000,) or data["labels"].shape != (10_000,):
                raise ValueError(f"Wrong array shape in {path}")
            if not np.isin(data["labels"], (0, 1, 2)).all():
                raise ValueError(f"Invalid splice labels in {path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
