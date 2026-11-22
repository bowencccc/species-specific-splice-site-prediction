#!/usr/bin/env python3
"""Build the SHA256 manifest for large artifacts excluded from Git."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


SPECIES = ("human", "chicken", "arabidopsis", "drosophila")
MODELS = {
    "human": ("alphagenome", "spliceai", "cnn_mamba_k7_human"),
    "chicken": ("alphagenome", "spliceai", "cnn_mamba_k7_human", "cnn_mamba_k7_species"),
    "arabidopsis": ("alphagenome", "spliceai", "cnn_mamba_k7_human", "cnn_mamba_k7_species"),
    "drosophila": ("alphagenome", "spliceai", "cnn_mamba_k7_human", "cnn_mamba_k7_species"),
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = []
    for species in SPECIES:
        entries.append((
            "checkpoint",
            args.benchmark_root / "checkpoints_cnn_mamba_k7" / species / "best_model.pt",
            Path("checkpoints") / species / "best_model.pt",
        ))
        entries.append((
            "canonical_candidates",
            args.benchmark_root / "candidates" / f"{species}.canonical_candidates.tsv.gz",
            Path("data/candidates") / f"{species}.canonical_candidates.tsv.gz",
        ))
        for model in MODELS[species]:
            name = f"{species}.{model}.tsv.gz"
            entries.append((
                "per_site_scores",
                args.benchmark_root / "site_scores" / name,
                Path("data/site_scores") / name,
            ))

    rows = []
    for role, source, destination in entries:
        if not source.is_file():
            raise FileNotFoundError(source)
        print(f"hashing {source}", flush=True)
        rows.append({
            "role": role,
            "artifact_path": destination.as_posix(),
            "bytes": source.stat().st_size,
            "sha256": digest(source),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
