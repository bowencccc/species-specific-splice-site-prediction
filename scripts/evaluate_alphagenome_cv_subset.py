#!/usr/bin/env python3
"""Evaluate the hybrid CNN--Mamba model on the April-16 shared candidates.

The April-16 AlphaGenome table uses 1-based exon-boundary coordinates, while
the unified CNN--Mamba score table uses the 1-based first base of the genomic
canonical dinucleotide.  This script performs the strand/role-aware coordinate
conversion, exact key join, and tied-score average-precision calculation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def open_text(path: Path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="")


def motif_position(boundary: int, strand: str, role: str) -> int:
    """Convert 1-based exon boundary to 1-based genomic motif start."""
    if (strand, role) in {("+", "donor"), ("-", "acceptor")}:
        return boundary + 1
    return boundary - 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphagenome-candidates", type=Path, required=True)
    parser.add_argument("--spliceai-scores", type=Path, required=True)
    parser.add_argument("--cnn-mamba-site-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--per-site-output", type=Path,
        help="Optional joined per-site TSV or TSV.GZ for downstream PR analysis",
    )
    args = parser.parse_args()

    rows = []
    wanted = {}
    with open_text(args.alphagenome_candidates) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            is_canonical = (
                (row["role"] == "donor" and row["oriented_motif"] == "GT")
                or (row["role"] == "acceptor" and row["oriented_motif"] == "AG")
            )
            if not is_canonical:
                continue
            key = (
                row["chrom"],
                motif_position(
                    int(row["position_1based_exon_boundary"]),
                    row["strand"], row["role"],
                ),
                row["strand"], row["role"],
            )
            if key in wanted:
                raise ValueError(f"Duplicate April-16 candidate: {key}")
            wanted[key] = len(rows)
            rows.append({
                "key": key,
                "role": row["role"],
                "label": int(row["label_chess"]),
                "motif": row["oriented_motif"],
                "alphagenome": float(row["alphagenome_probability_mean"]),
                "cnn_mamba": None,
                "spliceai": None,
            })

    chrom_codes = {"chr1": 1, "chr9": 9}

    def encode(chrom, pos, strand, role):
        return (
            (np.asarray(chrom, dtype=np.uint64) << np.uint64(35))
            | (np.asarray(strand, dtype=np.uint64) << np.uint64(34))
            | (np.asarray(role, dtype=np.uint64) << np.uint64(33))
            | np.asarray(pos, dtype=np.uint64)
        )

    target_keys = np.asarray([
        encode(
            chrom_codes[row["key"][0]], row["key"][1],
            row["key"][2] == "-", row["key"][3] == "acceptor",
        )
        for row in rows
    ], dtype=np.uint64)
    target_order = np.argsort(target_keys)
    sorted_targets = target_keys[target_order]
    found = 0
    reader = pd.read_csv(
        args.cnn_mamba_site_scores, sep="\t", compression="gzip",
        usecols=["chrom", "pos", "strand", "role", "label", "score"],
        dtype={"pos": np.uint32, "label": np.uint8, "score": np.float32},
        chunksize=2_000_000,
    )
    for chunk in reader:
        chrom = chunk["chrom"].map(chrom_codes).fillna(0).to_numpy(np.uint64)
        encoded = encode(
            chrom, chunk["pos"].to_numpy(np.uint64),
            (chunk["strand"].to_numpy() == "-").astype(np.uint64),
            (chunk["role"].to_numpy() == "acceptor").astype(np.uint64),
        )
        positions = np.searchsorted(sorted_targets, encoded)
        keep = positions < len(sorted_targets)
        keep[keep] &= sorted_targets[positions[keep]] == encoded[keep]
        if not keep.any():
            continue
        source_indices = np.flatnonzero(keep)
        row_indices = target_order[positions[keep]]
        labels = chunk["label"].to_numpy(np.uint8)[source_indices]
        scores = chunk["score"].to_numpy(np.float32)[source_indices]
        for row_index, label, score in zip(row_indices, labels, scores):
            if rows[int(row_index)]["label"] != int(label):
                raise ValueError(f"Label mismatch for {rows[int(row_index)]['key']}")
            if rows[int(row_index)]["cnn_mamba"] is not None:
                raise ValueError(f"Duplicate CNN--Mamba score for {rows[int(row_index)]['key']}")
            rows[int(row_index)]["cnn_mamba"] = float(score)
            found += 1
    if found != len(rows):
        missing = [row["key"] for row in rows if row["cnn_mamba"] is None]
        raise RuntimeError(f"CNN--Mamba join matched {found}/{len(rows)}; missing={missing[:5]}")

    # The SpliceAI table does not carry the motif column.  Match it to the
    # already-filtered candidate keys rather than inferring canonical status.
    allowed = {row["key"] for row in rows}
    with open_text(args.spliceai_scores) as handle:
        spliceai_rows = []
        for row in csv.DictReader(handle, delimiter="\t"):
            observed = (
                row["chrom"],
                motif_position(
                    int(row["position_1based_exon_boundary"]),
                    row["strand"], row["role"],
                ),
                row["strand"], row["role"],
            )
            if observed in allowed:
                spliceai_rows.append(row)
    if len(spliceai_rows) != len(rows):
        raise RuntimeError(f"SpliceAI row count {len(spliceai_rows)} != {len(rows)}")
    for target, source in zip(rows, spliceai_rows):
        observed = (
            source["chrom"],
            motif_position(
                int(source["position_1based_exon_boundary"]),
                source["strand"], source["role"],
            ),
            source["strand"], source["role"],
        )
        if observed != target["key"] or int(source["label_chess"]) != target["label"]:
            raise RuntimeError(f"SpliceAI candidate mismatch: {observed} != {target['key']}")
        target["spliceai"] = float(source["spliceai_official_ensemble_probability"])

    if args.per_site_output:
        args.per_site_output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = (
            gzip.open(args.per_site_output, "wt", newline="")
            if args.per_site_output.suffix == ".gz"
            else args.per_site_output.open("w", newline="")
        )
        with output_handle as handle:
            fieldnames = [
                "chrom", "pos", "strand", "role", "label",
                "alphagenome", "cnn_mamba", "spliceai",
            ]
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                chrom, pos, strand, role = row["key"]
                writer.writerow({
                    "chrom": chrom,
                    "pos": pos,
                    "strand": strand,
                    "role": role,
                    "label": row["label"],
                    "alphagenome": row["alphagenome"],
                    "cnn_mamba": row["cnn_mamba"],
                    "spliceai": row["spliceai"],
                })

    models = ("alphagenome", "cnn_mamba", "spliceai")
    output_rows = []
    canonical = rows
    for role in ("donor", "acceptor"):
        selected = [row for row in canonical if row["role"] == role]
        labels = np.asarray([row["label"] for row in selected], dtype=np.uint8)
        for model in models:
            scores = np.asarray([row[model] for row in selected], dtype=np.float64)
            output_rows.append({
                "scope": "chr1+chr9_canonical_Apr16_fold3_valid",
                "model": model,
                "role": role,
                "n_candidates": len(labels),
                "n_positives": int(labels.sum()),
                "AP": float(average_precision_score(labels, scores)),
            })
    for model in models:
        role_rows = [row for row in output_rows if row["model"] == model]
        output_rows.append({
            "scope": "chr1+chr9_canonical_Apr16_fold3_valid",
            "model": model,
            "role": "mean",
            "n_candidates": sum(row["n_candidates"] for row in role_rows),
            "n_positives": sum(row["n_positives"] for row in role_rows),
            "AP": sum(row["AP"] for row in role_rows) / 2,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0]), delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    for row in output_rows:
        if row["role"] == "mean":
            print(f"{row['model']}: mean AP={row['AP']:.6f}")


if __name__ == "__main__":
    main()
