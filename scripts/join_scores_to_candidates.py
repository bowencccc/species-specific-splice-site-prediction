#!/usr/bin/env python3
"""Join raw streamed SSM motif scores to the shared Figure 1 candidate table."""

from __future__ import annotations

import argparse
import csv
import gzip
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np


def open_text(path: Path, mode: str = "rt"):
    return gzip.open(path, mode, newline="") if path.suffix == ".gz" else path.open(mode, newline="")


def parse_map(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        source, display = value.split("=", 1)
        result[source] = display
    return result


def raw_role(strand: str, motif: str) -> str:
    if motif in {"GT", "AC"}:
        return "donor"
    if motif in {"AG", "CT"}:
        return "acceptor"
    raise ValueError(f"Unexpected motif {motif!r} on {strand}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--raw-scores", type=Path, required=True)
    parser.add_argument(
        "--replacement-minus-scores",
        type=Path,
        default=None,
        help=(
            "Optional corrected minus-strand score table. When supplied, "
            "minus rows in --raw-scores are ignored and replaced by this file."
        ),
    )
    parser.add_argument("--chrom-map", nargs="+", required=True, help="raw_source=display")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    chrom_map = parse_map(args.chrom_map)

    positions: dict[tuple[str, str, str], array] = defaultdict(lambda: array("I"))
    scores: dict[tuple[str, str, str], array] = defaultdict(lambda: array("f"))
    sources = [(args.raw_scores, None if args.replacement_minus_scores is None else "+")]
    if args.replacement_minus_scores is not None:
        sources.append((args.replacement_minus_scores, "-"))
    for source_path, required_strand in sources:
        with open_text(source_path) as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "chrom", "pos0", "strand", "motif", "donor_prob", "acceptor_prob"
            }
            if not required <= set(reader.fieldnames or []):
                raise ValueError(
                    f"Raw score columns are not valid in {source_path}: "
                    f"{reader.fieldnames}"
                )
            for row in reader:
                strand = row["strand"]
                if required_strand is not None and strand != required_strand:
                    continue
                display = chrom_map[row["chrom"]]
                role = raw_role(strand, row["motif"])
                key = (display, strand, role)
                positions[key].append(int(row["pos0"]))
                scores[key].append(float(row[f"{role}_prob"]))

    sorted_groups = {}
    for key in positions:
        pos = np.frombuffer(positions[key], dtype=np.uint32)
        score = np.frombuffer(scores[key], dtype=np.float32)
        order = np.argsort(pos, kind="mergesort")
        sorted_pos = pos[order]
        if len(sorted_pos) > 1 and np.any(sorted_pos[1:] == sorted_pos[:-1]):
            raise RuntimeError(f"Duplicate raw SSM candidate keys in {key}")
        sorted_groups[key] = (sorted_pos, score[order], 0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with open_text(args.candidates) as source, open_text(args.output, "wt") as destination:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["chrom", "pos", "strand", "role", "label", "score"])
        for row in reader:
            key = (row["chrom"], row["strand"], row["role"])
            if key not in sorted_groups:
                raise RuntimeError(f"Missing raw SSM group {key}")
            pos, score, index = sorted_groups[key]
            expected_pos0 = int(row["pos"]) - 1
            if index >= len(pos) or int(pos[index]) != expected_pos0:
                observed = None if index >= len(pos) else int(pos[index])
                raise RuntimeError(f"SSM/candidate mismatch for {key}: expected {expected_pos0}, got {observed}")
            writer.writerow([row["chrom"], row["pos"], row["strand"], row["role"], row["label"], float(score[index])])
            sorted_groups[key] = (pos, score, index + 1)
            counts[key] += 1

    leftovers = {str(k): len(v[0]) - v[2] for k, v in sorted_groups.items() if len(v[0]) != v[2]}
    if leftovers:
        raise RuntimeError(f"Unconsumed raw SSM rows: {leftovers}")
    print(f"Saved {args.output}; rows={sum(counts.values()):,}; groups={dict(counts)}", flush=True)


if __name__ == "__main__":
    main()
