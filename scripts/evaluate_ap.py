#!/usr/bin/env python3
"""Compute exact donor/acceptor AP from a unified per-site score table."""

from __future__ import annotations

import argparse
from array import array
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    labels = {"donor": array("B"), "acceptor": array("B")}
    scores = {"donor": array("f"), "acceptor": array("f")}
    with gzip.open(args.input, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chrom", "pos", "strand", "role", "label", "score"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Unexpected columns in {args.input}: {reader.fieldnames}")
        for row in reader:
            role = row["role"]
            if role not in labels:
                raise ValueError(f"Unexpected role {role!r}")
            labels[role].append(int(row["label"]))
            score = float(row["score"])
            if not np.isfinite(score):
                raise ValueError(f"Non-finite score in {args.input}")
            scores[role].append(score)

    result = {
        "species": args.species,
        "model": args.model,
        "input": str(args.input.resolve()),
        "metric": "sklearn.metrics.average_precision_score (exact tied-score handling)",
    }
    aps = []
    for role in ("donor", "acceptor"):
        y = np.frombuffer(labels[role], dtype=np.uint8)
        s = np.frombuffer(scores[role], dtype=np.float32)
        ap = float(average_precision_score(y, s))
        result[role] = {
            "n_candidates": int(len(y)),
            "n_positives": int(y.sum()),
            "AP": ap,
        }
        aps.append(ap)
    result["mean_AP"] = float(sum(aps) / 2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
