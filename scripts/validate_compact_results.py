#!/usr/bin/env python3
"""Cross-check committed compact results without loading large score tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def close(left: float, right: float, tolerance: float = 5e-7) -> bool:
    return abs(left - right) <= tolerance


def main() -> None:
    values = {}
    with (REPO_ROOT / "results" / "fig1_values.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            values[(row["species"], row["model"], row["role"])] = row

    failures = []
    for path in sorted((REPO_ROOT / "results" / "metrics").glob("*.json")):
        data = json.loads(path.read_text())
        species, model = data["species"], data["model"]
        for role in ("donor", "acceptor"):
            expected = values[(species, model, role)]
            observed = data[role]
            if int(expected["n_candidates"]) != int(observed["n_candidates"]):
                failures.append(f"candidate count: {species} {model} {role}")
            if int(expected["n_positives"]) != int(observed["n_positives"]):
                failures.append(f"positive count: {species} {model} {role}")
            if not close(float(expected["AP"]), float(observed["AP"])):
                failures.append(f"AP: {species} {model} {role}")
        expected_mean = float(values[(species, model, "mean")]["AP"])
        if not close(expected_mean, float(data["mean_AP"])):
            failures.append(f"mean AP: {species} {model}")

    # Candidate counts and positive counts must be identical across models.
    for species in ("human", "chicken", "arabidopsis", "drosophila"):
        for role in ("donor", "acceptor"):
            pairs = {
                (int(row["n_candidates"]), int(row["n_positives"]))
                for (sp, _model, ro), row in values.items()
                if sp == species and ro == role
            }
            if len(pairs) != 1:
                failures.append(f"shared candidate assertion: {species} {role} {pairs}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"OK: {len(values)} Figure-1 rows and all compact metric JSON files agree")


if __name__ == "__main__":
    main()
