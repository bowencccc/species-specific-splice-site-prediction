#!/usr/bin/env python3
"""Compute exact tied-score PR operating points for the unified Fig. 1 data.

This script is deliberately read-only with respect to the existing benchmark.
It consumes the final per-site score tables and fig1_values.tsv, recomputes AP
from distinct score thresholds, aborts on any sanity-check mismatch, and then
writes new TSV/README outputs beside this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("SPLICE_BENCHMARK_ROOT", REPO_ROOT))
SCORE_DIR = Path(
    os.environ.get("SPLICE_SITE_SCORE_DIR", ROOT / "data" / "site_scores")
)
FIG1_VALUES = ROOT / "results" / "fig1_values.tsv"
OUT_DIR = ROOT / "results" / "pr_operating_points"

MODELS = {
    "human": ["alphagenome", "spliceai", "cnn_mamba_k7_human"],
    "chicken": [
        "alphagenome", "spliceai", "cnn_mamba_k7_human",
        "cnn_mamba_k7_species",
    ],
    "arabidopsis": [
        "alphagenome", "spliceai", "cnn_mamba_k7_human",
        "cnn_mamba_k7_species",
    ],
    "drosophila": [
        "alphagenome", "spliceai", "cnn_mamba_k7_human",
        "cnn_mamba_k7_species",
    ],
}
ROLES = ("donor", "acceptor")
RECALL_TARGETS = sorted(
    {0.01, 0.05, 0.10, 0.75, 0.85, 0.95, 0.99}
    | {round(x / 10, 2) for x in range(2, 10)}
)
PRECISION_TARGETS = (0.99, 0.95, 0.90, 0.80, 0.50)
CHUNK_ROWS = 2_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_expected() -> dict[tuple[str, str, str], dict[str, float | int]]:
    expected: dict[tuple[str, str, str], dict[str, float | int]] = {}
    with FIG1_VALUES.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["role"] not in ROLES:
                continue
            expected[(row["species"], row["model"], row["role"])] = {
                "n_candidates": int(row["n_candidates"]),
                "n_positives": int(row["n_positives"]),
                "AP": float(row["AP"]),
            }
    wanted = {
        (species, model, role)
        for species, models in MODELS.items()
        for model in models
        for role in ROLES
    }
    if set(expected) != wanted:
        missing = sorted(wanted - set(expected))
        extra = sorted(set(expected) - wanted)
        raise RuntimeError(f"Unexpected fig1_values rows; missing={missing}, extra={extra}")
    return expected


def load_site_table(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load labels/scores as the same uint8/float32 types used for Fig. 1."""
    label_chunks: dict[str, list[np.ndarray]] = {role: [] for role in ROLES}
    score_chunks: dict[str, list[np.ndarray]] = {role: [] for role in ROLES}
    reader = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        usecols=["role", "label", "score"],
        dtype={"role": "category", "label": np.uint8, "score": np.float32},
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        role_values = chunk["role"].to_numpy()
        labels = chunk["label"].to_numpy(dtype=np.uint8, copy=False)
        scores = chunk["score"].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite score in {path}")
        for role in ROLES:
            keep = role_values == role
            if keep.any():
                label_chunks[role].append(labels[keep].copy())
                score_chunks[role].append(scores[keep].copy())
    result = {}
    for role in ROLES:
        if not label_chunks[role]:
            raise ValueError(f"No {role} rows in {path}")
        result[role] = (
            np.concatenate(label_chunks[role]),
            np.concatenate(score_chunks[role]),
        )
    return result


def exact_curve(labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """PR curve at distinct descending score thresholds, with exact ties."""
    n = len(labels)
    n_pos = int(labels.sum(dtype=np.int64))
    if n_pos <= 0:
        raise ValueError("A role has no positive examples")

    order = np.argsort(scores, kind="quicksort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    del order

    # End index of every distinct-score group. Metrics are evaluated only after
    # the complete tied group has been included.
    ends = np.flatnonzero(
        np.r_[sorted_scores[:-1] != sorted_scores[1:], np.bool_(True)]
    )
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)
    tp = cumulative_tp[ends]
    predicted = ends.astype(np.int64) + 1
    precision = tp / predicted
    recall = tp / n_pos
    thresholds = sorted_scores[ends]
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision, dtype=np.float64))
    return {
        "n_candidates": n,
        "n_positives": n_pos,
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "AP": ap,
    }


def at_recall(curve: dict[str, Any], target: float) -> tuple[float, float, float, float]:
    recall = curve["recall"]
    idx = int(np.searchsorted(recall, target, side="left"))
    if idx >= len(recall):
        idx = len(recall) - 1
    achieved = float(recall[idx])
    threshold = float(curve["thresholds"][idx])
    p_step = float(curve["precision"][idx])
    p_interp = float(np.max(curve["precision"][idx:]))
    return achieved, threshold, p_step, p_interp


def pstep(curve: dict[str, Any], target: float) -> float:
    idx = int(np.searchsorted(curve["recall"], target, side="left"))
    if idx >= len(curve["recall"]):
        idx = len(curve["recall"]) - 1
    return float(curve["precision"][idx])


def psteps(curve: dict[str, Any], targets: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(curve["recall"], targets, side="left")
    idx = np.minimum(idx, len(curve["recall"]) - 1)
    return curve["precision"][idx]


def exact_recall_grid(curve: dict[str, Any], low: float, high: float) -> np.ndarray:
    """All positive-count recall levels in [low, high], shared by all models."""
    n_pos = int(curve["n_positives"])
    first = max(1, int(math.ceil(low * n_pos - 1e-12)))
    last = min(n_pos, int(math.floor(high * n_pos + 1e-12)))
    return np.arange(first, last + 1, dtype=np.float64) / n_pos


def at_precision(curve: dict[str, Any], target: float) -> tuple[float, float, float]:
    keep = curve["precision"] >= target
    if not bool(np.any(keep)):
        return 0.0, math.nan, math.nan
    candidate_idx = np.flatnonzero(keep)
    idx = int(candidate_idx[np.argmax(curve["recall"][candidate_idx])])
    return (
        float(curve["recall"][idx]),
        float(curve["precision"][idx]),
        float(curve["thresholds"][idx]),
    )


def ranges(grid: np.ndarray, mask: np.ndarray) -> list[list[float]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    return [[float(grid[g[0]]), float(grid[g[-1]])] for g in groups]


def verdict(value: bool, partial: bool = False) -> str:
    if value:
        return "TRUE"
    return "PARTLY" if partial else "FALSE"


def claim_rows(curves: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    crossover_grid = np.round(np.arange(0.005, 0.9901, 0.005), 3)

    def ps(species: str, model: str, role: str, values: np.ndarray) -> np.ndarray:
        return psteps(curves[(species, model, role)], values)

    def add(claim_id: str, role: str, numbers: dict[str, Any], result: str) -> None:
        rows.append({
            "claim_id": claim_id,
            "role": role,
            "numbers": json.dumps(numbers, sort_keys=True, separators=(",", ":")),
            "verdict": result,
        })

    for role in ROLES:
        # Human H1
        g = exact_recall_grid(curves[("human", "alphagenome", role)], 0.0, 0.70)
        alpha = ps("human", "alphagenome", role, g)
        min_i = int(np.argmin(alpha))
        ok = bool(np.all(alpha >= 0.98))
        add("H1", role, {
            "recall_grid": "all positive-count recall levels",
            "min_precision_recall_le_0.70": float(alpha[min_i]),
            "min_at_recall": float(g[min_i]),
            "precision_at_0.70": pstep(curves[("human", "alphagenome", role)], 0.70),
        }, verdict(ok))

        # Human H2
        grid = crossover_grid
        h = ps("human", "cnn_mamba_k7_human", role, grid)
        s = ps("human", "spliceai", role, grid)
        delta = h - s
        sign = np.sign(delta)
        switches = []
        for i in range(1, len(grid)):
            if sign[i] != sign[i - 1]:
                switches.append([float(grid[i - 1]), float(grid[i])])
        low = grid < 0.70
        high = grid >= 0.70
        low_fraction = float(np.mean(delta[low] > 0))
        high_fraction = float(np.mean(delta[high] < 0))
        full = low_fraction == 1.0 and high_fraction == 1.0
        partial = low_fraction > 0.5 and high_fraction > 0.5
        add("H2", role, {
            "grid_step": 0.005,
            "crossover_intervals": switches,
            "human_trained_higher_fraction_below_0.70": low_fraction,
            "human_trained_lower_fraction_at_or_above_0.70": high_fraction,
            "human_trained_higher_ranges": ranges(grid, delta > 0),
        }, verdict(full, partial))

        # Human H3
        g = exact_recall_grid(curves[("human", "alphagenome", role)], 0.90, 0.99)
        vals = np.vstack([
            ps("human", model, role, g)
            for model in ("alphagenome", "spliceai", "cnn_mamba_k7_human")
        ])
        spread = vals.max(axis=0) - vals.min(axis=0)
        i = int(np.argmax(spread))
        maximum = float(spread[i])
        add("H3", role, {
            "recall_grid": "all positive-count recall levels",
            "max_pairwise_precision_difference": maximum,
            "at_recall": float(g[i]),
        }, verdict(maximum <= 0.05, maximum <= 0.10))

        # Chicken C1
        g = exact_recall_grid(curves[("chicken", "alphagenome", role)], 0.0, 0.99)
        all_models = MODELS["chicken"]
        vals = {m: ps("chicken", m, role, g) for m in all_models}
        best_other = np.max(np.vstack([vals[m] for m in all_models if m != "alphagenome"]), axis=0)
        advantage = vals["alphagenome"] - best_other
        i = int(np.argmax(advantage))
        highest_fraction = float(np.mean(advantage >= 0))
        add("C1", role, {
            "recall_grid": "all positive-count recall levels through 0.99",
            "alphagenome_highest_fraction": highest_fraction,
            "advantage_gt_0.05_ranges": ranges(g, advantage > 0.05),
            "largest_advantage": float(advantage[i]),
            "largest_advantage_at_recall": float(g[i]),
        }, verdict(highest_fraction > 0.5, highest_fraction > 0.25))

        # Chicken C2
        g = exact_recall_grid(curves[("chicken", "alphagenome", role)], 0.30, 0.75)
        d = ps("chicken", "cnn_mamba_k7_species", role, g) - ps("chicken", "spliceai", role, g)
        fraction = float(np.mean(d > 0))
        add("C2", role, {
            "recall_grid": "all positive-count recall levels",
            "species_trained_gt_spliceai_fraction": fraction,
            "holding_ranges": ranges(g, d > 0),
            "min_difference": float(d.min()),
            "max_difference": float(d.max()),
        }, verdict(fraction == 1.0, fraction > 0.0))

        # Chicken C3
        g = exact_recall_grid(curves[("chicken", "alphagenome", role)], 0.80, 0.95)
        vals3 = np.vstack([
            ps("chicken", model, role, g)
            for model in ("cnn_mamba_k7_species", "cnn_mamba_k7_human", "spliceai")
        ])
        spread = vals3.max(axis=0) - vals3.min(axis=0)
        i = int(np.argmax(spread))
        maximum = float(spread[i])
        add("C3", role, {
            "recall_grid": "all positive-count recall levels",
            "max_pairwise_precision_difference": maximum,
            "at_recall": float(g[i]),
        }, verdict(maximum <= 0.05, maximum <= 0.10))

        # Chicken C4
        c4_values: dict[str, dict[str, float]] = {}
        all_lower = True
        for model in all_models:
            c4_values[model] = {
                "P_at_0.01": pstep(curves[("chicken", model, role)], 0.01),
                "P_at_0.05": pstep(curves[("chicken", model, role)], 0.05),
            }
        for rlabel in ("P_at_0.01", "P_at_0.05"):
            human_p = c4_values["cnn_mamba_k7_human"][rlabel]
            all_lower &= all(
                human_p < c4_values[m][rlabel]
                for m in all_models if m != "cnn_mamba_k7_human"
            )
        add("C4", role, c4_values, verdict(bool(all_lower)))

        # Arabidopsis A1
        curve = curves[("arabidopsis", "cnn_mamba_k7_species", role)]
        r95 = at_precision(curve, 0.95)[0]
        r98 = at_precision(curve, 0.98)[0]
        add("A1", role, {
            "largest_recall_precision_ge_0.95": r95,
            "largest_recall_precision_ge_0.98": r98,
        }, verdict(r95 >= 0.85, r95 >= 0.80))

        # Arabidopsis A2
        zero_models = ("alphagenome", "spliceai", "cnn_mamba_k7_human")
        a2 = {m: pstep(curves[("arabidopsis", m, role)], 0.50) for m in zero_models}
        in_range = [0.70 <= p <= 0.80 for p in a2.values()]
        add("A2", role, {"P_step_at_recall_0.50": a2}, verdict(all(in_range), any(in_range)))

        # Drosophila D1
        curve = curves[("drosophila", "cnn_mamba_k7_species", role)]
        r90 = at_precision(curve, 0.90)[0]
        add("D1", role, {
            "largest_recall_precision_ge_0.90": r90,
        }, verdict(r90 >= 0.75, r90 >= 0.70))

        # Drosophila D2
        d2: dict[str, dict[str, float]] = {}
        for model in zero_models:
            curve = curves[("drosophila", model, role)]
            idx = int(np.searchsorted(curve["recall"], 0.01, side="left"))
            d2[model] = {
                "max_precision_recall_ge_0.01": float(np.max(curve["precision"][idx:])),
                "unrestricted_max_precision": float(np.max(curve["precision"])),
            }
        checks = [v["max_precision_recall_ge_0.01"] <= 0.93 for v in d2.values()]
        add("D2", role, d2, verdict(all(checks), any(checks)))

        # Drosophila D3
        d3 = {m: pstep(curves[("drosophila", m, role)], 0.50) for m in zero_models}
        in_range = [0.50 <= p <= 0.75 for p in d3.values()]
        add("D3", role, {"P_step_at_recall_0.50": d3}, verdict(all(in_range), any(in_range)))

    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global SCORE_DIR, FIG1_VALUES, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, default=SCORE_DIR)
    parser.add_argument("--fig1-values", type=Path, default=FIG1_VALUES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace operating-point TSV/README files in --out-dir.",
    )
    args = parser.parse_args()
    SCORE_DIR, FIG1_VALUES, OUT_DIR = (
        args.score_dir, args.fig1_values, args.out_dir
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        OUT_DIR / "precision_at_recall.tsv",
        OUT_DIR / "recall_at_precision.tsv",
        OUT_DIR / "claims_check.tsv",
        OUT_DIR / "README.md",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; use a new --out-dir or --overwrite: "
            f"{existing}"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    expected = load_expected()
    curves: dict[tuple[str, str, str], dict[str, Any]] = {}
    input_hashes: dict[Path, str] = {FIG1_VALUES: sha256(FIG1_VALUES)}
    sanity_rows = []

    for species, models in MODELS.items():
        for model in models:
            path = SCORE_DIR / f"{species}.{model}.tsv.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            print(f"Loading {path}", flush=True)
            input_hashes[path] = sha256(path)
            table = load_site_table(path)
            for role in ROLES:
                print(f"Computing {species} {model} {role}", flush=True)
                labels, scores = table.pop(role)
                curve = exact_curve(labels, scores)
                curves[(species, model, role)] = curve
                old = expected[(species, model, role)]
                delta = abs(float(curve["AP"]) - float(old["AP"]))
                count_ok = (
                    int(curve["n_candidates"]) == int(old["n_candidates"])
                    and int(curve["n_positives"]) == int(old["n_positives"])
                )
                sanity_rows.append({
                    "species": species,
                    "model": model,
                    "role": role,
                    "n_candidates": int(curve["n_candidates"]),
                    "n_positives": int(curve["n_positives"]),
                    "AP_recomputed": float(curve["AP"]),
                    "AP_fig1_values": float(old["AP"]),
                    "abs_delta": delta,
                    "counts_match": count_ok,
                })
                if not count_ok or delta > 1e-6:
                    raise RuntimeError(
                        "Step 0 FAILED: "
                        f"{species}/{model}/{role}: counts_match={count_ok}, "
                        f"AP_recomputed={curve['AP']:.12g}, AP_table={old['AP']:.12g}, "
                        f"delta={delta:.3g}. No operating-point outputs were written."
                    )
                del labels, scores
            del table

    print("Step 0 PASSED for all species/model/role combinations", flush=True)

    precision_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    for species, models in MODELS.items():
        for model in models:
            for role in ROLES:
                curve = curves[(species, model, role)]
                for target in RECALL_TARGETS:
                    achieved, threshold, p_step, p_interp = at_recall(curve, target)
                    precision_rows.append({
                        "species": species,
                        "model": model,
                        "role": role,
                        "recall_target": f"{target:.2f}",
                        "achieved_recall": f"{achieved:.12g}",
                        "threshold": f"{threshold:.12g}",
                        "precision_step": f"{p_step:.12g}",
                        "precision_interp": f"{p_interp:.12g}",
                    })
                for target in PRECISION_TARGETS:
                    achieved_r, achieved_p, threshold = at_precision(curve, target)
                    recall_rows.append({
                        "species": species,
                        "model": model,
                        "role": role,
                        "precision_target": f"{target:.2f}",
                        "achieved_precision": "NA" if math.isnan(achieved_p) else f"{achieved_p:.12g}",
                        "recall": f"{achieved_r:.12g}",
                        "threshold": "NA" if math.isnan(threshold) else f"{threshold:.12g}",
                    })

    claims = claim_rows(curves)
    write_tsv(
        OUT_DIR / "precision_at_recall.tsv",
        precision_rows,
        ["species", "model", "role", "recall_target", "achieved_recall", "threshold", "precision_step", "precision_interp"],
    )
    write_tsv(
        OUT_DIR / "recall_at_precision.tsv",
        recall_rows,
        ["species", "model", "role", "precision_target", "achieved_precision", "recall", "threshold"],
    )
    write_tsv(
        OUT_DIR / "claims_check.tsv",
        claims,
        ["claim_id", "role", "numbers", "verdict"],
    )

    max_delta = max(float(row["abs_delta"]) for row in sanity_rows)
    readme_lines = [
        "# Fig. 1 exact precision–recall operating points",
        "",
        "This directory was generated without rescoring or modifying any existing benchmark output.",
        "The per-site score tables below are the exact tables used to generate `fig1_values.tsv`.",
        "Scores were parsed as float32, matching the original AP script.",
        "",
        "## Step 0 AP sanity check",
        "",
        f"**PASS:** all {len(sanity_rows)} species × model × role comparisons had exactly matching candidate and positive counts, and recomputed AP matched `fig1_values.tsv` within 1e-6.",
        f"The maximum absolute AP difference from the six-decimal table was `{max_delta:.12g}`.",
        "AP was recomputed at complete distinct-score groups as `sum((R_k - R_(k-1)) * P_k)`.",
        "",
        "## Inputs and SHA256",
        "",
        "| Input | SHA256 |",
        "|---|---|",
    ]
    for path in sorted(input_hashes, key=str):
        readme_lines.append(f"| `{path}` | `{input_hashes[path]}` |")
    readme_lines.extend([
        "",
        "## Output definitions",
        "",
        "- `precision_at_recall.tsv`: `precision_step` is precision at the first complete tied-score threshold reaching the requested recall; `precision_interp` is the maximum precision among all complete thresholds at or above that recall.",
        "- `recall_at_precision.tsv`: greatest recall at a complete tied-score threshold satisfying the precision target.",
        "- `claims_check.tsv`: exact role-specific checks. H2 crossovers use the requested 0.005 recall grid; H1/H3 and Chicken interval analyses use every positive-count recall level.",
        "",
        "For qualitative verdicts, literal numerical bounds were used where supplied. “Converge/similar” means maximum pairwise difference <=0.05 (PARTLY if <=0.10); “most” means >50% of the grid; approximate 0.7–0.8 and 0.5–0.75 statements were checked against those stated ranges. A1/D1 are TRUE when the requested boundary is reached and PARTLY when within 0.05 recall.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/compute_pr_operating_points.py --out-dir results/pr_operating_points_recomputed",
        "```",
        "",
        "The script uses no network access and refuses to overwrite its TSV outputs.",
    ])
    (OUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"Wrote outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
