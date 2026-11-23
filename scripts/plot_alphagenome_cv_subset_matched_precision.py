#!/usr/bin/env python3
"""Compute and plot matched-precision points for the April-16 shared subset.

Tied scores are entered as a group. For a target recall, ``precision_step`` is
the precision at the first distinct-score threshold whose recall reaches or
exceeds that target. ``precision_interp`` is the maximum precision among all
thresholds at or above the target recall.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("alphagenome", "cnn_mamba", "spliceai")
MODEL_LABELS = {
    "alphagenome": "AlphaGenome",
    "cnn_mamba": "CNN–Mamba-k7",
    "spliceai": "SpliceAI",
}
COLORS = {
    "alphagenome": "#287d8e",
    "cnn_mamba": "#d9a321",
    "spliceai": "#d9584c",
}
MARKERS = {"alphagenome": "o", "cnn_mamba": "s", "spliceai": "^"}
ROLES = ("donor", "acceptor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-ap", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--output-grid", type=Path, required=True)
    parser.add_argument("--figure-prefix", type=Path, required=True)
    parser.add_argument(
        "--targets", type=float, nargs="+",
        default=[0.50, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95],
    )
    parser.add_argument(
        "--figure-targets", type=float, nargs="+",
        default=[0.50, 0.70, 0.80, 0.90],
    )
    parser.add_argument("--grid-step", type=float, default=0.005)
    return parser.parse_args()


def exact_threshold_curve(labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray | float]:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be equally sized one-dimensional arrays")
    if not np.isfinite(scores).all():
        raise ValueError("non-finite score encountered")
    n_positive = int(labels.sum())
    if n_positive == 0:
        raise ValueError("no positive examples")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)[group_ends]
    cumulative_count = group_ends.astype(np.int64) + 1
    cumulative_fp = cumulative_count - cumulative_tp
    precision = cumulative_tp / cumulative_count
    recall = cumulative_tp / n_positive
    thresholds = sorted_scores[group_ends]
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return {
        "threshold": thresholds,
        "tp": cumulative_tp,
        "fp": cumulative_fp,
        "precision": precision,
        "recall": recall,
        "ap": average_precision,
    }


def point_at_recall(curve: dict[str, np.ndarray | float], target: float) -> dict[str, float | int]:
    recall = np.asarray(curve["recall"])
    eligible = np.flatnonzero(recall >= target)
    if not len(eligible):
        raise ValueError(f"target recall {target} cannot be reached")
    index = int(eligible[0])
    return {
        "recall_target": target,
        "achieved_recall": float(recall[index]),
        "threshold": float(np.asarray(curve["threshold"])[index]),
        "precision_step": float(np.asarray(curve["precision"])[index]),
        "precision_interp": float(np.max(np.asarray(curve["precision"])[eligible])),
        "tp": int(np.asarray(curve["tp"])[index]),
        "fp": int(np.asarray(curve["fp"])[index]),
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, sep="\t", compression="infer")
    expected_columns = {"role", "label", *MODELS}
    missing = expected_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")

    expected = pd.read_csv(args.expected_ap, sep="\t")
    expected = expected[expected["role"].isin(ROLES)].set_index(["role", "model"])
    curves: dict[tuple[str, str], dict[str, np.ndarray | float]] = {}
    counts: dict[str, tuple[int, int]] = {}
    for role in ROLES:
        selected = frame[frame["role"] == role]
        labels = selected["label"].to_numpy(np.uint8)
        counts[role] = (len(labels), int(labels.sum()))
        for model in MODELS:
            curve = exact_threshold_curve(labels, selected[model].to_numpy(np.float64))
            expected_ap = float(expected.loc[(role, model), "AP"])
            if abs(float(curve["ap"]) - expected_ap) > 1e-12:
                raise RuntimeError(
                    f"AP mismatch for {role}/{model}: {curve['ap']} != {expected_ap}"
                )
            curves[(role, model)] = curve

    output_rows = []
    for role in ROLES:
        n_candidates, n_positives = counts[role]
        for model in MODELS:
            for target in sorted(set(args.targets)):
                output_rows.append({
                    "scope": "chr1+chr9_canonical_Apr16_fold3_valid",
                    "role": role,
                    "model": model,
                    "n_candidates": n_candidates,
                    "n_positives": n_positives,
                    **point_at_recall(curves[(role, model)], target),
                })
    output = pd.DataFrame(output_rows)
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_table, sep="\t", index=False, float_format="%.9g")

    grid_targets = np.arange(args.grid_step, 1.0, args.grid_step)
    grid_rows = []
    for role in ROLES:
        for model in MODELS:
            for target in grid_targets:
                point = point_at_recall(curves[(role, model)], float(target))
                grid_rows.append({
                    "role": role,
                    "model": model,
                    **point,
                })
    pd.DataFrame(grid_rows).to_csv(
        args.output_grid, sep="\t", index=False, float_format="%.9g",
    )

    figure_targets = sorted(set(args.figure_targets))
    plotted = output[output["recall_target"].isin(figure_targets)]
    if len(plotted) != len(ROLES) * len(MODELS) * len(figure_targets):
        raise ValueError("every figure target must also be present in --targets")

    y_min = max(0.0, float(plotted["precision_step"].min()) - 0.035)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.35), sharey=True)
    for axis, role in zip(axes, ROLES):
        for model in MODELS:
            selected = plotted[(plotted["role"] == role) & (plotted["model"] == model)]
            selected = selected.sort_values("recall_target")
            axis.plot(
                selected["recall_target"], selected["precision_step"],
                label=MODEL_LABELS[model], color=COLORS[model], marker=MARKERS[model],
                linewidth=2.0, markersize=6.5,
            )
        axis.set_title(role.capitalize(), fontsize=14, fontweight="bold")
        axis.set_xlabel("Recall target", fontsize=11)
        axis.set_xlim(min(figure_targets) - 0.035, max(figure_targets) + 0.035)
        axis.set_xticks(figure_targets)
        axis.set_ylim(y_min, 1.005)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Precision at matched recall", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10.5)
    fig.subplots_adjust(bottom=0.22, left=0.085, right=0.985, top=0.90, wspace=0.10)
    args.figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            args.figure_prefix.with_suffix(f".{suffix}"), dpi=300,
            bbox_inches="tight", facecolor="white",
        )
    plt.close(fig)

    print(f"AP verification passed for {len(curves)} role/model combinations")
    print(f"Wrote {args.output_table}")
    print(f"Wrote {args.output_grid}")
    print(f"Wrote {args.figure_prefix}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
