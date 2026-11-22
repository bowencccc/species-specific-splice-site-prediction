#!/usr/bin/env python3
"""Plot exact donor/acceptor precision-sensitivity curves by species."""

from __future__ import annotations

import argparse
from array import array
import csv
import gzip
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
SCORES = REPO_ROOT / "data" / "site_scores"
METRICS = REPO_ROOT / "results" / "metrics"
OUT = REPO_ROOT / "figures"

CONFIG = {
    "human": {
        "label": "Human",
        "chrom": "chr1 + chr9",
        "models": ["alphagenome", "spliceai", "cnn_mamba_k7_human"],
    },
    "chicken": {
        "label": "Chicken",
        "chrom": "chr1",
        "models": [
            "alphagenome", "spliceai", "cnn_mamba_k7_human",
            "cnn_mamba_k7_species",
        ],
    },
    "arabidopsis": {
        "label": r"$\it{A.\ thaliana}$",
        "chrom": "chr1",
        "models": [
            "alphagenome", "spliceai", "cnn_mamba_k7_human",
            "cnn_mamba_k7_species",
        ],
    },
    "drosophila": {
        "label": r"$\it{D.\ melanogaster}$",
        "chrom": "chrX",
        "models": [
            "alphagenome", "spliceai", "cnn_mamba_k7_human",
            "cnn_mamba_k7_species",
        ],
    },
}

MODEL_LABELS = {
    "alphagenome": "AlphaGenome",
    "spliceai": "SpliceAI",
    "cnn_mamba_k7_human": "CNN–Mamba (human-trained)",
    "cnn_mamba_k7_species": "CNN–Mamba (species-trained)",
}
COLORS = {
    "alphagenome": "#287D86",
    "spliceai": "#D95F4F",
    "cnn_mamba_k7_human": "#587A34",
    "cnn_mamba_k7_species": "#C79B19",
}


def load_role(path: Path, role: str) -> tuple[np.ndarray, np.ndarray]:
    labels = array("B")
    scores = array("f")
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["role"] != role:
                continue
            labels.append(int(row["label"]))
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score in {path}")
            scores.append(score)
    return (
        np.frombuffer(labels, dtype=np.uint8),
        np.frombuffer(scores, dtype=np.float32),
    )


def thin_curve(recall: np.ndarray, precision: np.ndarray, limit: int = 6000):
    if len(recall) <= limit:
        return recall, precision
    indices = np.unique(np.linspace(0, len(recall) - 1, limit, dtype=np.int64))
    return recall[indices], precision[indices]


def model_ap(species: str, model: str, role: str) -> float:
    with (METRICS / f"{species}.{model}.json").open() as handle:
        return float(json.load(handle)[role]["AP"])


def plot_species(species: str) -> None:
    cfg = CONFIG[species]
    models = cfg["models"]
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for role in ("donor", "acceptor"):
        expected_labels = None
        for model in models:
            path = SCORES / f"{species}.{model}.tsv.gz"
            print(f"[{species}] loading {role}: {path.name}", flush=True)
            labels, scores = load_role(path, role)
            if expected_labels is None:
                expected_labels = labels.copy()
            elif not np.array_equal(labels, expected_labels):
                raise RuntimeError(
                    f"Candidate labels differ for {species} {role} {model}"
                )
            precision, recall, _ = precision_recall_curve(
                labels, scores, drop_intermediate=True
            )
            recall, precision = thin_curve(recall, precision)
            curves[(role, model)] = (recall.copy(), precision.copy())
            print(
                f"[{species}] {role} {model}: candidates={len(labels):,}, "
                f"positives={int(labels.sum()):,}, plotted_points={len(recall):,}",
                flush=True,
            )
            del labels, scores, precision, recall

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.1), sharex=True, sharey=True)
    for ax, role in zip(axes, ("donor", "acceptor")):
        for model in models:
            recall, precision = curves[(role, model)]
            ap = model_ap(species, model, role)
            ax.plot(
                recall,
                precision,
                color=COLORS[model],
                linewidth=2.0,
                label=f"{MODEL_LABELS[model]} (AP={ap:.3f})",
            )
        ax.set_title(role.capitalize(), fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(color="#D8D8D8", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Sensitivity (Recall)")
        ax.set_box_aspect(1)
        ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    axes[0].set_ylabel("Precision")

    fig.suptitle(f"{cfg['label']} ({cfg['chrom']})", y=0.985, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.955), w_pad=2.3)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"{species}_donor_acceptor_prsn"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    print(f"Saved {stem}.[png|pdf|svg]", flush=True)


def main() -> None:
    global SCORES, METRICS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--species", nargs="+", choices=sorted(CONFIG), default=list(CONFIG)
    )
    parser.add_argument("--score-dir", type=Path, default=SCORES)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    SCORES, METRICS, OUT = args.score_dir, args.metrics_dir, args.output_dir
    for species in args.species:
        plot_species(species)


if __name__ == "__main__":
    main()
