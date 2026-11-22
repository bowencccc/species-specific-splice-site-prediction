#!/usr/bin/env python3
"""Plot the unified four-species splice-site benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[1]

SPECIES = ["human", "chicken", "arabidopsis", "drosophila"]
SPECIES_LABELS = {
    "human": r"Human" + "\n" + r"chr1 + chr9",
    "chicken": r"Chicken" + "\n" + r"chr1",
    "arabidopsis": r"$\it{A.\ thaliana}$" + "\n" + r"chr1",
    "drosophila": r"$\it{D.\ melanogaster}$" + "\n" + r"chrX",
}
MODELS = [
    "alphagenome",
    "spliceai",
    "cnn_mamba_k7_human",
    "cnn_mamba_k7_species",
]
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


def load_mean_values(values_path: Path) -> dict[tuple[str, str], float]:
    result = {}
    with values_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["role"] == "mean":
                result[(row["species"], row["model"])] = float(row["AP"])
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--values", type=Path, default=REPO_ROOT / "results" / "fig1_values.tsv"
    )
    parser.add_argument(
        "--output-stem", type=Path,
        default=REPO_ROOT / "figures" / "fig1_unified_latest",
    )
    args = parser.parse_args()
    values = load_mean_values(args.values)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    group_centers = [0.0, 1.45, 2.90, 4.35]
    bar_width = 0.19

    for center, species in zip(group_centers, SPECIES):
        active = [model for model in MODELS if (species, model) in values]
        offsets = [(i - (len(active) - 1) / 2) * bar_width for i in range(len(active))]
        for offset, model in zip(offsets, active):
            value = values[(species, model)]
            x = center + offset
            ax.bar(
                x,
                value,
                width=bar_width * 0.92,
                color=COLORS[model],
                edgecolor="none",
                zorder=3,
            )
            ax.text(
                x,
                value + 0.016,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
            )

    ax.set_ylabel("Mean AP")
    ax.set_ylim(0, 1.02)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.set_xticks(group_centers, [SPECIES_LABELS[s] for s in SPECIES])
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.85, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", colors="#333333")

    legend_handles = [
        Patch(facecolor=COLORS[model], edgecolor="none", label=MODEL_LABELS[model])
        for model in MODELS
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.105),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        handlelength=0.8,
        columnspacing=1.45,
        handletextpad=0.45,
    )

    fig.text(
        0.075,
        0.012,
        "Mean of donor and acceptor average precision; both strands evaluated on identical canonical candidate sets.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.992, top=0.84, bottom=0.20)

    for suffix in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        args.output_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_stem.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
