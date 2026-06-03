#!/usr/bin/env python3
"""Plot compact Murphy MSE decomposition by model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("outputs/tables/murphy_decomposition_all_stations.csv")
DEFAULT_OUTPUT_STEM = Path("outputs/figures/figure_murphy_decomposition")
MODEL_ORDER = ["hgb_direct", "ridge_direct", "stl_ridge_direct", "sarima", "seasonal_naive"]
COMPONENTS = [
    ("bias_sq", "Bias", "#8dd3c7"),
    ("cond_bias_sq", "Conditional bias", "#fb8072"),
    ("irreducible_sq", "Irreducible", "#80b1d3"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Murphy decomposition.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    if not models:
        models = sorted(df["model"].unique())
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    rows = []
    for model in models:
        mdf = df[df["model"].eq(model)]
        medians = {component: float(mdf[component].median()) for component, _, _ in COMPONENTS}
        total = sum(medians.values())
        row = {"model": model, "total": total}
        for component, _, _ in COMPONENTS:
            row[component] = 100.0 * medians[component] / total if total else 0.0
        rows.append(row)
    profile = pd.DataFrame(rows)

    labels = {
        "hgb_direct": "HGB",
        "ridge_direct": "Ridge",
        "sarima": "SARIMA",
        "seasonal_naive": "Seasonal\nnaive",
        "stl_ridge_direct": "STL+\nRidge",
    }
    x = range(len(profile))
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    bottom = [0.0] * len(profile)
    for component, label, color in COMPONENTS:
        values = profile[component].to_list()
        ax.bar(x, values, bottom=bottom, color=color, width=0.68, label=label, edgecolor="white", linewidth=0.7)
        bottom = [b + v for b, v in zip(bottom, values)]

    for idx, row in profile.iterrows():
        ax.text(idx, 103, f"MSE\n{row['total']:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([labels.get(model, model) for model in profile["model"]])
    ax.set_ylim(0, 116)
    ax.set_ylabel("Median component share of MSE (%)")
    ax.set_title("Murphy decomposition: relative error structure by model")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=True, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_stem}.pdf/.png")


if __name__ == "__main__":
    main()
