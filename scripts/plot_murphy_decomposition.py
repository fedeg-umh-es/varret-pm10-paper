#!/usr/bin/env python3
"""Plot Murphy MSE decomposition by model and horizon."""

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
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.6), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        profile = df[df["model"].eq(model)].groupby("horizon")[[c[0] for c in COMPONENTS]].mean().sort_index()
        bottom = None
        for component, label, color in COMPONENTS:
            ax.bar(profile.index, profile[component], bottom=bottom, color=color, label=label, width=0.75)
            bottom = profile[component] if bottom is None else bottom + profile[component]
        ax.set_title(model)
        ax.set_xlabel("h")
        ax.set_xticks(profile.index)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("MSE component")
    axes[-1].legend(frameon=True, loc="best")
    fig.tight_layout()
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_stem}.pdf/.png")


if __name__ == "__main__":
    main()
