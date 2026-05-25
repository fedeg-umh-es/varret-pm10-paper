#!/usr/bin/env python3
"""Plot collapse-rate sensitivity over alpha thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("outputs/tables/threshold_sensitivity.csv")
DEFAULT_OUTPUT_STEM = Path("outputs/figures/figure_threshold_sensitivity")
MODEL_COLORS = {
    "hgb_direct": "#d62728",
    "ridge_direct": "#1f77b4",
    "stl_ridge_direct": "#2ca02c",
    "sarima": "#9467bd",
    "seasonal_naive": "#ff7f0e",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot alpha-threshold collapse sensitivity.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"threshold", "model", "collapse_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Threshold sensitivity table missing columns: {sorted(missing)}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for model, group in df.groupby("model", sort=True):
        group = group.sort_values("threshold")
        ax.plot(
            group["threshold"],
            group["collapse_rate"],
            marker="o",
            linewidth=1.8,
            color=MODEL_COLORS.get(model, "#555555"),
            label=model,
        )
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=0.9, label="Chosen threshold")
    ax.set_xlabel("Alpha threshold")
    ax.set_ylabel("Collapse rate")
    ax.set_ylim(0, 1.04)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_stem}.pdf/.png")


if __name__ == "__main__":
    main()
