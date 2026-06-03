#!/usr/bin/env python3
"""Plot compact exceedance recall summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("outputs/tables/exceedance_all_stations.csv")
DEFAULT_OUTPUT_STEM = Path("outputs/figures/figure_exceedance_recall")
MODELS = ["persistence", "hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"]
MODEL_LABELS = {
    "persistence": "Persistence",
    "hgb_direct": "HGB",
    "ridge_direct": "Ridge",
    "sarima": "SARIMA",
    "seasonal_naive": "Seasonal naive",
    "stl_ridge_direct": "STL+Ridge",
}
MODEL_COLORS = {
    "persistence": "#4d4d4d",
    "hgb_direct": "#d62728",
    "ridge_direct": "#1f77b4",
    "sarima": "#9467bd",
    "seasonal_naive": "#ff7f0e",
    "stl_ridge_direct": "#2ca02c",
}
MODEL_MARKERS = {
    "persistence": "o",
    "hgb_direct": "o",
    "ridge_direct": "s",
    "sarima": "P",
    "seasonal_naive": "^",
    "stl_ridge_direct": "D",
}
THRESHOLDS = ["abs_50", "p75", "p90"]
THRESHOLD_LABELS = {"abs_50": "50 ug/m3", "p75": "P75", "p90": "P90"}


def _class_from_type(value: str) -> str:
    low = str(value).lower()
    if "industrial" in low:
        return "industrial"
    if "rural" in low or "emep" in low:
        return "rural"
    return "urban"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exceedance recall figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    summary = (
        df[df["model"].isin(MODELS) & df["threshold_type"].isin(THRESHOLDS)]
        .groupby(["model", "threshold_type"], as_index=False)["recall"]
        .mean()
    )
    x = list(range(len(THRESHOLDS)))
    for model in MODELS:
        values = []
        for threshold in THRESHOLDS:
            row = summary[(summary["model"].eq(model)) & (summary["threshold_type"].eq(threshold))]
            values.append(float(row["recall"].iloc[0]) * 100 if not row.empty else float("nan"))
        ax.plot(
            x,
            values,
            color=MODEL_COLORS[model],
            linewidth=2.2,
            marker=MODEL_MARKERS[model],
            markersize=6,
            label=MODEL_LABELS[model],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([THRESHOLD_LABELS[t] for t in THRESHOLDS])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Mean recall across stations and horizons (%)")
    ax.set_xlabel("Exceedance threshold")
    ax.set_title("Event recall reveals missed episodes and warning saturation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True, ncol=2, loc="upper left")
    fig.tight_layout()
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_stem}.pdf/.png")


if __name__ == "__main__":
    main()
