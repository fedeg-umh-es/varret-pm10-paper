#!/usr/bin/env python3
"""Plot exceedance recall profiles for P75 and P90 thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("outputs/tables/exceedance_all_stations.csv")
DEFAULT_OUTPUT_STEM = Path("outputs/figures/figure_exceedance_recall")
MODELS = ["hgb_direct", "ridge_direct", "stl_ridge_direct"]
THRESHOLDS = ["p75", "p90"]
CLASS_COLORS = {"industrial": "#d62728", "urban": "#1f77b4", "rural": "#2ca02c"}


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
    if "station_class" not in df.columns:
        source = df["station_type"] if "station_type" in df.columns else pd.Series(["urban"] * len(df))
        df["station_class"] = source.map(_class_from_type)

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, sharey=True)
    for row_idx, threshold in enumerate(THRESHOLDS):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            subset = df[(df["threshold_type"].eq(threshold)) & (df["model"].eq(model))]
            if subset.empty:
                ax.set_title(model)
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
                continue
            station_col = "station_name" if "station_name" in subset.columns else "dataset"
            for _, station_group in subset.groupby(station_col):
                station_group = station_group.sort_values("horizon")
                color = CLASS_COLORS.get(str(station_group["station_class"].iloc[0]), "#777777")
                ax.plot(station_group["horizon"], station_group["recall"], color=color, alpha=0.35, linewidth=0.8)
            mean_profile = subset.groupby("horizon")["recall"].mean().sort_index()
            ax.plot(mean_profile.index, mean_profile.values, color="black", linewidth=2.1, marker="o", markersize=3)
            ax.set_title(model)
            ax.set_ylim(0, 1.02)
            ax.set_xticks(sorted(subset["horizon"].unique()))
            ax.grid(True, alpha=0.2)
            if col_idx == 0:
                ax.set_ylabel(f"{threshold.upper()} recall")
            if row_idx == 1:
                ax.set_xlabel("horizon")
    handles = [
        plt.Line2D([], [], color=color, linewidth=2, label=label.title())
        for label, color in CLASS_COLORS.items()
    ]
    handles.append(plt.Line2D([], [], color="black", linewidth=2, marker="o", label="Station mean"))
    axes[0, -1].legend(handles=handles, frameon=True, loc="lower left")
    fig.tight_layout()
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.output_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_stem}.pdf/.png")


if __name__ == "__main__":
    main()
