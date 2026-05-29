#!/usr/bin/env python3
"""Generate KGE diagnostic figures.

AUDIT SOURCE: visual style adapted from scripts/14_generate_skill_alpha_figure.py
and KGE component intent from manuscript_assets/paper_c_kge/paper.tex.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
KGE_PATH = RESULTS_DIR / "kge_horizon_table.csv"
RANK_PATH = RESULTS_DIR / "rank_correlation_kge_phi.csv"
METRIC_LONG_PATH = RESULTS_DIR / "rank_correlation_metric_matrix_long.csv"

COLORS = {
    "hgb_direct": "#008080",
    "ridge_direct": "#2E4053",
    "sarima": "#8E44AD",
    "seasonal_naive": "#F39C12",
    "stl_ridge_direct": "#E74C3C",
}
MARKERS = {1: "o", 2: "^", 3: "s", 4: "p", 5: "h", 6: "8", 7: "D"}


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES_DIR / f"{stem}.pdf"
    png = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf.relative_to(ROOT)}")
    print(f"Wrote {png.relative_to(ROOT)}")


def _iqr_bounds(group: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        group.groupby(["model", "h"], sort=True)[metric]
        .agg(q25=lambda s: s.quantile(0.25), median="median", q75=lambda s: s.quantile(0.75))
        .reset_index()
    )


def figure_components(df: pd.DataFrame) -> None:
    panels = [("r_h", "A. Pearson r"), ("alpha_h", "B. alpha / phi"), ("beta_h", "C. beta")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    for ax, (metric, label) in zip(axes, panels):
        profile = _iqr_bounds(df, metric)
        for model, group in profile.groupby("model", sort=True):
            color = COLORS.get(model)
            x = group["h"].to_numpy(dtype=float)
            median = group["median"].to_numpy(dtype=float)
            q25 = group["q25"].to_numpy(dtype=float)
            q75 = group["q75"].to_numpy(dtype=float)
            ax.plot(x, median, color=color, marker="o", linewidth=2.0, markersize=4.5, label=model)
            ax.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)
        ref = 1.0 if metric in {"alpha_h", "beta_h"} else 0.0
        ax.axhline(ref, color="#5D6D7E", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_title(label, fontsize=11, fontweight="semibold")
        ax.set_xlabel("Horizon")
        ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#BDC3C7", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Median with IQR across stations")
    axes[0].legend(frameon=True, facecolor="white", edgecolor="#D5D8DC", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_kge_components_horizon")


def _rank_corr(group: pd.DataFrame, left: str, right: str) -> float:
    valid = group[["model", left, right]].dropna()
    if valid["model"].nunique() < 3:
        return np.nan
    return float(
        valid[left].rank(ascending=False, method="average").corr(
            valid[right].rank(ascending=False, method="average"), method="spearman"
        )
    )


def figure_rank_heatmap(df: pd.DataFrame) -> None:
    metric_cols = {
        "Skill_h": "Skill_h",
        "phi_h": "phi_h",
        "r_h": "r_h",
        "KGE_h": "KGE_h",
        "KGE_skill_h": "KGE_skill_h",
    }
    labels = list(metric_cols)
    matrix = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    for left, right in combinations(labels, 2):
        values = [
            _rank_corr(group, metric_cols[left], metric_cols[right])
            for _, group in df.groupby(["station", "h"], sort=False)
        ]
        value = float(pd.Series(values).median(skipna=True))
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Median Spearman rank correlation across station-horizon cells", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, "fig_rank_correlation_heatmap")


def figure_r_vs_phi(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for model, group in df.groupby("model", sort=True):
        ax.scatter(
            group["phi_h"],
            group["r_h"],
            s=22,
            alpha=0.58,
            color=COLORS.get(model),
            label=model,
            edgecolors="none",
        )
    ax.axvline(1.0, color="#27AE60", linestyle="-.", linewidth=1.0, alpha=0.8)
    ax.axhline(0.0, color="#5D6D7E", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(r"$\phi_h = \alpha_h = SD(\hat{y}_h) / SD(y_h)$")
    ax.set_ylabel(r"$r_h$ (Pearson correlation)")
    ax.grid(True, linestyle=":", linewidth=0.6, color="#BDC3C7", alpha=0.55)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D5D8DC", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig_rh_vs_phih_scatter")


def main() -> None:
    if not KGE_PATH.exists():
        raise FileNotFoundError(f"Missing KGE table: {KGE_PATH.relative_to(ROOT)}")
    df = pd.read_csv(KGE_PATH)
    figure_components(df)
    figure_rank_heatmap(df)
    figure_r_vs_phi(df)


if __name__ == "__main__":
    main()

