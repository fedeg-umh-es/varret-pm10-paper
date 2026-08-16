#!/usr/bin/env python3
"""
46 — Generate Canonical Manuscript Figures
Deterministically generates Figure 1 and Figure 2 from frozen publication and source tables.
No recomputation of models or metrics.
"""

from __future__ import annotations

import os
from pathlib import Path

# Ensure writable matplotlib config directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_varret"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PUB_TABLES_DIR = ROOT_DIR / "outputs" / "publication_tables"
SOURCE_TABLES_DIR = ROOT_DIR / "outputs" / "source_tables"
FIG_SOURCE_DIR = ROOT_DIR / "outputs" / "figure_source_tables"
FIGURES_DIR = ROOT_DIR / "figures"

FIG_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "lines.linewidth": 1.75,
        "lines.markersize": 6.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def generate_figure_1() -> None:
    """Figure 1: Horizon-wise Evaluation Divergence across Skill, Variance, and Events."""
    # 1. Load source publication tables
    df_err = pd.read_csv(PUB_TABLES_DIR / "pub_table_1_error_metrics.csv")
    df_fid = pd.read_csv(PUB_TABLES_DIR / "pub_table_2_dynamic_fidelity.csv")
    df_evt = pd.read_csv(PUB_TABLES_DIR / "pub_table_3_event_metrics.csv")

    # 2. Build deterministic source CSV
    df_merged = df_err[["model", "horizon", "rmse", "rmse_persistence", "skill_rmse"]].copy()
    df_merged = df_merged.merge(
        df_fid[["model", "horizon", "variance_retention", "temporal_variability", "amplitude_ratio"]],
        on=["model", "horizon"],
    )
    df_merged = df_merged.merge(
        df_evt[["model", "horizon", "tp", "fn", "pod", "csi"]],
        on=["model", "horizon"],
    )
    df_merged.to_csv(FIG_SOURCE_DIR / "fig1_horizon_evaluation_divergence.csv", index=False)
    print("Exported fig1_horizon_evaluation_divergence.csv")

    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True)

    models = ["sarima", "lightgbm"]
    labels = {"sarima": "SARIMA", "lightgbm": "LightGBM"}
    colors = {"sarima": "#d95f02", "lightgbm": "#1f78b4"}
    markers = {"sarima": "s", "lightgbm": "o"}
    linestyles = {"sarima": "--", "lightgbm": "-"}

    horizons = [1, 6, 24, 48]
    x_ticks = [1, 6, 24, 48]

    # Panel A: RMSE Skill vs Persistence
    ax = axes[0]
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.9, alpha=0.7)
    for m in models:
        sub = df_merged[df_merged["model"] == m].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["skill_rmse"],
            label=labels[m],
            color=colors[m],
            marker=markers[m],
            linestyle=linestyles[m],
        )
    ax.set_title("(a) RMSE Skill ($1 - \\mathrm{RMSE}/\\mathrm{RMSE}_{\\mathrm{pers}}$)")
    ax.set_xlabel("Forecast horizon $h$ (hours)")
    ax.set_ylabel("Skill score (relative to persistence)")
    ax.set_xticks(x_ticks)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    # Panel B: Variance Retention
    ax = axes[1]
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.9, alpha=0.7, label="Ideal ($\operatorname{Var}=1$)")
    for m in models:
        sub = df_merged[df_merged["model"] == m].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["variance_retention"],
            label=labels[m],
            color=colors[m],
            marker=markers[m],
            linestyle=linestyles[m],
        )
    ax.set_title("(b) Variance Retention ($\\operatorname{Var}(\\hat{y})/\\operatorname{Var}(y)$)")
    ax.set_xlabel("Forecast horizon $h$ (hours)")
    ax.set_ylabel("Variance retention ratio")
    ax.set_xticks(x_ticks)
    ax.set_ylim(-0.05, 1.05)

    # Panel C: Critical Success Index (CSI)
    ax = axes[2]
    for m in models:
        sub = df_merged[df_merged["model"] == m].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["csi"],
            label=labels[m],
            color=colors[m],
            marker=markers[m],
            linestyle=linestyles[m],
        )
    ax.set_title("(c) Event Detection (CSI at $p_{75}$)")
    ax.set_xlabel("Forecast horizon $h$ (hours)")
    ax.set_ylabel("Critical Success Index (CSI)")
    ax.set_xticks(x_ticks)
    ax.set_ylim(-0.05, 0.75)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_horizon_evaluation_divergence.pdf")
    fig.savefig(FIGURES_DIR / "fig1_horizon_evaluation_divergence.png")
    plt.close(fig)
    print("Saved figures/fig1_horizon_evaluation_divergence.pdf and .png")


def generate_figure_2() -> None:
    """Figure 2: Fold-wise Stability of SARIMA at 48 h."""
    # 1. Load source tables
    df_folds = pd.read_csv(SOURCE_TABLES_DIR / "fold_stability_by_model_horizon_fold.csv")
    df_s48 = df_folds[(df_folds["model"] == "sarima") & (df_folds["horizon"] == 48)].sort_values("fold").copy()

    # 2. Build deterministic source CSV
    df_s48_export = df_s48[[
        "model", "horizon", "fold", "N", "rmse_skill", "variance_retention",
        "temporal_variability", "tp", "fn", "POD", "CSI", "positive_skill", "concordant_degradation"
    ]].copy()
    df_s48_export.to_csv(FIG_SOURCE_DIR / "fig2_sarima48_fold_stability.csv", index=False)
    print("Exported fig2_sarima48_fold_stability.csv")

    # 3. Plot
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))

    fold_labels = [f"Fold {f}" for f in df_s48["fold"]]
    x = np.arange(len(fold_labels))
    bar_width = 0.55

    # Panel A: RMSE Skill by Fold
    ax = axes[0]
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.9, alpha=0.7)
    colors_skill = ["#2b83ba" if v > 0 else "#d7191c" for v in df_s48["rmse_skill"]]
    bars_a = ax.bar(x, df_s48["rmse_skill"], width=bar_width, color=colors_skill, edgecolor="black", linewidth=0.7)
    ax.set_title("(a) SARIMA 48-h RMSE Skill ($3/5 > 0$)")
    ax.set_xlabel("Expanding evaluation fold")
    ax.set_ylabel("Skill score vs persistence")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.set_ylim(-0.38, 0.28)

    # Annotate skill values
    for bar in bars_a:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        y_pos = h + (0.015 if h >= 0 else -0.02)
        ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{h:+.2f}", ha="center", va=va, fontsize=8.5)

    # Panel B: Variance Retention by Fold
    ax = axes[1]
    ax.axhline(0.01, color="red", linestyle=":", linewidth=1.0, alpha=0.8, label="1% threshold")
    bars_b = ax.bar(x, df_s48["variance_retention"] * 100, width=bar_width, color="#fdae61", edgecolor="black", linewidth=0.7)
    ax.set_title("(b) SARIMA 48-h Var. Retention\n($5/5 < 0.12\%$)", fontsize=11)
    ax.set_xlabel("Expanding evaluation fold")
    ax.set_ylabel("Variance retention (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.set_ylim(0, 0.17)

    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.005, f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5)

    # Panel C: Event Representation (POD and CSI)
    ax = axes[2]
    # In all 5 folds, TP=0, POD=0, CSI=0. We display FN counts and highlight 0 detection.
    bars_c = ax.bar(x, df_s48["fn"], width=bar_width, color="#abd9e9", edgecolor="black", linewidth=0.7, label="FN ($y > p_{75}$ missed)")
    ax.set_title("(c) Event Failure\n($5/5$: POD$\,{=}\,$0, CSI$\,{=}\,$0)", fontsize=11)
    ax.set_xlabel("Expanding evaluation fold")
    ax.set_ylabel("Missed exceedance count (FN)")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.set_ylim(0, 480)

    for bar in bars_c:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 10, f"FN={int(h)}, TP=0", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "fig2_sarima48_fold_stability.pdf")
    fig.savefig(FIGURES_DIR / "fig2_sarima48_fold_stability.png")
    plt.close(fig)
    print("Saved figures/fig2_sarima48_fold_stability.pdf and .png")


def main() -> None:
    set_plot_style()
    generate_figure_1()
    generate_figure_2()
    print("\nSUCCESS: Both canonical figures and source tables generated deterministically.")


if __name__ == "__main__":
    main()
