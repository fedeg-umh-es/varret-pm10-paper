#!/usr/bin/env python3
"""
Generate all reproducible figures and tables for the VARRET PM10 paper.

Usage:
    python scripts/make_figures_tables.py

Inputs:
    outputs/tables/master_diagnostic_table.csv
    outputs/tables/murphy_decomposition_all_stations.csv
    outputs/tables/exceedance_all_stations.csv
    outputs/metrics/predictions_all_stations.csv
    data/external/madrid_meteo/*.csv  (from benchmark-pm-hstar)
    data/external/ireland_meteo/*.csv (from benchmark-pm-hstar)

Outputs:
    outputs/figures/fig01_station_map.{pdf,png}
    outputs/figures/fig02_evaluation_workflow.{pdf,png}
    outputs/figures/fig03_station_horizon_heatmap.{pdf,png}
    outputs/figures/fig04_horizon_distribution.{pdf,png}
    outputs/figures/fig05_skill_variance_plane.{pdf,png}
    outputs/figures/fig06_station_collapse_rates.{pdf,png}
    outputs/figures/fig07_exceedance_diagnostics.{pdf,png}
    outputs/figures/fig08_murphy_decomposition.{pdf,png}
    outputs/figures/fig09_episode_timeseries.{pdf,png}
    outputs/figures/fig10_meteo_delta_skill_by_horizon.{pdf,png}
    outputs/figures/fig11_meteo_phi_comparison.{pdf,png}
    outputs/figures/fig12_meteo_ablation.{pdf,png}
    outputs/tables/table01_model_family_diagnostic_summary.{csv,tex}
    outputs/tables/table02_horizon_diagnostic_summary.{csv,tex}
    outputs/tables/table03_station_diagnostic_summary.{csv,tex}
    outputs/tables/table04_skill_retention_quadrants.{csv,tex}
    outputs/tables/table05_exceedance_summary.{csv,tex}
    outputs/tables/table06_murphy_decomposition_summary.{csv,tex}
    outputs/tables/table07_figure_source_mapping.csv
    outputs/tables/table08_meteo_value_added_summary.{csv,tex}
    outputs/tables/table09_meteo_dm_significance.{csv,tex}

This script does NOT modify any .tex manuscript files.
"""

import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FAMILY_MAP = {
    "hgb_direct": "Direct ML",
    "ridge_direct": "Direct ML",
    "sarima": "Statistical baseline",
    "seasonal_naive": "Variance-preserving naive",
    "stl_ridge_direct": "Decomposition + Ridge",
}

FAMILY_ORDER = [
    "Direct ML",
    "Decomposition + Ridge",
    "Statistical baseline",
    "Variance-preserving naive",
]

FAMILY_COLORS = {
    "Direct ML": "#d62728",
    "Decomposition + Ridge": "#2ca02c",
    "Statistical baseline": "#1f77b4",
    "Variance-preserving naive": "#ff7f0e",
}

FAMILY_MARKERS = {
    "Direct ML": "o",
    "Decomposition + Ridge": "s",
    "Statistical baseline": "^",
    "Variance-preserving naive": "D",
}

generated_figures = []
generated_tables = []
omitted = []


def log(msg):
    print(f"[make_figures_tables] {msg}")


def savefig(fig, stem):
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures.append(stem)
    log(f"  FIGURE saved: {stem}.pdf / .png")


def save_table(df, stem, latex=True):
    csv_path = TAB_DIR / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    generated_tables.append(f"{stem}.csv")
    log(f"  TABLE saved: {stem}.csv ({len(df)} rows)")
    if latex:
        tex_path = TAB_DIR / f"{stem}.tex"
        df.to_latex(tex_path, index=False, float_format="%.4f", escape=True)
        generated_tables.append(f"{stem}.tex")
        log(f"  TABLE saved: {stem}.tex")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_master():
    p = ROOT / "outputs" / "tables" / "master_diagnostic_table.csv"
    if not p.exists():
        sys.exit(f"CRITICAL: {p} not found")
    df = pd.read_csv(p)
    df["model_family"] = df["model"].map(MODEL_FAMILY_MAP)
    log(f"Loaded master_diagnostic_table.csv: {len(df)} rows, {df['station_id'].nunique()} stations, {df['model'].nunique()} models")
    return df


def load_murphy():
    p = ROOT / "outputs" / "tables" / "murphy_decomposition_all_stations.csv"
    if not p.exists():
        log(f"WARNING: {p} not found — Murphy figures/tables will be skipped")
        return None
    df = pd.read_csv(p)
    df["model_family"] = df["model"].map(MODEL_FAMILY_MAP)
    df["station_id"] = df["dataset"].str.replace("e1_rr_", "", regex=False)
    log(f"Loaded murphy_decomposition_all_stations.csv: {len(df)} rows")
    return df


def load_exceedance():
    p = ROOT / "outputs" / "tables" / "exceedance_all_stations.csv"
    if not p.exists():
        log(f"WARNING: {p} not found — exceedance figures/tables will be skipped")
        return None
    df = pd.read_csv(p)
    df["model_family"] = df["model"].map(MODEL_FAMILY_MAP)
    log(f"Loaded exceedance_all_stations.csv: {len(df)} rows")
    return df


def load_predictions():
    p = ROOT / "outputs" / "metrics" / "predictions_all_stations.csv"
    if not p.exists():
        log(f"WARNING: {p} not found — episode time series will be skipped")
        return None
    log(f"Loading predictions_all_stations.csv (large file)...")
    df = pd.read_csv(p)
    df["station_id"] = df["dataset"].str.replace("e1_rr_", "", regex=False)
    df["date"] = pd.to_datetime(df["date"])
    log(f"Loaded predictions_all_stations.csv: {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# FIGURE 1 — Station map
# ---------------------------------------------------------------------------
def fig01_station_map(master):
    log("Generating fig01_station_map...")
    stations = master.groupby("station_id").agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        station_type=("station_type", "first"),
        station_class=("station_class", "first"),
        collapse_rate=("collapse_flag", "mean"),
        station_name=("station_name", "first"),
    ).reset_index()

    class_colors = {"urban": "#d62728", "industrial": "#ff7f0e", "rural": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in stations.iterrows():
        c = class_colors.get(row["station_class"], "#999999")
        size = 40 + 260 * row["collapse_rate"]
        ax.scatter(row["lon"], row["lat"], s=size, c=c, edgecolors="k",
                   linewidths=0.5, alpha=0.85, zorder=5)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("PM$_{10}$ monitoring stations — NE Spain")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    handles = [mpatches.Patch(color=c, label=k.capitalize()) for k, c in class_colors.items()]
    size_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
               markersize=np.sqrt(40 + 260 * v), label=f"{v:.0%} collapse")
        for v in [0.0, 0.5, 1.0]
    ]
    leg1 = ax.legend(handles=handles, title="Station class", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=size_legend, title="Collapse rate", loc="lower left", fontsize=8)

    savefig(fig, "fig01_station_map")


# ---------------------------------------------------------------------------
# FIGURE 2 — Evaluation workflow (conceptual)
# ---------------------------------------------------------------------------
def fig02_evaluation_workflow():
    log("Generating fig02_evaluation_workflow...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.5, 4.5, "PM$_{10}$\nobservations"),
        (2.3, 4.5, "Chronological\nrolling-origin\nsplit"),
        (4.3, 4.5, "Train-only\npreprocessing"),
        (6.3, 4.5, "Model fitting\n(5 models)"),
        (8.3, 4.5, "Horizon-wise\nforecasts\n(h = 1…7)"),
        (2.3, 2.0, "Persistence\nbaseline"),
        (5.3, 2.0, "Forecast\nverification\ntable"),
        (8.3, 2.0, "Diagnostic\noutputs\n(figures/tables)"),
    ]

    bw, bh = 1.6, 1.2
    for x, y, txt in boxes:
        rect = mpatches.FancyBboxPatch((x, y - bh / 2), bw, bh,
                                        boxstyle="round,pad=0.1",
                                        facecolor="#e8f0fe", edgecolor="#4a86c8",
                                        linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + bw / 2, y, txt, ha="center", va="center", fontsize=7.5,
                fontweight="bold")

    arrows = [
        (0.5 + bw, 4.5, 2.3, 4.5),
        (2.3 + bw, 4.5, 4.3, 4.5),
        (4.3 + bw, 4.5, 6.3, 4.5),
        (6.3 + bw, 4.5, 8.3, 4.5),
        (8.3 + bw / 2, 4.5 - bh / 2, 8.3 + bw / 2, 2.0 + bh / 2),
        (2.3 + bw / 2, 4.5 - bh / 2, 2.3 + bw / 2, 2.0 + bh / 2),
        (2.3 + bw, 2.0, 5.3, 2.0),
        (5.3 + bw, 2.0, 8.3, 2.0),
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#4a86c8", lw=1.5))

    notes = [
        (1.4, 5.4, "No random split", "italic"),
        (4.3, 5.4, "Train-only scaling", "italic"),
        (6.3, 1.0, "Persistence-relative comparison\nVariance-retention diagnostics", "italic"),
    ]
    for x, y, txt, style in notes:
        ax.text(x, y, txt, fontsize=7, fontstyle=style, color="#666666")

    ax.set_title("Leakage-free rolling-origin evaluation workflow", fontsize=11, fontweight="bold")
    savefig(fig, "fig02_evaluation_workflow")


# ---------------------------------------------------------------------------
# FIGURE 3 — Station × horizon heatmap (alpha)
# ---------------------------------------------------------------------------
def fig03_station_horizon_heatmap(master):
    log("Generating fig03_station_horizon_heatmap...")
    ml_only = master[master["model_family"] == "Direct ML"].copy()
    pivot = ml_only.groupby(["station_name", "horizon"])["alpha"].median().reset_index()
    heatmap = pivot.pivot(index="station_name", columns="horizon", values="alpha")
    station_order = heatmap.mean(axis=1).sort_values().index
    heatmap = heatmap.loc[station_order]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(heatmap.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1.0)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels([f"h={h}" for h in heatmap.columns])
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index, fontsize=7)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("Station")
    ax.set_title("Variance retention ratio (α) — Direct ML models\n(station × horizon median)")

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            val = heatmap.values[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.25 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("α (variance retention ratio)")
    ax.axhline(-0.5, color="k", lw=0.5)

    savefig(fig, "fig03_station_horizon_heatmap")


# ---------------------------------------------------------------------------
# FIGURE 4 — Horizon-wise distribution
# ---------------------------------------------------------------------------
def fig04_horizon_distribution(master):
    log("Generating fig04_horizon_distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, var, label in zip(axes, ["skill", "alpha"],
                               ["Persistence-relative skill", "Variance retention (α)"]):
        data_by_h = [master[master["horizon"] == h][var].dropna()
                     for h in sorted(master["horizon"].unique())]
        bp = ax.boxplot(data_by_h, tick_labels=[str(h) for h in sorted(master["horizon"].unique())],
                        patch_artist=True, showfliers=True, flierprops=dict(markersize=3))
        for patch in bp["boxes"]:
            patch.set_facecolor("#b3cde3")
        ax.set_xlabel("Forecast horizon (days)")
        ax.set_ylabel(label)
        if var == "skill":
            ax.axhline(0, color="red", ls="--", lw=1, alpha=0.7, label="skill = 0")
            ax.legend(fontsize=8)
        if var == "alpha":
            ax.axhline(1, color="green", ls="--", lw=1, alpha=0.7, label="α = 1")
            ax.axhline(0.5, color="red", ls=":", lw=1, alpha=0.7, label="collapse (α = 0.5)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Diagnostic distributions by forecast horizon (all models × stations)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig04_horizon_distribution")


# ---------------------------------------------------------------------------
# FIGURE 5 — Skill–variance-retention plane
# ---------------------------------------------------------------------------
def fig05_skill_variance_plane(master):
    log("Generating fig05_skill_variance_plane...")
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axhspan(-999, 0, color="#fde0dd", alpha=0.3, label="Negative skill")
    ax.axvspan(0, 0.5, color="#fee8c8", alpha=0.3, label="Collapsed (α < 0.5)")
    ax.axvspan(0.8, 1.2, color="#e0f3db", alpha=0.2, label="Near-retained (0.8 ≤ α ≤ 1.2)")
    ax.axhline(0, color="red", ls="--", lw=1, alpha=0.7)
    ax.axvline(1, color="green", ls="--", lw=1, alpha=0.5)

    for fam in FAMILY_ORDER:
        sub = master[master["model_family"] == fam]
        ax.scatter(sub["alpha"], sub["skill"],
                   c=FAMILY_COLORS[fam], marker=FAMILY_MARKERS[fam],
                   s=20, alpha=0.5, label=fam, edgecolors="none")

    ax.set_xlabel("Variance retention ratio (α)")
    ax.set_ylabel("Persistence-relative skill (1 − MSE/MSE_pers)")
    ax.set_title("Skill vs. variance retention — all model × station × horizon cells")
    ax.set_xlim(-0.05, max(2.1, master["alpha"].quantile(0.99) + 0.1))
    ax.set_ylim(master["skill"].quantile(0.01) - 0.1, master["skill"].max() + 0.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    savefig(fig, "fig05_skill_variance_plane")


# ---------------------------------------------------------------------------
# FIGURE 6 — Station-level collapse rates
# ---------------------------------------------------------------------------
def fig06_station_collapse_rates(master):
    log("Generating fig06_station_collapse_rates...")
    collapse = master.groupby(["station_name", "station_class"]).agg(
        collapse_rate=("collapse_flag", "mean"),
    ).reset_index().sort_values("collapse_rate")

    class_colors = {"urban": "#d62728", "industrial": "#ff7f0e", "rural": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(range(len(collapse)), collapse["collapse_rate"] * 100,
                   color=[class_colors.get(c, "#999") for c in collapse["station_class"]],
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(collapse)))
    ax.set_yticklabels(collapse["station_name"], fontsize=7)
    ax.set_xlabel("Collapse rate (% of cells with α < 0.5)")
    ax.set_title("Station-level variance collapse rates (all models × horizons)")
    ax.grid(True, axis="x", alpha=0.3)

    handles = [mpatches.Patch(color=c, label=k.capitalize()) for k, c in class_colors.items()]
    ax.legend(handles=handles, title="Station class", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig06_station_collapse_rates")


# ---------------------------------------------------------------------------
# FIGURE 7 — Exceedance diagnostics
# ---------------------------------------------------------------------------
def fig07_exceedance_diagnostics(exc_df):
    if exc_df is None:
        omitted.append(("fig07_exceedance_diagnostics", "exceedance data not found"))
        return
    log("Generating fig07_exceedance_diagnostics...")

    thresholds = sorted(exc_df["threshold_type"].unique())
    n_thresh = len(thresholds)
    fig, axes = plt.subplots(1, n_thresh, figsize=(5 * n_thresh, 5), sharey=True)
    if n_thresh == 1:
        axes = [axes]

    for ax, thr in zip(axes, thresholds):
        sub = exc_df[exc_df["threshold_type"] == thr]
        agg = sub.groupby("model_family").agg(
            median_recall=("recall", "median"),
            median_f1=("f1", "median"),
        ).reindex(FAMILY_ORDER).dropna()

        x = np.arange(len(agg))
        w = 0.35
        ax.bar(x - w / 2, agg["median_recall"], w, label="Recall", color="#1f77b4", alpha=0.8)
        ax.bar(x + w / 2, agg["median_f1"], w, label="F1", color="#ff7f0e", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=30, ha="right", fontsize=7)
        ax.set_title(f"Threshold: {thr}", fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel("Median metric value")
            ax.legend(fontsize=8)

    fig.suptitle("Exceedance detection diagnostics by model family and threshold", fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig07_exceedance_diagnostics")


# ---------------------------------------------------------------------------
# FIGURE 8 — Murphy decomposition
# ---------------------------------------------------------------------------
def fig08_murphy_decomposition(murphy_df):
    if murphy_df is None:
        omitted.append(("fig08_murphy_decomposition", "Murphy data not found"))
        return
    log("Generating fig08_murphy_decomposition...")

    agg = murphy_df.groupby("model_family").agg(
        bias_sq=("bias_sq", "median"),
        cond_bias_sq=("cond_bias_sq", "median"),
        irreducible_sq=("irreducible_sq", "median"),
        mse=("mse", "median"),
    ).reindex(FAMILY_ORDER).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agg))
    w = 0.6
    bottom = np.zeros(len(agg))
    components = [
        ("bias_sq", "Bias²", "#d62728"),
        ("cond_bias_sq", "Conditional bias²", "#ff7f0e"),
        ("irreducible_sq", "Irreducible²", "#1f77b4"),
    ]
    for col, label, color in components:
        ax.bar(x, agg[col], w, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += agg[col].values

    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Median MSE component (µg/m³)²")
    ax.set_title("Murphy-style MSE decomposition by model family")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, "fig08_murphy_decomposition")


# ---------------------------------------------------------------------------
# FIGURE 9 — Episode time series
# ---------------------------------------------------------------------------
def fig09_episode_timeseries(pred_df, master):
    if pred_df is None:
        omitted.append(("fig09_episode_timeseries", "predictions data not found"))
        return
    log("Generating fig09_episode_timeseries...")

    h1 = pred_df[pred_df["horizon"] == 1].copy()
    station_peaks = h1.groupby("station_id")["y_true"].quantile(0.95).sort_values(ascending=False)
    target_station = station_peaks.index[0]
    st = h1[h1["station_id"] == target_station].copy()
    st = st.sort_values("date")

    peak_date = st.loc[st["y_true"].idxmax(), "date"]
    window_start = peak_date - pd.Timedelta(days=15)
    window_end = peak_date + pd.Timedelta(days=15)
    episode = st[(st["date"] >= window_start) & (st["date"] <= window_end)]

    station_name = master.loc[master["station_id"] == target_station, "station_name"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    obs = episode[episode["model"] == episode["model"].unique()[0]]
    ax.plot(obs["date"], obs["y_true"], "k-", lw=1.5, label="Observed", zorder=5)

    naive = episode[episode["model"] == "seasonal_naive"]
    if len(naive) > 0:
        ax.plot(naive["date"], naive["y_pred"], "--", color="#ff7f0e", lw=1,
                label="Seasonal naive", alpha=0.8)

    hgb = episode[episode["model"] == "hgb_direct"]
    if len(hgb) > 0:
        ax.plot(hgb["date"], hgb["y_pred"], "-", color="#d62728", lw=1,
                label="HGB Direct", alpha=0.8)

    stl = episode[episode["model"] == "stl_ridge_direct"]
    if len(stl) > 0:
        ax.plot(stl["date"], stl["y_pred"], "-", color="#2ca02c", lw=1,
                label="STL Ridge", alpha=0.8)

    ax.axhline(50, color="grey", ls=":", lw=1, alpha=0.6, label="50 µg/m³ threshold")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM$_{10}$ (µg/m³)")
    ax.set_title(f"Episode window — {station_name} (h=1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    savefig(fig, "fig09_episode_timeseries")


# ---------------------------------------------------------------------------
# TABLE 1 — Model-family diagnostic summary
# ---------------------------------------------------------------------------
def table01_model_family_summary(master):
    log("Generating table01_model_family_diagnostic_summary...")
    agg = master.groupby("model_family").agg(
        n_cells=("skill", "count"),
        median_skill=("skill", "median"),
        median_alpha=("alpha", "median"),
        median_skill_vp=("skill_vp", "median"),
        collapse_pct=("collapse_flag", lambda x: x.sum() / len(x) * 100),
        near_retained_pct=("near_ideal_flag", lambda x: x.sum() / len(x) * 100),
        inflation_pct=("inflation_flag", lambda x: x.sum() / len(x) * 100),
        positive_skill_pct=("skill", lambda x: (x > 0).sum() / len(x) * 100),
    ).reindex(FAMILY_ORDER).reset_index()

    if "dm_significant" in master.columns:
        sig = master.groupby("model_family").agg(
            sig_improvement_pct=("dm_significant", lambda x: x.sum() / len(x) * 100),
        ).reindex(FAMILY_ORDER).reset_index()
        agg = agg.merge(sig, on="model_family")

    save_table(agg, "table01_model_family_diagnostic_summary")


# ---------------------------------------------------------------------------
# TABLE 2 — Horizon-wise diagnostic summary
# ---------------------------------------------------------------------------
def table02_horizon_summary(master):
    log("Generating table02_horizon_diagnostic_summary...")
    agg = master.groupby(["horizon", "model_family"]).agg(
        n_cells=("skill", "count"),
        median_skill=("skill", "median"),
        median_alpha=("alpha", "median"),
        collapse_pct=("collapse_flag", lambda x: x.sum() / len(x) * 100),
        near_retained_pct=("near_ideal_flag", lambda x: x.sum() / len(x) * 100),
        n_stations=("station_id", "nunique"),
    ).reset_index()
    family_cat = pd.CategoricalDtype(FAMILY_ORDER, ordered=True)
    agg["model_family"] = agg["model_family"].astype(family_cat)
    agg = agg.sort_values(["horizon", "model_family"])
    save_table(agg, "table02_horizon_diagnostic_summary")


# ---------------------------------------------------------------------------
# TABLE 3 — Station-wise diagnostic summary
# ---------------------------------------------------------------------------
def table03_station_summary(master):
    log("Generating table03_station_diagnostic_summary...")
    agg = master.groupby(["station_name", "station_id"]).agg(
        station_type=("station_type", "first"),
        station_class=("station_class", "first"),
        median_skill=("skill", "median"),
        median_alpha=("alpha", "median"),
        collapse_pct=("collapse_flag", lambda x: x.sum() / len(x) * 100),
        near_retained_pct=("near_ideal_flag", lambda x: x.sum() / len(x) * 100),
        n_cells=("skill", "count"),
    ).reset_index()

    best = master.loc[master.groupby(["station_id"])["skill"].idxmax()][
        ["station_id", "model_family"]
    ].rename(columns={"model_family": "best_family_by_skill"})
    agg = agg.merge(best, on="station_id", how="left")
    agg = agg.sort_values("median_skill", ascending=False)
    save_table(agg, "table03_station_diagnostic_summary")


# ---------------------------------------------------------------------------
# TABLE 4 — Skill–retention quadrant counts
# ---------------------------------------------------------------------------
def table04_quadrants(master):
    log("Generating table04_skill_retention_quadrants...")

    def classify_alpha(a):
        if a < 0.5:
            return "collapsed"
        elif 0.8 <= a <= 1.2:
            return "near-retained"
        elif a > 1.2:
            return "inflated"
        else:
            return "intermediate"

    master = master.copy()
    master["skill_sign"] = np.where(master["skill"] > 0, "positive skill", "non-positive skill")
    master["alpha_class"] = master["alpha"].apply(classify_alpha)

    counts = master.groupby(["skill_sign", "alpha_class"]).size().reset_index(name="count")
    total = counts["count"].sum()
    counts["pct"] = (counts["count"] / total * 100).round(2)
    save_table(counts, "table04_skill_retention_quadrants")


# ---------------------------------------------------------------------------
# TABLE 5 — Exceedance summary
# ---------------------------------------------------------------------------
def table05_exceedance_summary(exc_df):
    if exc_df is None:
        omitted.append(("table05_exceedance_summary", "exceedance data not found"))
        return
    log("Generating table05_exceedance_summary...")
    cols_to_agg = {}
    for c in ["recall", "precision", "f1"]:
        if c in exc_df.columns:
            cols_to_agg[f"median_{c}"] = (c, "median")
    cols_to_agg["n_cells"] = ("recall", "count")

    agg = exc_df.groupby(["threshold_type", "model_family"]).agg(**cols_to_agg).reset_index()
    family_cat = pd.CategoricalDtype(FAMILY_ORDER, ordered=True)
    agg["model_family"] = agg["model_family"].astype(family_cat)
    agg = agg.sort_values(["threshold_type", "model_family"])
    save_table(agg, "table05_exceedance_summary")


# ---------------------------------------------------------------------------
# TABLE 6 — Murphy decomposition summary
# ---------------------------------------------------------------------------
def table06_murphy_summary(murphy_df):
    if murphy_df is None:
        omitted.append(("table06_murphy_decomposition_summary", "Murphy data not found"))
        return
    log("Generating table06_murphy_decomposition_summary...")
    agg = murphy_df.groupby("model_family").agg(
        median_mse=("mse", "median"),
        median_bias_sq=("bias_sq", "median"),
        median_cond_bias_sq=("cond_bias_sq", "median"),
        median_irreducible_sq=("irreducible_sq", "median"),
        n_cells=("mse", "count"),
    ).reindex(FAMILY_ORDER).reset_index()

    total = agg["median_bias_sq"] + agg["median_cond_bias_sq"] + agg["median_irreducible_sq"]
    agg["bias_pct"] = (agg["median_bias_sq"] / total * 100).round(2)
    agg["cond_bias_pct"] = (agg["median_cond_bias_sq"] / total * 100).round(2)
    agg["irreducible_pct"] = (agg["median_irreducible_sq"] / total * 100).round(2)
    save_table(agg, "table06_murphy_decomposition_summary")


# ---------------------------------------------------------------------------
# Meteorology data loaders
# ---------------------------------------------------------------------------
METEO_DIR_MADRID = ROOT / "data" / "external" / "madrid_meteo"
METEO_DIR_IRELAND = ROOT / "data" / "external" / "ireland_meteo"


def load_meteo_data():
    data = {}
    madrid_delta = METEO_DIR_MADRID / "table_delta_lags_meteo_vs_lags_only.csv"
    madrid_master = METEO_DIR_MADRID / "master_meteorology_diagnostic_table.csv"
    madrid_ablation = METEO_DIR_MADRID / "meteorology_ablation_summary.csv"
    madrid_dm = METEO_DIR_MADRID / "dm_lags_meteo_vs_lags_only.csv"
    ireland_delta = METEO_DIR_IRELAND / "table_delta_skill_meteo_vs_lags.csv"
    ireland_dm = METEO_DIR_IRELAND / "dm_lags_meteo_vs_lags_only.csv"

    if not madrid_delta.exists() or not ireland_delta.exists():
        log("WARNING: Meteorology data not found in data/external/ — meteo figures/tables will be skipped")
        return None

    data["madrid_delta"] = pd.read_csv(madrid_delta)
    data["madrid_master"] = pd.read_csv(madrid_master)
    data["madrid_ablation"] = pd.read_csv(madrid_ablation)
    data["madrid_dm"] = pd.read_csv(madrid_dm)
    data["ireland_delta"] = pd.read_csv(ireland_delta)
    data["ireland_dm"] = pd.read_csv(ireland_dm)
    log(f"Loaded meteorology data: Madrid ({len(data['madrid_master'])} rows master, "
        f"{len(data['madrid_delta'])} horizons delta), Ireland ({len(data['ireland_delta'])} rows delta)")
    return data


# ---------------------------------------------------------------------------
# FIGURE 10 — Meteorological value-added: delta skill by horizon (Madrid + Ireland)
# ---------------------------------------------------------------------------
def fig10_meteo_delta_skill(meteo):
    if meteo is None:
        omitted.append(("fig10_meteo_delta_skill_by_horizon", "meteorology data not found"))
        return
    log("Generating fig10_meteo_delta_skill_by_horizon...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Madrid — single aggregated curve
    md = meteo["madrid_delta"]
    ax = axes[0]
    ax.bar(md["horizon"], md["delta_skill_rmse_meteo_minus_lags"] * 100,
           color="#1f77b4", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Δ Skill (meteo − lags-only) [percentage points]")
    ax.set_title("Madrid (15 stations, aggregated)")
    ax.grid(True, axis="y", alpha=0.3)

    # Mark significant DM horizons
    dm_m = meteo["madrid_dm"]
    sig_m = dm_m[dm_m["p_value"] < 0.05]
    for _, row in sig_m.iterrows():
        h = row["horizon"]
        val = md.loc[md["horizon"] == h, "delta_skill_rmse_meteo_minus_lags"].values
        if len(val) > 0:
            ax.annotate("*", xy=(h, val[0] * 100), ha="center", va="bottom",
                        fontsize=14, fontweight="bold", color="red")

    # Ireland — per-station lines
    ir = meteo["ireland_delta"]
    ax = axes[1]
    for station in sorted(ir["station"].unique()):
        sub = ir[ir["station"] == station].sort_values("horizon")
        ax.plot(sub["horizon"], sub["delta_skill_rmse_meteo_minus_lags"] * 100,
                "o-", ms=3, lw=0.8, alpha=0.5, label=station)
    median_ir = ir.groupby("horizon")["delta_skill_rmse_meteo_minus_lags"].median() * 100
    ax.plot(median_ir.index, median_ir.values, "k-", lw=2, label="Median", zorder=10)
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_title("Ireland (8 stations, per-station)")
    ax.grid(True, axis="y", alpha=0.3)

    # Mark significant DM tests
    dm_i = meteo["ireland_dm"]
    sig_i = dm_i[dm_i["p_value"] < 0.05]
    sig_label_added = False
    for _, row in sig_i.iterrows():
        h = row["horizon"]
        st = row["station"]
        val = ir.loc[(ir["horizon"] == h) & (ir["station"] == st),
                     "delta_skill_rmse_meteo_minus_lags"].values
        if len(val) > 0:
            lbl = "DM p < 0.05" if not sig_label_added else ""
            ax.plot(h, val[0] * 100, "r*", ms=10, zorder=15, label=lbl)
            sig_label_added = True

    ax.legend(fontsize=6, ncol=2, loc="upper right")

    fig.suptitle("Meteorological value-added: Δ persistence-relative skill (RMSE)\n"
                 "lags + meteorology vs. lags only — XGBoost direct", fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig10_meteo_delta_skill_by_horizon")


# ---------------------------------------------------------------------------
# FIGURE 11 — Variance retention (φ_h) with vs. without meteorology
# ---------------------------------------------------------------------------
def fig11_meteo_phi_comparison(meteo):
    if meteo is None:
        omitted.append(("fig11_meteo_phi_comparison", "meteorology data not found"))
        return
    log("Generating fig11_meteo_phi_comparison...")

    m = meteo["madrid_master"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: phi_h by horizon, both conditions
    for ax, cond, color, label in [
        (axes[0], "lag_only", "#d62728", "Lags only"),
        (axes[0], "lag_plus_met", "#1f77b4", "Lags + meteorology"),
    ]:
        sub = m[m["condition"] == cond]
        med = sub.groupby("horizon")["phi_h"].median()
        q25 = sub.groupby("horizon")["phi_h"].quantile(0.25)
        q75 = sub.groupby("horizon")["phi_h"].quantile(0.75)
        ax.plot(med.index, med.values, "o-", color=color, ms=4, lw=1.5, label=label)
        ax.fill_between(med.index, q25.values, q75.values, color=color, alpha=0.15)

    axes[0].axhline(0.5, color="red", ls=":", lw=1, alpha=0.5, label="Collapse threshold (φ = 0.5)")
    axes[0].axhline(1.0, color="green", ls=":", lw=1, alpha=0.5, label="Perfect retention (φ = 1)")
    axes[0].set_xlabel("Forecast horizon (hours)")
    axes[0].set_ylabel("Variance retention ratio (φ_h)")
    axes[0].set_title("(a) Median φ_h by horizon")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.5)

    # Panel B: scatter φ_h lags-only vs lags+met
    pivot = m.pivot_table(values="phi_h", index=["station", "horizon"], columns="condition")
    axes[1].scatter(pivot["lag_only"], pivot["lag_plus_met"],
                    s=15, alpha=0.4, c="#1f77b4", edgecolors="none")
    lims = [0, max(pivot.max().max() + 0.1, 1.5)]
    axes[1].plot(lims, lims, "k--", lw=1, alpha=0.5, label="y = x")
    axes[1].axhline(0.5, color="red", ls=":", lw=0.8, alpha=0.4)
    axes[1].axvline(0.5, color="red", ls=":", lw=0.8, alpha=0.4)
    axes[1].set_xlabel("φ_h (lags only)")
    axes[1].set_ylabel("φ_h (lags + meteorology)")
    axes[1].set_title("(b) Pairwise φ_h comparison (station × horizon)")
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    n_below_both = ((pivot["lag_only"] < 0.5) & (pivot["lag_plus_met"] < 0.5)).sum()
    n_total = len(pivot)
    axes[1].text(0.05, 0.95, f"{n_below_both}/{n_total} cells collapsed\nin both conditions",
                 transform=axes[1].transAxes, fontsize=8, va="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Variance retention with vs. without meteorological covariates\n"
                 "Ireland + Madrid pooled (XGBoost direct)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig11_meteo_phi_comparison")


# ---------------------------------------------------------------------------
# FIGURE 12 — Meteorology ablation (Madrid)
# ---------------------------------------------------------------------------
def fig12_meteo_ablation(meteo):
    if meteo is None:
        omitted.append(("fig12_meteo_ablation", "meteorology data not found"))
        return
    log("Generating fig12_meteo_ablation...")

    abl = meteo["madrid_ablation"]
    # Filter to rows with delta (exclude lag_only baseline)
    abl = abl[abl["condition"] != "lag_only"].copy()

    condition_labels = {
        "lag_plus_met_all": "All meteo",
        "lag_plus_rh": "+ Humidity",
        "lag_plus_temp": "+ Temperature",
        "lag_plus_wind": "+ Wind",
    }
    abl["label"] = abl["condition"].map(condition_labels)
    abl = abl.dropna(subset=["label"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: delta skill
    ax = axes[0]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    bars = ax.barh(range(len(abl)), abl["delta_skill_vs_lag_only"] * 100,
                   color=colors[:len(abl)], edgecolor="white")
    ax.set_yticks(range(len(abl)))
    ax.set_yticklabels(abl["label"])
    ax.set_xlabel("Δ Skill vs. lags-only [percentage points]")
    ax.set_title("(a) Skill improvement")
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    # Panel B: delta phi
    ax = axes[1]
    bars = ax.barh(range(len(abl)), abl["delta_phi_vs_lag_only"],
                   color=colors[:len(abl)], edgecolor="white")
    ax.set_yticks(range(len(abl)))
    ax.set_yticklabels(abl["label"])
    ax.set_xlabel("Δ φ (variance retention) vs. lags-only")
    ax.set_title("(b) Variance retention change")
    ax.axvline(0, color="k", lw=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Meteorological covariate ablation — Madrid PM$_{10}$\n"
                 "(XGBoost direct, median across stations × horizons)", fontsize=11)
    fig.tight_layout()
    savefig(fig, "fig12_meteo_ablation")


# ---------------------------------------------------------------------------
# TABLE 8 — Meteorological value-added summary
# ---------------------------------------------------------------------------
def table08_meteo_value_added(meteo):
    if meteo is None:
        omitted.append(("table08_meteo_value_added_summary", "meteorology data not found"))
        return
    log("Generating table08_meteo_value_added_summary...")

    rows = []

    # Madrid aggregated
    md = meteo["madrid_delta"]
    rows.append({
        "network": "Madrid",
        "n_stations": 15,
        "n_horizons": len(md),
        "median_delta_skill_rmse_pct": round(md["delta_skill_rmse_meteo_minus_lags"].median() * 100, 2),
        "mean_delta_skill_rmse_pct": round(md["delta_skill_rmse_meteo_minus_lags"].mean() * 100, 2),
        "max_delta_skill_rmse_pct": round(md["delta_skill_rmse_meteo_minus_lags"].max() * 100, 2),
        "pct_horizons_positive": round((md["delta_skill_rmse_meteo_minus_lags"] > 0).sum() / len(md) * 100, 1),
        "dm_significant_cells": len(meteo["madrid_dm"][meteo["madrid_dm"]["p_value"] < 0.05]),
        "dm_total_cells": len(meteo["madrid_dm"]),
    })

    # Ireland per-station
    ir = meteo["ireland_delta"]
    rows.append({
        "network": "Ireland",
        "n_stations": ir["station"].nunique(),
        "n_horizons": ir["horizon"].nunique(),
        "median_delta_skill_rmse_pct": round(ir["delta_skill_rmse_meteo_minus_lags"].median() * 100, 2),
        "mean_delta_skill_rmse_pct": round(ir["delta_skill_rmse_meteo_minus_lags"].mean() * 100, 2),
        "max_delta_skill_rmse_pct": round(ir["delta_skill_rmse_meteo_minus_lags"].max() * 100, 2),
        "pct_horizons_positive": round((ir["delta_skill_rmse_meteo_minus_lags"] > 0).sum() / len(ir) * 100, 1),
        "dm_significant_cells": len(meteo["ireland_dm"][meteo["ireland_dm"]["p_value"] < 0.05]),
        "dm_total_cells": len(meteo["ireland_dm"]),
    })

    # Phi comparison (from Madrid master)
    m = meteo["madrid_master"]
    phi_lag = m[m["condition"] == "lag_only"]["phi_h"].median()
    phi_met = m[m["condition"] == "lag_plus_met"]["phi_h"].median()
    for r in rows:
        if r["network"] == "Madrid":
            r["median_phi_lags_only"] = round(phi_lag, 4)
            r["median_phi_lags_meteo"] = round(phi_met, 4)
            r["delta_phi"] = round(phi_met - phi_lag, 4)

    df = pd.DataFrame(rows)
    save_table(df, "table08_meteo_value_added_summary")


# ---------------------------------------------------------------------------
# TABLE 9 — DM significance for meteorology comparison
# ---------------------------------------------------------------------------
def table09_meteo_dm_significance(meteo):
    if meteo is None:
        omitted.append(("table09_meteo_dm_significance", "meteorology data not found"))
        return
    log("Generating table09_meteo_dm_significance...")

    # Combine Madrid and Ireland DM results
    dm_m = meteo["madrid_dm"].copy()
    dm_m["network"] = "Madrid"
    dm_m["station"] = "aggregated"

    dm_i = meteo["ireland_dm"].copy()
    dm_i["network"] = "Ireland"

    combined = pd.concat([dm_m, dm_i], ignore_index=True)
    combined["significant"] = combined["p_value"] < 0.05
    cols = ["network", "station", "horizon", "n", "dm_stat", "p_value", "favours", "significant"]
    combined = combined[[c for c in cols if c in combined.columns]]
    save_table(combined, "table09_meteo_dm_significance")


# ---------------------------------------------------------------------------
# TABLE 7 — Figure-source mapping
# ---------------------------------------------------------------------------
def table07_figure_source_mapping():
    log("Generating table07_figure_source_mapping...")
    mapping = []
    figure_specs = [
        ("fig01", "fig01_station_map", "master_diagnostic_table.csv", "lat, lon, station_class, collapse_flag"),
        ("fig02", "fig02_evaluation_workflow", "(conceptual — no data input)", "N/A"),
        ("fig03", "fig03_station_horizon_heatmap", "master_diagnostic_table.csv", "station_name, horizon, alpha, model_family"),
        ("fig04", "fig04_horizon_distribution", "master_diagnostic_table.csv", "horizon, skill, alpha"),
        ("fig05", "fig05_skill_variance_plane", "master_diagnostic_table.csv", "alpha, skill, model_family"),
        ("fig06", "fig06_station_collapse_rates", "master_diagnostic_table.csv", "station_name, station_class, collapse_flag"),
        ("fig07", "fig07_exceedance_diagnostics", "exceedance_all_stations.csv", "threshold_type, model_family, recall, f1"),
        ("fig08", "fig08_murphy_decomposition", "murphy_decomposition_all_stations.csv", "model_family, bias_sq, cond_bias_sq, irreducible_sq"),
        ("fig09", "fig09_episode_timeseries", "predictions_all_stations.csv", "date, y_true, y_pred, model, station_id"),
        ("fig10", "fig10_meteo_delta_skill_by_horizon", "data/external/{madrid,ireland}_meteo/", "horizon, delta_skill_rmse, dm p-value"),
        ("fig11", "fig11_meteo_phi_comparison", "data/external/madrid_meteo/master_meteorology_diagnostic_table.csv", "condition, horizon, phi_h, station"),
        ("fig12", "fig12_meteo_ablation", "data/external/madrid_meteo/meteorology_ablation_summary.csv", "condition, delta_skill, delta_phi"),
    ]

    omitted_stems = {o[0] for o in omitted}
    for fid, stem, inputs, variables in figure_specs:
        was_generated = stem in generated_figures
        reason = ""
        if not was_generated:
            match = [o for o in omitted if o[0] == stem]
            reason = match[0][1] if match else "unknown"
        mapping.append({
            "figure_id": fid,
            "figure_file": f"{stem}.pdf" if was_generated else "",
            "input_files": inputs,
            "variables_used": variables,
            "generated": "yes" if was_generated else "no",
            "omission_reason": reason,
        })

    save_table(pd.DataFrame(mapping), "table07_figure_source_mapping", latex=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 60)
    log("VARRET PM10 — Figure & Table Generation")
    log("=" * 60)

    master = load_master()
    murphy = load_murphy()
    exc = load_exceedance()
    pred = load_predictions()
    meteo = load_meteo_data()

    log("-" * 60)
    log("FIGURES")
    log("-" * 60)
    fig01_station_map(master)
    fig02_evaluation_workflow()
    fig03_station_horizon_heatmap(master)
    fig04_horizon_distribution(master)
    fig05_skill_variance_plane(master)
    fig06_station_collapse_rates(master)
    fig07_exceedance_diagnostics(exc)
    fig08_murphy_decomposition(murphy)
    fig09_episode_timeseries(pred, master)
    fig10_meteo_delta_skill(meteo)
    fig11_meteo_phi_comparison(meteo)
    fig12_meteo_ablation(meteo)

    log("-" * 60)
    log("TABLES")
    log("-" * 60)
    table01_model_family_summary(master)
    table02_horizon_summary(master)
    table03_station_summary(master)
    table04_quadrants(master)
    table05_exceedance_summary(exc)
    table06_murphy_summary(murphy)
    table08_meteo_value_added(meteo)
    table09_meteo_dm_significance(meteo)
    table07_figure_source_mapping()

    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"Figures generated: {len(generated_figures)}")
    for f in generated_figures:
        log(f"  ✓ {f}")
    log(f"Tables generated: {len(generated_tables)}")
    for t in generated_tables:
        log(f"  ✓ {t}")
    if omitted:
        log(f"Omitted outputs: {len(omitted)}")
        for name, reason in omitted:
            log(f"  ✗ {name}: {reason}")
    else:
        log("No outputs omitted — all figures and tables generated.")
    log("No .tex manuscript files were modified.")
    log("=" * 60)


if __name__ == "__main__":
    main()
