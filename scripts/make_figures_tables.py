#!/usr/bin/env python3
"""
Generate reproducible figures and tables for the VARRET PM10 manuscript phase.

Run from the repository root:
    python scripts/make_figures_tables.py

The script uses only local outputs already present in the repository. It does
not modify manuscript .tex files and does not create meteorology claims unless
explicit meteorology-vs-lags diagnostic inputs are present in the canonical
PM10 outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
README_PATH = ROOT / "outputs" / "README_figures_tables.md"

MASTER_PATH = TAB_DIR / "master_diagnostic_table.csv"
EXCEEDANCE_PATH = TAB_DIR / "exceedance_all_stations.csv"
MURPHY_PATH = TAB_DIR / "murphy_decomposition_all_stations.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "metrics" / "predictions_all_stations.csv"

MODEL_FAMILY_MAP = {
    "hgb_direct": "Direct ML",
    "ridge_direct": "Direct ML",
    "stl_ridge_direct": "Decomposition + Ridge",
    "persistence": "Persistence baseline",
    "sarima": "Statistical baseline",
    "seasonal_naive": "Variance-preserving naive",
}
FAMILY_ORDER = [
    "Direct ML",
    "Decomposition + Ridge",
    "Persistence baseline",
    "Statistical baseline",
    "Variance-preserving naive",
]
FAMILY_COLORS = {
    "Direct ML": "#b54a3f",
    "Decomposition + Ridge": "#2f7f62",
    "Persistence baseline": "#6b6b6b",
    "Statistical baseline": "#4d6f9f",
    "Variance-preserving naive": "#b9852f",
}


@dataclass
class Artifact:
    artifact_id: str
    stem: str
    kind: str
    input_files: str
    variables_used: str
    generated: bool
    omission_reason: str = ""


ARTIFACTS: list[Artifact] = []
INPUTS_USED: list[str] = []
OMITTED_NOTES: list[str] = []


def log(message: str) -> None:
    print(f"[make_figures_tables] {message}")


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, required: bool = False) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input not found: {path}")
        log(f"Skipping missing optional input: {path.relative_to(ROOT)}")
        return None
    df = pd.read_csv(path)
    INPUTS_USED.append(str(path.relative_to(ROOT)))
    log(f"Read {path.relative_to(ROOT)}: {len(df)} rows, {len(df.columns)} columns")
    return df


def add_model_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "model_family" not in out.columns and "model" in out.columns:
        out["model_family"] = out["model"].map(MODEL_FAMILY_MAP).fillna(out["model"])
    return out


def save_figure(fig: plt.Figure, artifact_id: str, stem: str, input_files: str, variables: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    ARTIFACTS.append(Artifact(artifact_id, stem, "figure", input_files, variables, True))
    log(f"Wrote outputs/figures/{stem}.pdf and .png")


def omit(artifact_id: str, stem: str, kind: str, input_files: str, variables: str, reason: str) -> None:
    ARTIFACTS.append(Artifact(artifact_id, stem, kind, input_files, variables, False, reason))
    OMITTED_NOTES.append(f"{artifact_id} {stem}: {reason}")
    log(f"OMIT {stem}: {reason}")


def save_table(
    df: pd.DataFrame,
    artifact_id: str,
    stem: str,
    input_files: str,
    variables: str,
    latex: bool = True,
) -> None:
    csv_path = TAB_DIR / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    if latex:
        df.to_latex(TAB_DIR / f"{stem}.tex", index=False, escape=True, float_format="%.4f")
    ARTIFACTS.append(Artifact(artifact_id, stem, "table", input_files, variables, True))
    log(f"Wrote outputs/tables/{stem}.csv" + (" and .tex" if latex else ""))


def compact_model_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_family"] = pd.Categorical(out["model_family"], categories=FAMILY_ORDER, ordered=True)
    return out.sort_values("model_family")


def csi_from_precision_recall(precision: pd.Series, recall: pd.Series) -> pd.Series:
    denom = (1.0 / recall.replace(0, np.nan)) + (1.0 / precision.replace(0, np.nan)) - 1.0
    return 1.0 / denom


def diagnostic_variable(master: pd.DataFrame, prefer_heatmap: bool = False) -> tuple[str, str, bool]:
    candidates = ["skill_vp", "alpha", "skill", "collapse_flag"] if prefer_heatmap else ["skill_vp", "skill", "alpha"]
    for col in candidates:
        if col in master.columns and master[col].notna().any():
            centered = col in {"skill_vp", "skill"}
            labels = {
                "skill_vp": "SkillVP",
                "skill": "Persistence-relative skill",
                "alpha": "Variance retention alpha",
                "collapse_flag": "Collapse flag rate",
            }
            return col, labels[col], centered
    raise ValueError("No diagnostic variable available")


def print_plan(master: pd.DataFrame, exceedance: pd.DataFrame | None, murphy: pd.DataFrame | None, predictions_exists: bool) -> None:
    meteo_cols = [c for c in master.columns if "meteo" in c.lower() or "meteor" in c.lower()]
    log("Detected inputs and feasible outputs:")
    log(f"  canonical diagnostics: {MASTER_PATH.relative_to(ROOT)} ({len(master)} rows)")
    log(f"  exceedance diagnostics: {'yes' if exceedance is not None else 'no'}")
    log(f"  Murphy decomposition: {'yes' if murphy is not None else 'no'}")
    log(f"  predictions for episode plot: {'yes' if predictions_exists else 'no'}")
    log(f"  explicit meteorology columns/ablations in canonical outputs: {meteo_cols or 'none'}")
    log("  feasible figures: fig01-fig09 where required columns exist")
    log("  feasible tables: table01-table07 where required inputs exist")
    if not meteo_cols:
        log("  meteorology diagnostics omitted: no canonical meteo-vs-lags inputs detected")


def fig01_station_map(master: pd.DataFrame) -> None:
    needed = {"lat", "lon", "station_id"}
    if not needed.issubset(master.columns) or master[["lat", "lon"]].dropna().empty:
        omit("fig01", "fig01_station_map", "figure", "master_diagnostic_table.csv", "lat, lon", "coordinates unavailable")
        return
    stations = (
        master.groupby("station_id", dropna=False)
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            station_name=("station_name", "first"),
            station_type=("station_type", "first"),
            station_class=("station_class", "first"),
            collapse_rate=("collapse_flag", "mean"),
            obs_mean_ug=("obs_mean_ug", "first"),
        )
        .reset_index()
        .dropna(subset=["lat", "lon"])
    )
    class_colors = {"urban": "#b54a3f", "industrial": "#b9852f", "rural": "#2f7f62"}
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for _, row in stations.iterrows():
        color = class_colors.get(str(row.get("station_class", "")).lower(), "#666666")
        size = 45 + 230 * float(row["collapse_rate"])
        ax.scatter(row["lon"], row["lat"], s=size, color=color, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("PM10 stations sized by variance-collapse rate")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    handles = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in class_colors.items()]
    ax.legend(handles=handles, title="Station class", fontsize=8, loc="best")
    save_figure(fig, "fig01", "fig01_station_map", "master_diagnostic_table.csv", "lat, lon, station_class, collapse_flag")


def fig02_workflow() -> None:
    steps = [
        "PM10 observations",
        "Chronological rolling-origin split",
        "Train-only preprocessing",
        "Model fitting",
        "Horizon-wise forecasts",
        "Persistence baseline",
        "Forecast-verification table",
        "Diagnostic outputs",
    ]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis("off")
    positions = [(0.5, 3.1), (2.7, 3.1), (4.9, 3.1), (7.1, 3.1), (9.3, 3.1), (2.7, 1.1), (5.5, 1.1), (8.3, 1.1)]
    for (x, y), label in zip(positions, steps):
        box = mpatches.FancyBboxPatch((x - 0.85, y - 0.42), 1.7, 0.84, boxstyle="round,pad=0.06", fc="#eef2f3", ec="#4d6f9f", lw=1.2)
        ax.add_patch(box)
        ax.text(x, y, textwrap.fill(label, 18), ha="center", va="center", fontsize=8)
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 7), (1, 5), (5, 6), (6, 7)]
    for i, j in arrows:
        ax.annotate("", xy=positions[j], xytext=positions[i], arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#4d6f9f"})
    ax.text(0.55, 4.1, "No random split", fontsize=8, style="italic", color="#555555")
    ax.text(4.35, 4.1, "Train-only statistics", fontsize=8, style="italic", color="#555555")
    ax.text(5.0, 0.15, "Horizon-wise evaluation, persistence-relative skill, variance-retention diagnostics", fontsize=8, style="italic", color="#555555")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.7)
    ax.set_title("Leakage-free rolling-origin evaluation workflow", fontsize=12)
    save_figure(fig, "fig02", "fig02_evaluation_workflow", "conceptual", "protocol steps")


def fig03_heatmap(master: pd.DataFrame) -> None:
    if not {"station_name", "horizon"}.issubset(master.columns):
        omit("fig03", "fig03_station_horizon_heatmap", "figure", "master_diagnostic_table.csv", "station_name, horizon", "station or horizon column missing")
        return
    var, label, centered = diagnostic_variable(master, prefer_heatmap=True)
    plot_df = master.copy()
    if var == "collapse_flag":
        plot_df[var] = plot_df[var].astype(float)
    pivot = plot_df.groupby(["station_name", "horizon"], dropna=False)[var].median().unstack("horizon")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    fig_h = max(5.5, 0.32 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    values = pivot.to_numpy()
    if centered:
        vmax = np.nanmax(np.abs(values))
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        vmin, vmax, cmap = np.nanmin(values), np.nanmax(values), "viridis"
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Station")
    ax.set_title(f"Station x horizon diagnostic heatmap: {label}")
    fig.colorbar(im, ax=ax, shrink=0.82, label=label)
    save_figure(fig, "fig03", "fig03_station_horizon_heatmap", "master_diagnostic_table.csv", f"station_name, horizon, {var}")


def fig04_horizon_distribution(master: pd.DataFrame) -> None:
    if "horizon" not in master.columns:
        omit("fig04", "fig04_horizon_distribution", "figure", "master_diagnostic_table.csv", "horizon", "horizon column missing")
        return
    var, label, centered = diagnostic_variable(master, prefer_heatmap=False)
    horizons = sorted(master["horizon"].dropna().unique())
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    data = [master.loc[master["horizon"] == h, var].dropna() for h in horizons]
    bp = ax.boxplot(data, tick_labels=[str(h) for h in horizons], patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#d8e2dc")
        patch.set_edgecolor("#4d6f9f")
    rng = np.random.default_rng(123)
    for i, vals in enumerate(data, start=1):
        if len(vals) <= 250:
            ax.scatter(i + rng.normal(0, 0.035, len(vals)), vals, s=10, alpha=0.35, color="#333333", linewidths=0)
    if centered:
        ax.axhline(0, color="#b54a3f", ls="--", lw=1)
    if var == "alpha":
        ax.axhline(1, color="#2f7f62", ls="--", lw=1)
        ax.axhline(0.5, color="#b54a3f", ls=":", lw=1)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(label)
    ax.set_title(f"Horizon-wise distribution of {label}")
    ax.grid(True, axis="y", alpha=0.25)
    save_figure(fig, "fig04", "fig04_horizon_distribution", "master_diagnostic_table.csv", f"horizon, {var}")


def fig05_skill_variance_plane(master: pd.DataFrame) -> None:
    if not {"skill", "alpha", "model_family"}.issubset(master.columns):
        omit("fig05", "fig05_skill_variance_plane", "figure", "master_diagnostic_table.csv", "skill, alpha, model_family", "required columns missing")
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.axvspan(0, 0.5, color="#f2c6a0", alpha=0.35, label="collapsed alpha < 0.5")
    ax.axvspan(0.8, 1.2, color="#b7d7c2", alpha=0.25, label="near-retained alpha")
    ax.axhline(0, color="#b54a3f", ls="--", lw=1)
    ax.axvline(1, color="#2f7f62", ls="--", lw=1)
    for family in FAMILY_ORDER:
        sub = master[master["model_family"] == family]
        if sub.empty:
            continue
        ax.scatter(sub["alpha"], sub["skill"], s=18, alpha=0.55, color=FAMILY_COLORS[family], label=family, edgecolors="none")
    ax.set_xlabel("Variance retention alpha")
    ax.set_ylabel("Persistence-relative skill")
    ax.set_title("Skill vs variance retention")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    save_figure(fig, "fig05", "fig05_skill_variance_plane", "master_diagnostic_table.csv", "alpha, skill, model_family")


def fig06_station_collapse(master: pd.DataFrame) -> None:
    if "collapse_flag" not in master.columns and "alpha" not in master.columns:
        omit("fig06", "fig06_station_collapse_rates", "figure", "master_diagnostic_table.csv", "collapse_flag or alpha", "collapse information unavailable")
        return
    plot_df = master.copy()
    if "collapse_flag" not in plot_df.columns:
        plot_df["collapse_flag"] = plot_df["alpha"] < 0.5
    station_col = "station_name" if "station_name" in plot_df.columns else "station_id"
    station = (
        plot_df.groupby([station_col, "station_type" if "station_type" in plot_df.columns else station_col], dropna=False)
        .agg(collapse_pct=("collapse_flag", lambda x: 100 * x.astype(bool).mean()))
        .reset_index()
        .sort_values("collapse_pct")
    )
    type_col = "station_type" if "station_type" in station.columns else None
    fig, ax = plt.subplots(figsize=(8.2, max(5, 0.32 * len(station) + 1.5)))
    colors = plt.cm.Set2(np.linspace(0, 1, max(3, station[type_col].nunique() if type_col else 3)))
    color_map = {k: colors[i] for i, k in enumerate(sorted(station[type_col].dropna().unique()))} if type_col else {}
    bar_colors = [color_map.get(v, "#4d6f9f") for v in station[type_col]] if type_col else "#4d6f9f"
    ax.barh(station[station_col], station["collapse_pct"], color=bar_colors)
    ax.set_xlabel("Collapsed cells (%)")
    ax.set_ylabel("Station")
    ax.set_title("Station-level variance-collapse rates")
    ax.grid(True, axis="x", alpha=0.25)
    if type_col:
        handles = [mpatches.Patch(color=color_map[k], label=str(k)) for k in color_map]
        ax.legend(handles=handles, title="Station type", fontsize=7, loc="lower right")
    save_figure(fig, "fig06", "fig06_station_collapse_rates", "master_diagnostic_table.csv", "station, station_type, collapse_flag")


def fig07_exceedance(exceedance: pd.DataFrame | None) -> None:
    if exceedance is None:
        omit("fig07", "fig07_exceedance_diagnostics", "figure", "exceedance_all_stations.csv", "recall, precision, CSI/FAR", "exceedance file unavailable")
        return
    exc = add_model_family(exceedance)
    if "far" not in exc.columns and "precision" in exc.columns:
        exc["far"] = 1 - exc["precision"]
    if "csi" not in exc.columns and {"precision", "recall"}.issubset(exc.columns):
        exc["csi"] = csi_from_precision_recall(exc["precision"], exc["recall"])
    metrics = [c for c in ["recall", "far", "csi"] if c in exc.columns]
    if not metrics:
        omit("fig07", "fig07_exceedance_diagnostics", "figure", "exceedance_all_stations.csv", "recall, precision, CSI/FAR", "no exceedance metrics found")
        return
    thresholds = list(exc["threshold_type"].dropna().unique())
    fig, axes = plt.subplots(1, len(thresholds), figsize=(4.2 * len(thresholds), 4.8), sharey=True)
    if len(thresholds) == 1:
        axes = [axes]
    for ax, threshold in zip(axes, thresholds):
        sub = exc[exc["threshold_type"] == threshold]
        agg = compact_model_order(sub.groupby("model_family")[metrics].median().reset_index()).dropna(subset=metrics, how="all")
        x = np.arange(len(agg))
        width = 0.24
        for j, metric in enumerate(metrics):
            ax.bar(x + (j - (len(metrics) - 1) / 2) * width, agg[metric], width=width, label=metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(agg["model_family"], rotation=25, ha="right", fontsize=7)
        ax.set_title(str(threshold))
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Median metric")
    axes[0].legend(fontsize=7)
    fig.suptitle("Exceedance diagnostics by threshold and model family")
    save_figure(fig, "fig07", "fig07_exceedance_diagnostics", "exceedance_all_stations.csv", "threshold_type, model_family, recall, FAR, CSI")


def fig08_murphy(murphy: pd.DataFrame | None) -> None:
    if murphy is None:
        omit("fig08", "fig08_murphy_decomposition", "figure", "murphy_decomposition_all_stations.csv", "MSE components", "Murphy file unavailable")
        return
    murphy = add_model_family(murphy)
    needed = {"bias_sq", "cond_bias_sq", "irreducible_sq", "model_family"}
    if not needed.issubset(murphy.columns):
        omit("fig08", "fig08_murphy_decomposition", "figure", "murphy_decomposition_all_stations.csv", ", ".join(sorted(needed)), "required Murphy columns missing")
        return
    agg = compact_model_order(
        murphy.groupby("model_family")[["bias_sq", "cond_bias_sq", "irreducible_sq"]].median().reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    x = np.arange(len(agg))
    bottom = np.zeros(len(agg))
    comps = [("bias_sq", "Bias", "#b54a3f"), ("cond_bias_sq", "Conditional bias", "#b9852f"), ("irreducible_sq", "Residual", "#4d6f9f")]
    for col, label, color in comps:
        ax.bar(x, agg[col], bottom=bottom, color=color, label=label)
        bottom += agg[col].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels(agg["model_family"], rotation=20, ha="right")
    ax.set_ylabel("Median MSE component")
    ax.set_title("Murphy-style decomposition by model family")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, "fig08", "fig08_murphy_decomposition", "murphy_decomposition_all_stations.csv", "model_family, bias_sq, cond_bias_sq, irreducible_sq")


def fig09_episode(master: pd.DataFrame) -> None:
    if not PREDICTIONS_PATH.exists():
        omit("fig09", "fig09_episode_timeseries", "figure", "predictions_all_stations.csv", "date, y_true, y_pred", "predictions file unavailable")
        return
    pred = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    INPUTS_USED.append(str(PREDICTIONS_PATH.relative_to(ROOT)))
    pred["station_id"] = pred["dataset"].str.replace("e1_rr_", "", regex=False)
    h1 = pred[pred["horizon"] == 1].copy()
    if h1.empty or not {"date", "y_true", "y_pred", "model"}.issubset(h1.columns):
        omit("fig09", "fig09_episode_timeseries", "figure", "predictions_all_stations.csv", "date, y_true, y_pred", "required prediction columns unavailable")
        return
    target_station = h1.groupby("station_id")["y_true"].max().sort_values(ascending=False).index[0]
    station_df = h1[h1["station_id"] == target_station]
    peak_date = station_df.loc[station_df["y_true"].idxmax(), "date"]
    window = station_df[(station_df["date"] >= peak_date - pd.Timedelta(days=15)) & (station_df["date"] <= peak_date + pd.Timedelta(days=15))]
    station_name = master.loc[master["station_id"] == target_station, "station_name"].dropna()
    title_station = station_name.iloc[0] if not station_name.empty else target_station
    fig, ax = plt.subplots(figsize=(10, 4.6))
    obs = window.drop_duplicates("date").sort_values("date")
    ax.plot(obs["date"], obs["y_true"], color="black", lw=1.5, label="Observed")
    for model, color, label in [
        ("seasonal_naive", "#b9852f", "Seasonal naive"),
        ("hgb_direct", "#b54a3f", "HGB direct"),
        ("stl_ridge_direct", "#2f7f62", "STL Ridge"),
    ]:
        sub = window[window["model"] == model].sort_values("date")
        if not sub.empty:
            ax.plot(sub["date"], sub["y_pred"], lw=1.1, color=color, label=label)
    ax.axhline(50, color="#666666", ls=":", lw=1, label="50 ug/m3")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM10")
    ax.set_title(f"Representative high-PM10 episode, h=1: {title_station}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    save_figure(fig, "fig09", "fig09_episode_timeseries", "predictions_all_stations.csv", "date, y_true, y_pred, model, station_id")


def table01(master: pd.DataFrame) -> None:
    agg = (
        master.groupby("model_family")
        .agg(
            n_cells=("skill", "count"),
            median_skill=("skill", "median"),
            median_alpha=("alpha", "median"),
            median_skill_vp=("skill_vp", "median"),
            collapse_pct=("collapse_flag", lambda x: 100 * x.astype(bool).mean()),
            near_retained_pct=("near_ideal_flag", lambda x: 100 * x.astype(bool).mean()),
            inflation_pct=("inflation_flag", lambda x: 100 * x.astype(bool).mean()),
            positive_skill_pct=("skill", lambda x: 100 * (x > 0).mean()),
        )
        .reset_index()
    )
    if "dm_significant" in master.columns:
        sig = master.groupby("model_family")["dm_significant"].apply(lambda x: 100 * x.astype(bool).mean()).rename("dm_significant_pct").reset_index()
        agg = agg.merge(sig, on="model_family", how="left")
    save_table(compact_model_order(agg), "table01", "table01_model_family_diagnostic_summary", "master_diagnostic_table.csv", "model_family, skill, alpha, skill_vp, flags")


def table02(master: pd.DataFrame) -> None:
    agg = (
        master.groupby(["horizon", "model_family"])
        .agg(
            n_cells=("skill", "count"),
            n_stations=("station_id", "nunique"),
            median_skill=("skill", "median"),
            median_alpha=("alpha", "median"),
            collapse_pct=("collapse_flag", lambda x: 100 * x.astype(bool).mean()),
            near_retained_pct=("near_ideal_flag", lambda x: 100 * x.astype(bool).mean()),
        )
        .reset_index()
    )
    agg["model_family"] = pd.Categorical(agg["model_family"], categories=FAMILY_ORDER, ordered=True)
    save_table(agg.sort_values(["horizon", "model_family"]), "table02", "table02_horizon_diagnostic_summary", "master_diagnostic_table.csv", "horizon, model_family, skill, alpha, flags")


def table03(master: pd.DataFrame) -> None:
    station_col = "station_name" if "station_name" in master.columns else "station_id"
    base = (
        master.groupby(["station_id", station_col], dropna=False)
        .agg(
            station_type=("station_type", "first"),
            median_skill=("skill", "median"),
            median_alpha=("alpha", "median"),
            collapse_pct=("collapse_flag", lambda x: 100 * x.astype(bool).mean()),
            near_retained_pct=("near_ideal_flag", lambda x: 100 * x.astype(bool).mean()),
            n_cells=("skill", "count"),
        )
        .reset_index()
    )
    best = (
        master.groupby(["station_id", "model_family"])["skill"]
        .median()
        .reset_index()
        .sort_values(["station_id", "skill"], ascending=[True, False])
        .drop_duplicates("station_id")
        .rename(columns={"model_family": "best_model_family_by_median_skill", "skill": "best_family_median_skill"})
    )
    out = base.merge(best[["station_id", "best_model_family_by_median_skill"]], on="station_id", how="left")
    save_table(out.sort_values("median_skill", ascending=False), "table03", "table03_station_diagnostic_summary", "master_diagnostic_table.csv", "station, station_type, skill, alpha, best model family")


def table04(master: pd.DataFrame) -> None:
    df = master.copy()
    conditions = [
        (df["skill"] > 0) & (df["alpha"] < 0.5),
        (df["skill"] > 0) & (df["alpha"].between(0.8, 1.2)),
        (df["skill"] > 0) & (df["alpha"] > 1.2),
        (df["skill"] <= 0) & (df["alpha"] < 0.5),
        (df["skill"] <= 0) & (df["alpha"].between(0.8, 1.2)),
        (df["skill"] <= 0) & (df["alpha"] > 1.2),
    ]
    labels = [
        "positive skill + collapsed alpha",
        "positive skill + near-retained alpha",
        "positive skill + inflated alpha",
        "non-positive skill + collapsed alpha",
        "non-positive skill + near-retained alpha",
        "non-positive skill + inflated alpha",
    ]
    rows = []
    for label, cond in zip(labels, conditions):
        rows.append({"quadrant": label, "n_cells": int(cond.sum()), "pct_cells": round(100 * cond.mean(), 2)})
    save_table(pd.DataFrame(rows), "table04", "table04_skill_retention_quadrants", "master_diagnostic_table.csv", "skill, alpha")


def table05(exceedance: pd.DataFrame | None) -> None:
    if exceedance is None:
        omit("table05", "table05_exceedance_summary", "table", "exceedance_all_stations.csv", "threshold, recall, precision, FAR, CSI", "exceedance file unavailable")
        return
    exc = add_model_family(exceedance)
    if "far" not in exc.columns and "precision" in exc.columns:
        exc["far"] = 1 - exc["precision"]
    if "csi" not in exc.columns and {"precision", "recall"}.issubset(exc.columns):
        exc["csi"] = csi_from_precision_recall(exc["precision"], exc["recall"])
    agg_cols = {f"median_{c}": (c, "median") for c in ["recall", "precision", "far", "csi"] if c in exc.columns}
    agg_cols["n_cells"] = ("model", "count")
    out = exc.groupby(["threshold_type", "model_family"]).agg(**agg_cols).reset_index()
    out["model_family"] = pd.Categorical(out["model_family"], categories=FAMILY_ORDER, ordered=True)
    save_table(out.sort_values(["threshold_type", "model_family"]), "table05", "table05_exceedance_summary", "exceedance_all_stations.csv", "threshold_type, model_family, recall, precision, FAR, CSI")


def table06(murphy: pd.DataFrame | None) -> None:
    if murphy is None:
        omit("table06", "table06_murphy_decomposition_summary", "table", "murphy_decomposition_all_stations.csv", "MSE components", "Murphy file unavailable")
        return
    murphy = add_model_family(murphy)
    out = (
        murphy.groupby("model_family")
        .agg(
            median_mse=("mse", "median"),
            median_bias_sq=("bias_sq", "median"),
            median_cond_bias_sq=("cond_bias_sq", "median"),
            median_irreducible_sq=("irreducible_sq", "median"),
            n_cells=("mse", "count"),
        )
        .reset_index()
    )
    total = out["median_bias_sq"] + out["median_cond_bias_sq"] + out["median_irreducible_sq"]
    out["bias_pct"] = 100 * out["median_bias_sq"] / total
    out["cond_bias_pct"] = 100 * out["median_cond_bias_sq"] / total
    out["irreducible_pct"] = 100 * out["median_irreducible_sq"] / total
    save_table(compact_model_order(out), "table06", "table06_murphy_decomposition_summary", "murphy_decomposition_all_stations.csv", "model_family, mse, bias_sq, cond_bias_sq, irreducible_sq")


def table07_mapping() -> None:
    rows = [
        {
            "figure_id": a.artifact_id,
            "figure_file": f"{a.stem}.pdf" if a.kind == "figure" and a.generated else "",
            "input_files": a.input_files,
            "variables_used": a.variables_used,
            "generated": "yes" if a.generated else "no",
            "omission_reason": a.omission_reason,
        }
        for a in ARTIFACTS
        if a.kind == "figure"
    ]
    pd.DataFrame(rows).to_csv(TAB_DIR / "table07_figure_source_mapping.csv", index=False)
    ARTIFACTS.append(Artifact("table07", "table07_figure_source_mapping", "table", "script artifact registry", "figure_id, input_files, variables_used, omission_reason", True))
    log("Wrote outputs/tables/table07_figure_source_mapping.csv")


def write_readme(master: pd.DataFrame) -> None:
    generated_figs = [a for a in ARTIFACTS if a.kind == "figure" and a.generated]
    omitted_figs = [a for a in ARTIFACTS if a.kind == "figure" and not a.generated]
    generated_tables = [a for a in ARTIFACTS if a.kind == "table" and a.generated]
    omitted_tables = [a for a in ARTIFACTS if a.kind == "table" and not a.generated]
    meteo_note = "No meteorology-vs-lags diagnostic outputs were detected in the canonical PM10 tables, so no meteorology figures or claims were generated."
    text = [
        "# VARRET PM10 Figures and Tables",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "Reproduce from the repository root:",
        "",
        "```bash",
        "python scripts/make_figures_tables.py",
        "```",
        "",
        "This script does not modify the manuscript .tex files.",
        "",
        "## Inputs Used",
        "",
    ]
    text.extend(f"- `{p}`" for p in dict.fromkeys(INPUTS_USED))
    text.extend(
        [
            "",
            "## Figures Generated",
            "",
        ]
    )
    text.extend(f"- `{a.stem}.{{pdf,png}}`" for a in generated_figs)
    text.extend(["", "## Figures Omitted", ""])
    text.extend([f"- `{a.stem}`: {a.omission_reason}" for a in omitted_figs] or ["- None"])
    text.extend(["", "## Tables Generated", ""])
    text.extend(f"- `{a.stem}.csv" + ("` and `.tex`" if a.stem != "table07_figure_source_mapping" else "`") for a in generated_tables)
    text.extend(["", "## Tables Omitted", ""])
    text.extend([f"- `{a.stem}`: {a.omission_reason}" for a in omitted_tables] or ["- None"])
    text.extend(
        [
            "",
            "## Claims Supported",
            "",
            "- Persistence-relative skill must be read jointly with horizon and station diagnostics.",
            "- Positive skill coexists with low variance retention in many model-station-horizon cells.",
            "- Collapse rates and SkillVP/alpha diagnostics expose heterogeneity hidden by aggregate metrics.",
            "- Exceedance and Murphy diagnostics are complementary checks, not replacement primary metrics.",
            "",
        "## Claims Not Supported",
        "",
        f"- {meteo_note}",
        "- No causal claim about model internals or feature attribution is supported by these outputs.",
        "- No claim beyond the available PM10 stations and evaluated horizons is supported.",
        "",
        "## Unsupported Legacy Outputs",
        "",
        "- Any pre-existing `fig10`-`fig12` or `table08`-`table09` files in `outputs/` are not regenerated by this script and are not part of this phase.",
        "",
        "## Run Notes",
            "",
            f"- Canonical rows: {len(master)}",
            f"- Stations: {master['station_id'].nunique() if 'station_id' in master.columns else 'unknown'}",
            f"- Models: {master['model'].nunique() if 'model' in master.columns else 'unknown'}",
            f"- Horizons: {master['horizon'].nunique() if 'horizon' in master.columns else 'unknown'}",
        ]
    )
    README_PATH.write_text("\n".join(text) + "\n", encoding="utf-8")
    log("Wrote outputs/README_figures_tables.md")


def main() -> int:
    ensure_dirs()
    log("VARRET PM10 reproducible figure/table phase")
    master = add_model_family(read_csv(MASTER_PATH, required=True))
    exceedance = read_csv(EXCEEDANCE_PATH)
    murphy = read_csv(MURPHY_PATH)
    if murphy is not None:
        murphy = add_model_family(murphy)
    if exceedance is not None:
        exceedance = add_model_family(exceedance)
    print_plan(master, exceedance, murphy, PREDICTIONS_PATH.exists())

    fig01_station_map(master)
    fig02_workflow()
    fig03_heatmap(master)
    fig04_horizon_distribution(master)
    fig05_skill_variance_plane(master)
    fig06_station_collapse(master)
    fig07_exceedance(exceedance)
    fig08_murphy(murphy)
    fig09_episode(master)

    table01(master)
    table02(master)
    table03(master)
    table04(master)
    table05(exceedance)
    table06(murphy)
    table07_mapping()
    write_readme(master)

    log("Summary:")
    for artifact in ARTIFACTS:
        status = "generated" if artifact.generated else f"omitted: {artifact.omission_reason}"
        log(f"  {artifact.artifact_id} {artifact.stem}: {status}")
    log("No manuscript .tex files were modified by this script.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[make_figures_tables] ERROR: {exc}", file=sys.stderr)
        raise
