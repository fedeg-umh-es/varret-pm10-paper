#!/usr/bin/env python3
"""Plot predicted-vs-observed PM10 scatter under rolling-origin evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_PATHS = {
    "LightGBM": REPO_ROOT / "outputs" / "predictions" / "lgbm_rolling_origin.parquet",
    "SARIMA": REPO_ROOT / "outputs" / "predictions" / "sarima_rolling_origin.parquet",
}
OBS_PATH = REPO_ROOT / "data" / "processed" / "pm10_preprocessed.parquet"
NORM_PATH = REPO_ROOT / "data" / "processed" / "normalization_params.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "figures"
OUTPUT_PNG = OUTPUT_DIR / "figure_scatter_predicted_vs_observed_ro.png"
OUTPUT_PDF = OUTPUT_DIR / "figure_scatter_predicted_vs_observed_ro.pdf"
HORIZONS = [1, 6, 24, 48]
MODEL_ORDER = ["LightGBM", "SARIMA"]


def invert_zscore(values: pd.Series, mean: float, std: float) -> pd.Series:
    return values.astype(float) * std + mean


def load_panel(model_label: str, pred_path: Path, observations: pd.Series, mean: float, std: float) -> pd.DataFrame:
    pred = pd.read_parquet(pred_path).copy()
    pred["y_true_normalized"] = observations.iloc[pred["sample_idx"].to_numpy()].to_numpy()
    pred["y_true_pm10"] = invert_zscore(pred["y_true_normalized"], mean, std)
    pred["y_pred_pm10"] = invert_zscore(pred["y_pred"], mean, std)
    pred["model"] = model_label
    return pred.loc[:, ["model", "fold", "sample_idx", "horizon", "y_true_pm10", "y_pred_pm10"]]


def panel_stats(df: pd.DataFrame) -> dict[str, float]:
    var_obs = float(df["y_true_pm10"].var(ddof=0))
    var_pred = float(df["y_pred_pm10"].var(ddof=0))
    vr_pct = (100.0 * var_pred / var_obs) if var_obs > 0 else 0.0
    return {
        "n_points": int(len(df)),
        "var_obs": var_obs,
        "var_pred": var_pred,
        "vr_pct": vr_pct,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    observations = pd.read_parquet(OBS_PATH)["pm10_normalized"]
    params = json.loads(NORM_PATH.read_text(encoding="utf-8"))
    mean = float(params["mean"])
    std = float(params["std"])

    frames = [
        load_panel(model_label, pred_path, observations, mean, std)
        for model_label, pred_path in PREDICTION_PATHS.items()
    ]
    full = pd.concat(frames, ignore_index=True)
    plot_data = full[full["horizon"].isin(HORIZONS)].copy()

    combined_min = min(plot_data["y_true_pm10"].min(), plot_data["y_pred_pm10"].min())
    combined_max = max(plot_data["y_true_pm10"].max(), plot_data["y_pred_pm10"].max())
    padding = 0.03 * (combined_max - combined_min)
    lower = combined_min - padding
    upper = combined_max + padding

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), dpi=200, sharex=True, sharey=True)
    fig.suptitle("Predicted vs observed PM10 under rolling-origin evaluation", fontsize=12)

    for row_idx, model_label in enumerate(MODEL_ORDER):
        for col_idx, horizon in enumerate(HORIZONS):
            ax = axes[row_idx, col_idx]
            panel = plot_data[(plot_data["model"] == model_label) & (plot_data["horizon"] == horizon)].copy()

            ax.scatter(
                panel["y_true_pm10"],
                panel["y_pred_pm10"],
                s=8,
                alpha=0.18,
                color="#1f77b4" if model_label == "LightGBM" else "#d62728",
                edgecolors="none",
            )
            ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.0, color="#444444")
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
            ax.set_title(f"{model_label}, h = {horizon}", fontsize=10)
            ax.grid(True, alpha=0.25)

            stats = panel_stats(panel)
            print(
                f"model={model_label} | horizon={horizon} | n_points={stats['n_points']} | "
                f"observed_variance={stats['var_obs']:.6f} | predicted_variance={stats['var_pred']:.6f} | "
                f"variance_retention_pct={stats['vr_pct']:.3f}"
            )

    for ax in axes[1, :]:
        ax.set_xlabel("Observed PM10")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted PM10")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_PNG, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
