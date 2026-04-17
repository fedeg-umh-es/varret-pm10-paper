"""Plotting helpers for regime-conditioned diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_regime_figures(
    regime_skill: pd.DataFrame,
    seasonal_skill: pd.DataFrame,
    figures_dir: Path,
    primary_cluster_scheme: str = "cluster_k3",
    max_models: int = 4,
) -> list[Path]:
    """Render the required regime and seasonal figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_metric_heatmap(regime_skill, figures_dir / "regime_skill_heatmap.png", "skill_rmse", "Regime Skill_RMSE", primary_cluster_scheme, max_models),
        _plot_metric_heatmap(regime_skill, figures_dir / "regime_vr_heatmap.png", "vr", "Regime VR (%)", primary_cluster_scheme, max_models),
        _plot_metric_heatmap(regime_skill, figures_dir / "regime_skillvp_heatmap.png", "skill_vp", "Regime Skill_VP", primary_cluster_scheme, max_models),
        _plot_seasonal_comparison(seasonal_skill, figures_dir / "seasonal_comparison.png", max_models),
    ]
    return paths


def _plot_metric_heatmap(
    regime_skill: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    primary_cluster_scheme: str,
    max_models: int,
) -> Path:
    view = regime_skill[
        (regime_skill["scheme_type"].isin(["overall", "physical", "clustered"]))
        & (regime_skill["scheme_name"].isin(["overall", "ventilation", "moisture", "season", primary_cluster_scheme]))
    ].copy()
    view["row_label"] = np.where(
        view["scheme_type"] == "overall",
        "overall:all",
        view["scheme_name"] + ":" + view["regime"],
    )
    models = list(view["model"].drop_duplicates())[:max_models]
    if not models:
        raise ValueError("No models available to plot regime heatmaps.")

    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 6), squeeze=False)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#dddddd")

    for axis, model in zip(axes[0], models):
        model_view = view[view["model"] == model]
        pivot = model_view.pivot_table(index="row_label", columns="horizon", values=metric, aggfunc="mean").sort_index()
        image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap)
        axis.set_title(model)
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels(pivot.columns.tolist())
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels(pivot.index.tolist(), fontsize=8)
        axis.set_xlabel("Horizon")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                label = "NA" if pd.isna(value) else f"{value:.2f}"
                axis.text(j, i, label, ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_seasonal_comparison(
    seasonal_skill: pd.DataFrame,
    output_path: Path,
    max_models: int,
) -> Path:
    view = seasonal_skill[
        (seasonal_skill["scheme_name"] == "warm_cold")
        & (~seasonal_skill["insufficient_samples"])
    ].copy()
    models = list(view["model"].drop_duplicates())[:max_models]
    if not models:
        raise ValueError("No models available to plot seasonal comparison.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"warm_season": "#d95f02", "cold_season": "#1b9e77"}
    linestyles = {model: "-" for model in models}

    for model in models:
        for regime in ["warm_season", "cold_season"]:
            subset = view[(view["model"] == model) & (view["regime"] == regime)].sort_values("horizon")
            if subset.empty:
                continue
            axes[0].plot(subset["horizon"], subset["skill_rmse"], marker="o", linestyle=linestyles[model], color=colors[regime], alpha=0.75, label=f"{model} {regime}")
            axes[1].plot(subset["horizon"], subset["vr"], marker="s", linestyle=linestyles[model], color=colors[regime], alpha=0.75, label=f"{model} {regime}")

    axes[0].set_title("Seasonal Skill_RMSE")
    axes[1].set_title("Seasonal VR (%)")
    for axis in axes:
        axis.set_xlabel("Horizon")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
