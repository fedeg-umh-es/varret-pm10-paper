"""Build the two figures cited by ``paper_a.tex``.

The horizon-wise values are the canonical values reported in Table 2 of the
manuscript. Event results are plotted as reported across-horizon ranges because
the underlying hourly event-level predictions are not distributed in this
repository. This avoids reconstructing unsupported horizon-level values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("lgbm", "sarima")
LABELS = {"lgbm": "LightGBM", "sarima": "SARIMA"}
COLORS = {"lgbm": "#2166ac", "sarima": "#b2182b"}
MARKERS = {"lgbm": "o", "sarima": "s"}


def _series(frame: pd.DataFrame, model: str, column: str) -> tuple[np.ndarray, np.ndarray]:
    subset = frame.loc[frame["model"].eq(model)].sort_values("h")
    return subset["h"].to_numpy(), subset[column].to_numpy()


def _style_axis(axis: plt.Axes, horizons: list[int]) -> None:
    axis.set_xticks(horizons)
    axis.grid(True, alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)


def build_figure1(metrics: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    horizons = sorted(metrics["h"].unique().tolist())
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)

    for model in MODELS:
        h, values = _series(metrics, model, "skill")
        axes[0].plot(h, values, color=COLORS[model], marker=MARKERS[model],
                     linewidth=1.8, markersize=5, label=LABELS[model])
    axes[0].axhline(0, color="0.45", linestyle="--", linewidth=1)
    axes[0].set(title="(A) Persistence-relative RMSE skill",
                xlabel="Forecast horizon (h)", ylabel="Skill")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8)
    _style_axis(axes[0], horizons)

    for model in MODELS:
        h, values = _series(metrics, model, "var_pct")
        axes[1].plot(h, values, color=COLORS[model], marker=MARKERS[model],
                     linewidth=1.8, markersize=5, label=LABELS[model])
    axes[1].axhline(100, color="0.45", linestyle="--", linewidth=1,
                    label="Full variance")
    axes[1].set(title="(B) Variance retention",
                xlabel="Forecast horizon (h)", ylabel="Variance retention (%)")
    axes[1].set_ylim(0, 108)
    axes[1].legend(frameon=False, fontsize=8)
    _style_axis(axes[1], horizons)

    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"figure1_skill_variance.{suffix}", dpi=dpi,
                       bbox_inches="tight")
    plt.close(figure)


def build_figure2(
    metrics: pd.DataFrame,
    event_ranges: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    horizons = sorted(metrics["h"].unique().tolist())
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)

    for model in MODELS:
        h, values = _series(metrics, model, "skill_vp")
        axes[0].plot(h, values, color=COLORS[model], marker=MARKERS[model],
                     linewidth=1.8, markersize=5, label=LABELS[model])
    axes[0].axhline(0, color="0.45", linestyle="--", linewidth=1)
    axes[0].set(title=r"(C) Diagnostic Skill$_{VP}$",
                xlabel="Forecast horizon (h)", ylabel=r"Skill$_{VP}$")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(frameon=False, fontsize=8)
    _style_axis(axes[0], horizons)

    positions = {("lgbm", "Recall"): -0.12, ("sarima", "Recall"): 0.12,
                 ("lgbm", "Precision"): 0.88, ("sarima", "Precision"): 1.12}
    for row in event_ranges.itertuples(index=False):
        center = (row.min_value + row.max_value) / 2
        error = (row.max_value - row.min_value) / 2
        label = LABELS[row.model]
        axes[1].errorbar(
            positions[(row.model, row.metric)], center, yerr=error,
            color=COLORS[row.model], marker=MARKERS[row.model], markersize=6,
            linewidth=2, capsize=5, label=label if row.metric == "Recall" else None,
        )
        if row.approximate:
            axes[1].annotate("approx.", (positions[(row.model, row.metric)], center),
                             xytext=(5, 3), textcoords="offset points", fontsize=7)
    axes[1].set(title="(D) P75 event-metric ranges",
                ylabel="Metric value", xticks=[0, 1],
                xticklabels=["Recall", "Precision"])
    axes[1].set_xlim(-0.45, 1.45)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, axis="y", alpha=0.22, linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].tick_params(labelsize=8)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")

    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"figure2_skillvp_events.{suffix}", dpi=dpi,
                       bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path,
                        default=Path("data/manuscript/paper_a_horizon_metrics.csv"))
    parser.add_argument("--events", type=Path,
                        default=Path("data/manuscript/paper_a_event_metric_ranges.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/figures"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    event_ranges = pd.read_csv(args.events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_figure1(metrics, args.output_dir, args.dpi)
    build_figure2(metrics, event_ranges, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
