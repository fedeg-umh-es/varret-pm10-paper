"""Build reproducible manuscript figures for variance-retention diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUMMARY_PATH = Path("outputs/tables/variance_retention_summary.csv")
FIGURES_DIR = Path("outputs/figures")


def _save_current_figure(stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def build_reporting_gap_figure() -> None:
    """Build Figure 1 from the reporting-audit counts used in the manuscript."""
    audit = pd.DataFrame(
        {
            "dimension": [
                "Operational-usefulness framing",
                "Temporally appropriate validation",
                "Persistence or naive baseline",
                "Horizon-wise reporting",
                "Explicit leakage-safe safeguards",
            ],
            "count": [51, 24, 18, 11, 3],
            "denominator": [486, 486, 486, 486, 486],
        }
    )
    audit["percent"] = 100.0 * audit["count"] / audit["denominator"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(audit))
    ax.barh(y, audit["percent"])
    ax.set_yticks(y)
    ax.set_yticklabels(audit["dimension"])
    ax.set_xlabel("Explicitly reported in abstracts (%)")
    ax.set_xlim(0, max(12, float(audit["percent"].max()) + 2.0))
    ax.invert_yaxis()

    for idx, row in audit.iterrows():
        ax.text(
            float(row["percent"]) + 0.25,
            idx,
            f"{int(row['count'])}/{int(row['denominator'])} ({row['percent']:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_current_figure("figure1_reporting_gap_audit")


def build_skill_variance_retention_figure() -> None:
    """Build Figure 2 from variance_retention_summary.csv."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing variance retention table: {SUMMARY_PATH}")

    summary = pd.read_csv(SUMMARY_PATH)
    required = {"model", "skill", "alpha", "skill_vp"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Missing required columns in {SUMMARY_PATH}: {missing}")

    grouped = (
        summary.groupby("model", sort=True)[["skill", "alpha", "skill_vp"]]
        .mean()
        .reset_index()
    )

    metric_labels = ["Skill", "Alpha", "Skill_VP"]
    metric_cols = ["skill", "alpha", "skill_vp"]
    models = grouped["model"].tolist()

    x = np.arange(len(models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    offsets = [-width, 0.0, width]
    for offset, col, label in zip(offsets, metric_cols, metric_labels):
        values = grouped[col].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width, label=label)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.006,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Mean value across horizons")
    ax.set_ylim(0, max(0.35, float(grouped[metric_cols].max().max()) + 0.06))
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_current_figure("figure2_skill_variance_retention")


def main() -> None:
    build_reporting_gap_figure()
    build_skill_variance_retention_figure()


if __name__ == "__main__":
    main()
