"""Generate diagnostic figures from P33 pipeline outputs.

Reads variance_retention_summary.csv and skill_summary.csv and produces:
  Figure 1: Skill and VR vs horizon for each model
  Figure 2: Skill_VP vs horizon
  Figure 3: VR vs Skill scatter (ghost-skill diagnostic)

All outputs go to outputs/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "persistence":          "#888888",
    "seasonal_persistence": "#aec7e8",
    "sarima":               "#d62728",
    "lgbm":                 "#1f77b4",
}
MARKERS = {
    "persistence":          "x",
    "seasonal_persistence": "^",
    "sarima":               "s",
    "lgbm":                 "o",
}
LW, MS = 1.8, 6
FS = dict(title=9, label=9, tick=8, legend=8)


def _style(ax, title: str, xlabel: str, ylabel: str, xticks) -> None:
    ax.set_title(title, fontsize=FS["title"], fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=FS["label"])
    ax.set_ylabel(ylabel, fontsize=FS["label"])
    ax.set_xticks(xticks)
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS["legend"])


# ── Figure 1: Skill + VR vs horizon ──────────────────────────────────────────

def plot_figure1(vr_df: pd.DataFrame, figures_dir: Path) -> Path:
    models = [m for m in vr_df["model"].unique() if m != "persistence"]
    horizons = sorted(vr_df["horizon"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), dpi=150)
    fig.suptitle("Skill and Variance Retention vs Forecast Horizon",
                 fontsize=FS["title"], fontweight="bold")

    ax = axes[0]
    for m in models:
        d = vr_df[vr_df["model"] == m].sort_values("horizon")
        ax.plot(d["horizon"], d["skill"],
                marker=MARKERS.get(m, "o"), color=COLORS.get(m, None),
                linewidth=LW, markersize=MS, label=m)
    ax.axhline(0, color="#555", linestyle="--", linewidth=1, label="Skill = 0")
    ax.set_ylim([-0.1, 1.05])
    _style(ax, "(A) Skill vs Horizon", "Horizon (days)", "Skill_RMSE", horizons)

    ax = axes[1]
    for m in models:
        d = vr_df[vr_df["model"] == m].sort_values("horizon")
        ax.plot(d["horizon"], d["vr"],
                marker=MARKERS.get(m, "o"), color=COLORS.get(m, None),
                linewidth=LW, markersize=MS, label=m)
    ax.axhline(100, color="#2ca02c", linestyle="--", linewidth=1, label="VR = 100 %")
    ax.axhline(50, color="#ff7f0e", linestyle=":", linewidth=1, label="collapse (50 %)")
    _style(ax, "(B) Variance Retention vs Horizon", "Horizon (days)", "VR (%)", horizons)

    plt.tight_layout()
    out = figures_dir / "figure1_skill_vr.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ── Figure 2: Skill_VP vs horizon ─────────────────────────────────────────────

def plot_figure2(vr_df: pd.DataFrame, figures_dir: Path) -> Path:
    models = [m for m in vr_df["model"].unique() if m != "persistence"]
    horizons = sorted(vr_df["horizon"].unique())

    fig, ax = plt.subplots(figsize=(4, 3.2), dpi=150)
    fig.suptitle("Skill_VP vs Forecast Horizon",
                 fontsize=FS["title"], fontweight="bold")

    for m in models:
        d = vr_df[vr_df["model"] == m].sort_values("horizon")
        ax.plot(d["horizon"], d["skill_vp"],
                marker=MARKERS.get(m, "o"), color=COLORS.get(m, None),
                linewidth=LW, markersize=MS, label=m)
    ax.axhline(0, color="#555", linestyle="--", linewidth=1)
    _style(ax, "Skill_VP vs Horizon", "Horizon (days)", "Skill_VP", horizons)

    plt.tight_layout()
    out = figures_dir / "figure2_skill_vp.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ── Figure 3: VR vs Skill scatter (ghost-skill diagnostic) ───────────────────

def plot_figure3(vr_df: pd.DataFrame, figures_dir: Path) -> Path:
    models = [m for m in vr_df["model"].unique() if m != "persistence"]

    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=150)
    fig.suptitle("Ghost-Skill Diagnostic: VR vs Skill",
                 fontsize=FS["title"], fontweight="bold")

    for m in models:
        d = vr_df[vr_df["model"] == m]
        sc = ax.scatter(d["skill"], d["vr"],
                        c=[COLORS.get(m, None)] * len(d),
                        marker=MARKERS.get(m, "o"), s=40, label=m,
                        zorder=3)
        for _, row in d.iterrows():
            ax.annotate(f"h{int(row['horizon'])}", (row["skill"], row["vr"]),
                        textcoords="offset points", xytext=(4, 2),
                        fontsize=6, color=COLORS.get(m, "gray"))

    # Quadrant reference lines
    ax.axhline(100, color="#2ca02c", linestyle="--", linewidth=1, label="VR = 100 %")
    ax.axhline(50,  color="#ff7f0e", linestyle=":",  linewidth=1, label="collapse (50 %)")
    ax.axvline(0,   color="#555",    linestyle="--", linewidth=1)

    # Ghost-skill zone annotation
    ax.axhspan(-10, 50, xmin=0, alpha=0.05, color="#ff7f0e", label="ghost-skill zone")

    ax.set_xlabel("Skill_RMSE", fontsize=FS["label"])
    ax.set_ylabel("VR (%)", fontsize=FS["label"])
    ax.tick_params(labelsize=FS["tick"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS["legend"] - 1, loc="lower right")

    plt.tight_layout()
    out = figures_dir / "figure3_vr_vs_skill.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    tables_dir = Path("outputs/tables")
    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    vr_path = tables_dir / "variance_retention_summary.csv"
    if not vr_path.exists():
        raise FileNotFoundError(
            f"Missing: {vr_path}\nRun scripts/07_build_variance_retention_table.py first."
        )

    vr_df = pd.read_csv(vr_path)
    print(f"Loaded {len(vr_df)} rows from {vr_path}")

    for fn in (plot_figure1, plot_figure2, plot_figure3):
        out = fn(vr_df, figures_dir)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
