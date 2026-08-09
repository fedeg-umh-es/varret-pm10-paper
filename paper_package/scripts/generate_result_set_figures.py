"""Generate deterministic candidates for the central error--fidelity figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/tables/master_diagnostic_table.csv"
OUT = ROOT / "paper_package/result_set/figures"
OUT_MAIN = ROOT / "paper_package/figures"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    expected = {"hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"}
    if len(df) != 595 or set(df["model"]) != expected:
        raise ValueError("Unexpected canonical result set")
    if df["model"].astype(str).str.contains("lightgbm", case=False).any():
        raise ValueError("LightGBM rows are forbidden")
    if df.duplicated(["station_id", "model", "horizon"]).any():
        raise ValueError("Duplicate station-model-horizon keys")
    df["formal_discordant"] = (
        (df["skill"] > 0)
        & df["dm_significant"].astype(bool)
        & (df["recall_p75"] >= 0.20)
        & (df["alpha"] < 0.50)
    )
    return df


def style():
    return {
        "hgb_direct": ("HGB direct", "#0072B2", "o"),
        "ridge_direct": ("Ridge direct", "#D55E00", "s"),
        "sarima": ("SARIMA", "#009E73", "^"),
        "seasonal_naive": ("Seasonal naive", "#E69F00", "D"),
        "stl_ridge_direct": ("STL+Ridge", "#7B3F98", "P"),
    }


def candidate_a(df: pd.DataFrame) -> Path:
    limits = (-2.65, 0.55, -0.10, 2.05)
    order = list(style())
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, model in zip(axes, order):
        label, color, marker = style()[model]
        ax.add_patch(Rectangle((0, -0.10), 0.55, 0.60, color="#d9eaf7", alpha=0.40, zorder=0))
        sub = df[df["model"] == model]
        regular = ~sub["formal_discordant"]
        ax.scatter(sub.loc[regular, "skill"], sub.loc[regular, "alpha"], s=18, alpha=0.72,
                   color=color, marker=marker, edgecolor="white", linewidth=0.25, zorder=2)
        formal = sub[sub["formal_discordant"]]
        ax.scatter(formal["skill"], formal["alpha"], s=42, facecolors="none", edgecolors="black",
                   linewidth=0.9, marker="o", zorder=3)
        ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
        ax.axhline(0.50, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_title(label, fontsize=10, pad=5)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.grid(True, color="#e6e6e6", linewidth=0.5)
        ax.tick_params(labelsize=8)
    key = axes[5]
    key.axis("off")
    axes[3].set_xlabel("Persistence-relative RMSE skill", fontsize=9)
    axes[4].set_xlabel("Persistence-relative RMSE skill", fontsize=9)
    axes[0].set_ylabel("Variance retention $\\alpha$", fontsize=9)
    axes[3].set_ylabel("Variance retention $\\alpha$", fontsize=9)
    handles = [
        Patch(facecolor="#d9eaf7", edgecolor="none", alpha=0.40, label="Positive skill and $\\alpha<0.50$"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="none", linestyle="None",
               label="Formal discordance (n=101)"),
    ]
    key.legend(handles=handles, loc="center", ncol=1, frameon=False, fontsize=9,
               handlelength=2.2, labelspacing=1.4, borderaxespad=0.0)
    fig.suptitle("Error--fidelity structure by model family", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / "figure2_candidate_a_faceted.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT / "figure2_skill_alpha.pdf", bbox_inches="tight")
    fig.savefig(OUT_MAIN / "figure2_skill_alpha.pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def candidate_b(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    ax.axvspan(0, 0.55, ymin=0, ymax=(0.50 - (-0.10)) / (2.05 - (-0.10)), color="#d9eaf7", alpha=0.40)
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.axhline(0.50, color="#555555", linewidth=0.8, linestyle="--")
    for model, (label, color, marker) in style().items():
        sub = df[df["model"] == model]
        regular = ~sub["formal_discordant"]
        ax.scatter(sub.loc[regular, "skill"], sub.loc[regular, "alpha"], s=18, alpha=0.62,
                   color=color, marker=marker, edgecolor="white", linewidth=0.25, label=label)
        formal = sub[sub["formal_discordant"]]
        ax.scatter(formal["skill"], formal["alpha"], s=40, facecolors="none", edgecolors="black",
                   linewidth=0.9, marker="o")
    ax.set_xlim(-2.65, 0.55)
    ax.set_ylim(-0.10, 2.05)
    ax.set_xlabel("Persistence-relative RMSE skill")
    ax.set_ylabel("Variance retention $\\alpha$")
    ax.grid(True, color="#e6e6e6", linewidth=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=7)

    summary = []
    for model, (label, color, _) in style().items():
        sub = df[df["model"] == model]
        summary.append((label, color, (sub["alpha"] < 0.50).mean() * 100,
                        sub["formal_discordant"].mean() * 100))
    summary = list(reversed(summary))
    y = range(len(summary))
    ax2 = axes[1]
    ax2.barh([v - 0.18 for v in y], [row[2] for row in summary], height=0.32,
             color=[row[1] for row in summary], alpha=0.42, label="$\\alpha<0.50$")
    ax2.barh([v + 0.18 for v in y], [row[3] for row in summary], height=0.32,
             color=[row[1] for row in summary], alpha=0.95, label="Formal discordance")
    ax2.set_yticks(list(y), [row[0] for row in summary])
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Cells (%)")
    ax2.grid(axis="x", color="#e6e6e6", linewidth=0.5)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Pooled error--fidelity structure and family summaries", fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUT / "figure2_candidate_b_pooled_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    data = load_data()
    print(candidate_a(data))
    print(candidate_b(data))
