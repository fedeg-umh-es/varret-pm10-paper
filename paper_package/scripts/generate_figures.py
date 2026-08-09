"""Generate the two primary diagnostic figures from the frozen table."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs" / "tables" / "master_diagnostic_table.csv"
OUT = ROOT / "paper_package" / "figures"
MODELS = ["hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"]
LABELS = {
    "hgb_direct": "HGB direct",
    "ridge_direct": "Ridge direct",
    "sarima": "SARIMA",
    "seasonal_naive": "Seasonal naive",
    "stl_ridge_direct": "STL+Ridge",
}
COLORS = {m: c for m, c in zip(MODELS, ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#7B3294"])}


def load() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    assert len(df) == 595
    assert set(df["model"]) == set(MODELS)
    assert "lightgbm_direct" not in set(df["model"])
    assert not df.duplicated(["station_id", "model", "horizon"]).any()
    return df


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_horizon_profiles(df: pd.DataFrame) -> None:
    """Figure 1: median skill and alpha by horizon."""
    summary = df.groupby(["model", "horizon"], as_index=False).agg(skill=("skill", "median"), alpha=("alpha", "median"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharex=True)
    for model in MODELS:
        sub = summary[summary.model == model]
        axes[0].plot(sub.horizon, sub.skill, marker="o", ms=3, lw=1.5, color=COLORS[model], label=LABELS[model])
        axes[1].plot(sub.horizon, sub.alpha, marker="o", ms=3, lw=1.5, color=COLORS[model], label=LABELS[model])
    axes[0].axhline(0, color="0.35", lw=0.8)
    axes[1].axhline(0.5, color="0.35", ls="--", lw=0.8, label=r"$\alpha=0.50$")
    axes[0].set_ylabel("Median persistence-relative skill")
    axes[1].set_ylabel(r"Median variance retention $\alpha$")
    for ax in axes:
        ax.set_xlabel("Horizon (days)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Horizon-wise error and fidelity profiles", y=1.02)
    save(fig, "figure1_horizon_profiles")


def figure_skill_alpha(df: pd.DataFrame) -> None:
    """Figure 2: joint skill/fidelity plot with formal discordance outline."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    positive_low = (df.skill > 0) & (df.alpha < 0.5)
    formal = (df.skill > 0) & df.dm_significant & (df.recall_p75 >= 0.20) & (df.alpha < 0.5)
    ax.axvspan(0, max(0.5, float(df.skill.max()) + 0.03), color="#FDE0DD", alpha=0.35,
               label=r"Positive skill / low-$\alpha$ region", zorder=0)
    for model in MODELS:
        sub = df[df.model == model]
        ax.scatter(sub.skill, sub.alpha, s=18, alpha=0.45, color=COLORS[model], label=LABELS[model])
    ax.scatter(df.loc[formal, "skill"], df.loc[formal, "alpha"], s=42,
               facecolors="none", edgecolors="black", linewidths=0.8,
               label="Formal discordance (n=101)", zorder=4)
    ax.axvline(0, color="0.35", lw=0.8)
    ax.axhline(0.5, color="0.35", ls="--", lw=0.8)
    ax.set(xlabel="Persistence-relative RMSE skill", ylabel=r"Variance retention $\alpha$",
           title="Error skill and dynamic fidelity")
    handles, labels = ax.get_legend_handles_labels()
    # Keep the legend outside the data cloud and remove duplicate region labels.
    ax.legend(handles, labels, frameon=True, fontsize=7, ncol=2,
              loc="upper left", bbox_to_anchor=(0, 1.02), borderaxespad=0.2)
    ax.grid(alpha=0.2)
    save(fig, "figure2_skill_alpha")


if __name__ == "__main__":
    data = load()
    figure_horizon_profiles(data)
    figure_skill_alpha(data)
    print(f"Generated two primary figures from {len(data)} canonical rows")
