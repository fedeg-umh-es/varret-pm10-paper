"""Generate the three figures used by the EMS decision-audit manuscript.

All panels are derived from the frozen five-model diagnostic table.  The
script refuses LightGBM rows and verifies the 595-cell contract before
writing vector PDF figures.
"""

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


def figure_skill_alpha(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for model in MODELS:
        sub = df[df.model == model]
        eligible = (sub.skill > 0) & sub.dm_significant
        ax.scatter(sub.skill, sub.alpha, s=18, alpha=0.45, color=COLORS[model], label=LABELS[model])
        ax.scatter(sub.loc[eligible, "skill"], sub.loc[eligible, "alpha"], s=24,
                   facecolors="none", edgecolors=COLORS[model], linewidths=0.7)
    ax.axvline(0, color="0.35", lw=0.8)
    ax.axhline(0.5, color="0.35", ls="--", lw=0.8)
    ax.set(xlabel="Persistence-relative RMSE skill", ylabel=r"Variance retention $\alpha$",
           title="Error skill and dynamic fidelity")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    save(fig, "figure_skill_vs_alpha_scatter")


def figure_horizon_profiles(df: pd.DataFrame) -> None:
    summary = df.groupby(["model", "horizon"], as_index=False).agg(skill=("skill", "median"), alpha=("alpha", "median"))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharex=True)
    for model in MODELS:
        sub = summary[summary.model == model]
        axes[0].plot(sub.horizon, sub.skill, marker="o", ms=3, lw=1.5, color=COLORS[model], label=LABELS[model])
        axes[1].plot(sub.horizon, sub.alpha, marker="o", ms=3, lw=1.5, color=COLORS[model], label=LABELS[model])
    axes[0].axhline(0, color="0.35", lw=0.8)
    axes[1].axhline(0.5, color="0.35", ls="--", lw=0.8)
    axes[0].set_ylabel("Median RMSE skill")
    axes[1].set_ylabel(r"Median variance retention $\alpha$")
    for ax in axes:
        ax.set_xlabel("Horizon (days)")
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Horizon-wise error and fidelity profiles", y=1.02)
    save(fig, "figure_horizon_profiles")


def figure_eligibility(df: pd.DataFrame) -> None:
    a = df.skill.gt(0) & df.dm_significant
    b = a & df.alpha.ge(0.5) & df.recall_p75.ge(0.20)
    values = pd.DataFrame({"Rule A": a.groupby(df.model).sum(), "Rule B": b.groupby(df.model).sum()}).reindex(MODELS)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(MODELS))
    width = 0.36
    ax.bar([i - width / 2 for i in x], values["Rule A"], width, label="Rule A", color="#56B4E9")
    ax.bar([i + width / 2 for i in x], values["Rule B"], width, label="Rule B", color="#E69F00")
    ax.set_xticks(list(x), [LABELS[m] for m in MODELS], rotation=20, ha="right")
    ax.set_ylabel("Eligible station--model--horizon cells")
    ax.set_title("Eligibility under error-only and fidelity-aware rules")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "figure_eligibility_change")


if __name__ == "__main__":
    data = load()
    figure_skill_alpha(data)
    figure_horizon_profiles(data)
    figure_eligibility(data)
    print(f"Generated three figures from {len(data)} canonical rows")
