#!/usr/bin/env python3
"""Horizon-wise robustness breakdown for KGE complementarity.

AUDIT SOURCE: closes robustness around results/kge_horizon_table.csv and
results/rank_correlation_kge_phi.csv from scripts/38 and scripts/39.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
KGE_PATH = ROOT / "results" / "kge_horizon_table.csv"
RANK_PATH = ROOT / "results" / "rank_correlation_kge_phi.csv"
OUT_CSV = ROOT / "results" / "horizon_breakdown_kge.csv"
OUT_PDF = ROOT / "figures" / "fig_horizon_divergence_kge.pdf"
OUT_PNG = ROOT / "figures" / "fig_horizon_divergence_kge.png"


def _build_breakdown(kge: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, group in kge.groupby("h", sort=True):
        rgroup = ranks[ranks["h"].eq(h)]
        rows.append(
            {
                "h": int(h),
                "median_skill_h": float(group["Skill_h"].median(skipna=True)),
                "median_phi_h": float(group["phi_h"].median(skipna=True)),
                "median_r_h": float(group["r_h"].median(skipna=True)),
                "median_beta_h": float(group["beta_h"].median(skipna=True)),
                "median_KGE_h": float(group["KGE_h"].median(skipna=True)),
                "median_spearman_skill_vs_kge": float(rgroup["skill_vs_kge"].median(skipna=True)),
                "median_spearman_phi_vs_r": float(rgroup["phi_vs_r"].median(skipna=True)),
                "rank_reversal_cells": int(rgroup["rank_reversal_flag"].sum()),
                "rank_reversal_rate": float(rgroup["rank_reversal_flag"].mean()),
                "top_skill_top_kge_mismatch_cells": int(rgroup["top_skill_differs_from_top_kge"].sum()),
                "n_station_horizon_cells": int(len(rgroup)),
                "n_station_model_horizon_cells": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _horizon_band(row: pd.Series) -> str:
    h = int(row["h"])
    if h <= 2:
        return "short"
    if h <= 5:
        return "medium"
    return "long"


def _write_figure(out: pd.DataFrame) -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7.4, 4.4))
    ax1.plot(
        out["h"],
        out["median_spearman_skill_vs_kge"],
        color="#008080",
        marker="o",
        linewidth=2,
        label="Median Spearman Skill vs KGE",
    )
    ax1.plot(
        out["h"],
        out["median_spearman_phi_vs_r"],
        color="#8E44AD",
        marker="s",
        linewidth=2,
        label="Median Spearman phi vs r",
    )
    ax1.axhline(0.8, color="#5D6D7E", linestyle="--", linewidth=1.0, alpha=0.8)
    ax1.set_xlabel("Horizon")
    ax1.set_ylabel("Median Spearman rank correlation")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    ax2 = ax1.twinx()
    ax2.bar(
        out["h"],
        out["rank_reversal_rate"],
        color="#E74C3C",
        alpha=0.16,
        label="Rank reversal rate",
    )
    ax2.set_ylabel("Rank reversal rate")
    ax2.set_ylim(0, 1.05)
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc="lower left", frameon=True, fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PDF.relative_to(ROOT)}")
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}")


def main() -> None:
    if not KGE_PATH.exists():
        raise FileNotFoundError(f"Missing KGE table: {KGE_PATH.relative_to(ROOT)}")
    if not RANK_PATH.exists():
        raise FileNotFoundError(f"Missing rank table: {RANK_PATH.relative_to(ROOT)}")
    kge = pd.read_csv(KGE_PATH)
    ranks = pd.read_csv(RANK_PATH)
    out = _build_breakdown(kge, ranks)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    _write_figure(out)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)} with {len(out)} rows")


if __name__ == "__main__":
    main()
