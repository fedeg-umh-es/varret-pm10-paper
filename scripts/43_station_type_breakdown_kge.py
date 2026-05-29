#!/usr/bin/env python3
"""Station-type robustness breakdown for KGE complementarity.

AUDIT SOURCE: station metadata reused from outputs/tables/master_diagnostic_table.csv,
with KGE/rank diagnostics from scripts/38 and scripts/39.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MASTER_PATH = ROOT / "outputs" / "tables" / "master_diagnostic_table.csv"
RANK_PATH = ROOT / "results" / "rank_correlation_kge_phi.csv"
OUT_CSV = ROOT / "results" / "station_type_breakdown_kge.csv"
OUT_PDF = ROOT / "figures" / "fig_station_type_divergence_kge.pdf"
OUT_PNG = ROOT / "figures" / "fig_station_type_divergence_kge.png"
TIE_PATH = ROOT / "results" / "tie_sensitivity_rankings.csv"
HORIZON_PATH = ROOT / "results" / "horizon_breakdown_kge.csv"
CLOSURE_PATH = ROOT / "results" / "kge_robustness_closure.md"


def _metadata(master: pd.DataFrame) -> pd.DataFrame:
    cols = ["station_id", "station_type", "station_class"]
    meta = master[[col for col in cols if col in master.columns]].drop_duplicates()
    if "station_id" not in meta.columns:
        raise ValueError("master_diagnostic_table.csv does not contain station_id.")
    if "station_type" not in meta.columns:
        meta["station_type"] = "unknown"
    meta["station_type"] = meta["station_type"].fillna("unknown")
    return meta[["station_id", "station_type"]].drop_duplicates("station_id")


def _build_breakdown(ranks: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    ranks = ranks.copy()
    if "station_id" not in ranks.columns:
        ranks["station_id"] = ranks["station"]
    ranks = ranks.merge(meta, on="station_id", how="left", suffixes=("", "_meta"))
    if "station_type_meta" in ranks.columns:
        ranks["station_type"] = ranks["station_type"].fillna(ranks["station_type_meta"])
    ranks["station_type"] = ranks["station_type"].fillna("unknown")

    rows = []
    for station_type, group in ranks.groupby("station_type", sort=True):
        rows.append(
            {
                "station_type": station_type,
                "n_stations": int(group["station_id"].nunique()),
                "n_station_horizon_cells": int(len(group)),
                "median_spearman_skill_vs_kge": float(group["skill_vs_kge"].median(skipna=True)),
                "median_spearman_phi_vs_r": float(group["phi_vs_r"].median(skipna=True)),
                "rank_reversal_rate": float(group["rank_reversal_flag"].mean()),
                "top_skill_top_kge_mismatch_cells": int(group["top_skill_differs_from_top_kge"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _write_figure(out: pd.DataFrame) -> None:
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plot = out.sort_values("rank_reversal_rate", ascending=False).reset_index(drop=True)
    x = range(len(plot))
    fig, ax1 = plt.subplots(figsize=(8.8, 4.8))
    ax1.bar(x, plot["rank_reversal_rate"], color="#E74C3C", alpha=0.26, label="Rank reversal rate")
    ax1.set_ylabel("Rank reversal rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(plot["station_type"], rotation=30, ha="right")
    ax1.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.55)
    ax2 = ax1.twinx()
    ax2.plot(
        list(x),
        plot["median_spearman_skill_vs_kge"],
        color="#008080",
        marker="o",
        linewidth=2,
        label="Median Spearman Skill vs KGE",
    )
    ax2.plot(
        list(x),
        plot["median_spearman_phi_vs_r"],
        color="#8E44AD",
        marker="s",
        linewidth=2,
        label="Median Spearman phi vs r",
    )
    ax2.axhline(0.8, color="#5D6D7E", linestyle="--", linewidth=1.0, alpha=0.75)
    ax2.set_ylabel("Median Spearman rank correlation")
    ax2.set_ylim(-1.05, 1.05)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower left", frameon=True, fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PDF.relative_to(ROOT)}")
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}")


def _horizon_band(h: int) -> str:
    if h <= 2:
        return "short"
    if h <= 5:
        return "medium"
    return "long"


def _markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_closure(station_type: pd.DataFrame) -> None:
    if not TIE_PATH.exists():
        raise FileNotFoundError(f"Missing tie sensitivity output: {TIE_PATH.relative_to(ROOT)}")
    if not HORIZON_PATH.exists():
        raise FileNotFoundError(f"Missing horizon breakdown output: {HORIZON_PATH.relative_to(ROOT)}")

    ties = pd.read_csv(TIE_PATH)
    horizons = pd.read_csv(HORIZON_PATH)
    tie_stable = (
        ties["median_spearman_skill_vs_kge"].lt(0.8)
        | ties["median_spearman_phi_vs_r"].lt(0.8)
        | ties["rank_reversal_rate"].ge(0.2)
        | ties["top_skill_top_kge_mismatch_cells"].gt(0)
    ).nunique() == 1
    horizon_rates = horizons[["h", "rank_reversal_rate", "median_spearman_skill_vs_kge"]].copy()
    horizon_rates["band"] = horizon_rates["h"].map(lambda h: _horizon_band(int(h)))
    strongest_band = (
        horizon_rates.groupby("band")["median_spearman_skill_vs_kge"]
        .median()
        .sort_values()
        .index[0]
    )
    station_transversal = bool((station_type["rank_reversal_rate"] >= 0.8).all())
    robust = bool(tie_stable and station_transversal and (horizons["rank_reversal_rate"] >= 0.8).all())
    if not tie_stable:
        verdict = "sensitive to ranking conventions"
    elif robust:
        verdict = "robust complementary diagnostic"
    else:
        verdict = "partially robust, interpret with caution"
    recommendation = "keep as companion note" if verdict == "robust complementary diagnostic" else "fold into current manuscript as extended diagnostic section"

    lines = [
        "# KGE robustness closure",
        "",
        "## 1. New Scripts Executed",
        "",
        "- scripts/41_tie_sensitivity_rankings.py",
        "- scripts/42_horizon_breakdown_kge.py",
        "- scripts/43_station_type_breakdown_kge.py",
        "",
        "## 2. Inputs Used",
        "",
        "- outputs/metrics/predictions_all_stations.csv",
        "- outputs/tables/master_diagnostic_table.csv",
        "- results/kge_horizon_table.csv",
        "- results/rank_correlation_kge_phi.csv",
        "",
        "## 3. Tie Sensitivity",
        "",
        _markdown_table(ties),
        "",
        f"Result: ranking-method conclusion is {'stable' if tie_stable else 'not stable'} across average, dense and min tie handling.",
        "",
        "## 4. Horizon Breakdown",
        "",
        _markdown_table(horizons),
        "",
        f"Result: KGE complementarity is descriptively strongest in {strongest_band} horizons by median Skill-vs-KGE rank divergence.",
        "",
        "## 5. Station-Type Breakdown",
        "",
        _markdown_table(station_type),
        "",
        f"Result: divergence is {'transversal across station types' if station_transversal else 'not uniform across station types'}; small subgroups should not be overinterpreted.",
        "",
        "## 6. Final Verdict",
        "",
        verdict,
        "",
        "## 7. Editorial Recommendation",
        "",
        recommendation,
        "",
    ]
    CLOSURE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {CLOSURE_PATH.relative_to(ROOT)}")


def main() -> None:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing master table: {MASTER_PATH.relative_to(ROOT)}")
    if not RANK_PATH.exists():
        raise FileNotFoundError(f"Missing rank table: {RANK_PATH.relative_to(ROOT)}")
    master = pd.read_csv(MASTER_PATH)
    ranks = pd.read_csv(RANK_PATH)
    out = _build_breakdown(ranks, _metadata(master))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    _write_figure(out)
    _write_closure(out)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)} with {len(out)} rows")


if __name__ == "__main__":
    main()
