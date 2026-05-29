#!/usr/bin/env python3
"""Tie-method sensitivity for KGE rank comparisons.

AUDIT SOURCE: closes robustness around results/kge_horizon_table.csv and
results/rank_correlation_kge_phi.csv produced by scripts/38 and scripts/39.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
KGE_PATH = ROOT / "results" / "kge_horizon_table.csv"
OUT_CSV = ROOT / "results" / "tie_sensitivity_rankings.csv"
OUT_MD = ROOT / "results" / "tie_sensitivity_summary.md"
RANK_METHODS = ["average", "dense", "min"]
METRICS = {
    "skill": "Skill_h",
    "phi": "phi_h",
    "r": "r_h",
    "kge": "KGE_h",
}


def _spearman(group: pd.DataFrame, left: str, right: str, method: str) -> float:
    valid = group[["model", left, right]].dropna()
    if valid["model"].nunique() < 3:
        return float("nan")
    left_rank = valid[left].rank(ascending=False, method=method)
    right_rank = valid[right].rank(ascending=False, method=method)
    return float(left_rank.corr(right_rank, method="spearman"))


def _top_model(group: pd.DataFrame, metric: str) -> str | float:
    valid = group[["model", metric]].dropna()
    if valid.empty:
        return np.nan
    return str(valid.sort_values([metric, "model"], ascending=[False, True]).iloc[0]["model"])


def _has_tie(group: pd.DataFrame, metric: str) -> bool:
    values = group[metric].dropna()
    return bool(values.duplicated(keep=False).any())


def _cell_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in RANK_METHODS:
        for (station, h), group in df.groupby(["station", "h"], sort=True):
            skill_vs_kge = _spearman(group, METRICS["skill"], METRICS["kge"], method)
            phi_vs_r = _spearman(group, METRICS["phi"], METRICS["r"], method)
            top_skill = _top_model(group, METRICS["skill"])
            top_kge = _top_model(group, METRICS["kge"])
            tie_metrics = [name for name, col in METRICS.items() if _has_tie(group, col)]
            rank_reversal = (
                (np.isfinite(skill_vs_kge) and skill_vs_kge < 0.8)
                or (np.isfinite(phi_vs_r) and phi_vs_r < 0.8)
                or top_skill != top_kge
            )
            rows.append(
                {
                    "rank_method": method,
                    "station": station,
                    "h": int(h),
                    "spearman_skill_vs_kge": skill_vs_kge,
                    "spearman_phi_vs_r": phi_vs_r,
                    "top_skill": top_skill,
                    "top_kge": top_kge,
                    "top_skill_differs_from_top_kge": bool(top_skill != top_kge),
                    "rank_reversal_flag": bool(rank_reversal),
                    "n_tied_metrics": len(tie_metrics),
                    "tied_metrics": ",".join(tie_metrics),
                    "has_any_tie": bool(tie_metrics),
                }
            )
    return pd.DataFrame(rows)


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for method, group in rows.groupby("rank_method", sort=True):
        out.append(
            {
                "rank_method": method,
                "n_station_horizon_cells": int(len(group)),
                "median_spearman_skill_vs_kge": float(group["spearman_skill_vs_kge"].median(skipna=True)),
                "median_spearman_phi_vs_r": float(group["spearman_phi_vs_r"].median(skipna=True)),
                "rank_reversal_rate": float(group["rank_reversal_flag"].mean()),
                "top_skill_top_kge_mismatch_cells": int(group["top_skill_differs_from_top_kge"].sum()),
                "tie_affected_cells": int(group["has_any_tie"].sum()),
                "tie_affected_pct": float(100.0 * group["has_any_tie"].mean()),
                "tied_metric_instances": int(group["n_tied_metrics"].sum()),
            }
        )
    return pd.DataFrame(out)


def _conclusion(summary: pd.DataFrame) -> str:
    qualitative = []
    for _, row in summary.iterrows():
        complementary = (
            row["median_spearman_skill_vs_kge"] < 0.8
            or row["median_spearman_phi_vs_r"] < 0.8
            or row["rank_reversal_rate"] >= 0.2
            or row["top_skill_top_kge_mismatch_cells"] > 0
        )
        qualitative.append(bool(complementary))
    return "stable" if len(set(qualitative)) == 1 else "sensitive"


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


def _write_markdown(summary: pd.DataFrame) -> None:
    status = _conclusion(summary)
    lines = [
        "# Tie sensitivity summary",
        "",
        "Ranking methods evaluated: average, dense, min.",
        "",
        _markdown_table(summary),
        "",
        f"Conclusion: main KGE complementarity pattern is {status} across ranking tie methods.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not KGE_PATH.exists():
        raise FileNotFoundError(f"Missing KGE table: {KGE_PATH.relative_to(ROOT)}")
    df = pd.read_csv(KGE_PATH)
    rows = _cell_rows(df)
    summary = _summary(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)
    _write_markdown(summary)
    print(f"Wrote {OUT_CSV.relative_to(ROOT)} with {len(summary)} rows")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
