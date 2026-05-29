#!/usr/bin/env python3
"""Compare horizon-wise rankings from Skill, phi, r and KGE.

AUDIT SOURCE: adapted ranking intent from manuscript_assets/paper_c_kge/tables
and the five-model station-horizon table produced by scripts/38_compute_kge_horizon.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "results" / "kge_horizon_table.csv"
CORR_PATH = ROOT / "results" / "rank_correlation_kge_phi.csv"
SUMMARY_PATH = ROOT / "results" / "rank_reversal_summary.csv"
INTERPRETATION_PATH = ROOT / "results" / "kge_interpretation.md"
METRICS = {
    "skill": "Skill_h",
    "phi": "phi_h",
    "r": "r_h",
    "kge": "KGE_h",
    "kge_skill": "KGE_skill_h",
}


def _rank_corr(group: pd.DataFrame, left: str, right: str) -> float:
    cols = [METRICS[left], METRICS[right]]
    valid = group[["model", *cols]].dropna()
    if valid["model"].nunique() < 3:
        return float("nan")
    left_rank = valid[cols[0]].rank(ascending=False, method="average")
    right_rank = valid[cols[1]].rank(ascending=False, method="average")
    return float(left_rank.corr(right_rank, method="spearman"))


def _top_model(group: pd.DataFrame, metric: str) -> str | float:
    col = METRICS[metric]
    valid = group[["model", col]].dropna()
    if valid.empty:
        return np.nan
    return str(valid.sort_values([col, "model"], ascending=[False, True]).iloc[0]["model"])


def _build_rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("skill", "kge"),
        ("skill", "kge_skill"),
        ("skill", "phi"),
        ("phi", "r"),
        ("phi", "kge"),
        ("r", "kge"),
    ]
    rows = []
    for (station, h), group in df.groupby(["station", "h"], sort=True):
        row: dict[str, object] = {"station": station, "h": int(h)}
        for optional in ["station_id", "station_name", "station_type", "station_class"]:
            if optional in group.columns:
                row[optional] = group[optional].dropna().iloc[0] if group[optional].notna().any() else np.nan
        for left, right in pairs:
            row[f"{left}_vs_{right}"] = _rank_corr(group, left, right)
        for metric in METRICS:
            row[f"top_{metric}"] = _top_model(group, metric)
        corr_values = [row[f"{left}_vs_{right}"] for left, right in pairs]
        low_corr = any(np.isfinite(value) and value < 0.8 for value in corr_values)
        top_diff = row["top_skill"] != row["top_kge"]
        row["top_skill_differs_from_top_kge"] = bool(top_diff)
        row["rank_reversal_flag"] = bool(low_corr or top_diff)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["station", "h"]).reset_index(drop=True)


def _summary(rank_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for h, group in rank_df.groupby("h", sort=True):
        rows.append(
            {
                "aggregation": "horizon",
                "key": int(h),
                "n_cells": int(len(group)),
                "rank_reversal_rate": float(group["rank_reversal_flag"].mean()),
                "top_skill_top_kge_mismatch_rate": float(group["top_skill_differs_from_top_kge"].mean()),
            }
        )
    for model, group in rank_df.groupby("top_kge", dropna=False, sort=True):
        rows.append(
            {
                "aggregation": "top_kge_model",
                "key": model,
                "n_cells": int(len(group)),
                "rank_reversal_rate": float(group["rank_reversal_flag"].mean()),
                "top_skill_top_kge_mismatch_rate": float(group["top_skill_differs_from_top_kge"].mean()),
            }
        )
    if "station_class" in rank_df.columns:
        for station_class, group in rank_df.groupby("station_class", dropna=False, sort=True):
            rows.append(
                {
                    "aggregation": "station_class",
                    "key": station_class,
                    "n_cells": int(len(group)),
                    "rank_reversal_rate": float(group["rank_reversal_flag"].mean()),
                    "top_skill_top_kge_mismatch_rate": float(group["top_skill_differs_from_top_kge"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _write_interpretation(kge: pd.DataFrame, ranks: pd.DataFrame) -> None:
    total_cells = int(len(kge))
    max_diff = float(kge["abs_alpha_h_minus_phi_h"].max(skipna=True))
    skill_vs_kge_median = float(ranks["skill_vs_kge"].median(skipna=True))
    phi_vs_r_median = float(ranks["phi_vs_r"].median(skipna=True))
    reversal_rate = float(ranks["rank_reversal_flag"].mean())
    mismatch_count = int(ranks["top_skill_differs_from_top_kge"].sum())
    complementary = (
        skill_vs_kge_median < 0.8
        or phi_vs_r_median < 0.8
        or reversal_rate >= 0.2
        or mismatch_count > 0
    )
    conclusion = (
        "KGE adds complementary ranking information beyond phi_h through r_h and beta_h"
        if complementary
        else "KGE appears redundant with phi_h in this dataset"
    )
    text = "\n".join(
        [
            "# KGE horizon-wise interpretation",
            "",
            "- decision_rule: complementary if median skill_vs_kge < 0.8, median phi_vs_r < 0.8, reversal rate >= 0.2, or any Skill/KGE top-1 mismatch.",
            f"- total_station_model_h_cells: {total_cells}",
            f"- max_abs_alpha_h_minus_phi_h: {max_diff:.12g}",
            f"- median_spearman_skill_vs_kge: {skill_vs_kge_median:.6g}",
            f"- median_spearman_phi_vs_r: {phi_vs_r_median:.6g}",
            f"- rank_reversal_flag_rate: {reversal_rate:.6g}",
            f"- top_skill_top_kge_mismatch_cells: {mismatch_count}",
            "",
            f"Conclusion: {conclusion}.",
            "",
        ]
    )
    INTERPRETATION_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing KGE table: {INPUT_PATH.relative_to(ROOT)}. Run scripts/38_compute_kge_horizon.py first.")
    df = pd.read_csv(INPUT_PATH)
    ranks = _build_rank_rows(df)
    ranks.to_csv(CORR_PATH, index=False)
    summary = _summary(ranks)
    summary.to_csv(SUMMARY_PATH, index=False)
    _write_interpretation(df, ranks)
    print(f"Wrote {CORR_PATH.relative_to(ROOT)} with {len(ranks)} rows")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)} with {len(summary)} rows")
    print(f"Wrote {INTERPRETATION_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
