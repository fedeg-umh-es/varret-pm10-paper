#!/usr/bin/env python3
"""Compute KGE horizon-wise diagnostics from existing rolling-origin outputs.

AUDIT SOURCE: reused pipeline routes from scripts/build_unified_predictions_table.py,
scripts/09_build_comprehensive_unified_table.py, scripts/13_build_five_model_diagnostic_summary.py,
and KGE definitions from manuscript_assets/paper_c_kge/paper.tex.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kge_diagnostics import kge_horizon_table


PREDICTIONS_PATH = ROOT / "outputs" / "metrics" / "predictions_all_stations.csv"
MASTER_PATH = ROOT / "outputs" / "tables" / "master_diagnostic_table.csv"
RESULTS_DIR = ROOT / "results"
KGE_TABLE_PATH = RESULTS_DIR / "kge_horizon_table.csv"
SUMMARY_PATH = RESULTS_DIR / "kge_components_summary.csv"
INTERPRETATION_PATH = RESULTS_DIR / "kge_interpretation.md"
TARGET_MODELS = ["hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"]


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing canonical predictions table: {PREDICTIONS_PATH.relative_to(ROOT)}")
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing canonical diagnostic table: {MASTER_PATH.relative_to(ROOT)}")
    return pd.read_csv(PREDICTIONS_PATH), pd.read_csv(MASTER_PATH)


def _merge_master_columns(kge: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    master["h"] = master["horizon"].astype(int)
    keep = [
        "dataset",
        "model",
        "h",
        "skill",
        "mae_skill",
        "skill_vp",
        "alpha",
        "station_id",
        "station_name",
        "province",
        "station_type",
        "station_class",
        "lat",
        "lon",
        "altitude_m",
        "dem_code",
    ]
    keep = [col for col in keep if col in master.columns]
    merged = kge.merge(master[keep], on=["dataset", "model", "h"], how="inner")
    merged["station"] = merged.get("station_id", merged["dataset"])
    merged = merged.rename(columns={"skill": "Skill_h", "alpha": "alpha_source"})
    if "alpha_source" in merged.columns:
        merged["phi_source_sqrt_alpha"] = np.where(
            merged["alpha_source"].ge(0), np.sqrt(merged["alpha_source"]), np.nan
        )
        merged["abs_alpha_h_minus_phi_source_sqrt_alpha"] = (
            merged["alpha_h"] - merged["phi_source_sqrt_alpha"]
        ).abs()
    merged["abs_alpha_h_minus_phi_h"] = (merged["alpha_h"] - merged["phi_h"]).abs()
    first_cols = [
        "station",
        "station_id",
        "station_name",
        "dataset",
        "model",
        "h",
        "n",
        "Skill_h",
        "phi_h",
        "r_h",
        "alpha_h",
        "beta_h",
        "KGE_h",
        "KGE_persistence_h",
        "KGE_skill_h",
    ]
    ordered = [col for col in first_cols if col in merged.columns]
    ordered += [col for col in merged.columns if col not in ordered and col != "horizon"]
    return merged[ordered].sort_values(["station", "model", "h"]).reset_index(drop=True)


def _iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def _build_summary(table: pd.DataFrame) -> pd.DataFrame:
    metrics = ["r_h", "alpha_h", "beta_h", "KGE_h", "KGE_skill_h"]
    grouped = table.groupby(["model", "h"], sort=True)
    rows = []
    for (model, h), group in grouped:
        row: dict[str, object] = {
            "model": model,
            "h": int(h),
            "n_stations_valid": int(group[metrics].dropna(how="all").shape[0]),
        }
        for metric in metrics:
            row[f"{metric}_median"] = float(group[metric].median(skipna=True))
            row[f"{metric}_iqr"] = _iqr(group[metric].dropna())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_interpretation_stub(table: pd.DataFrame) -> None:
    max_diff = float(table["abs_alpha_h_minus_phi_h"].max(skipna=True))
    text = "\n".join(
        [
            "# KGE horizon-wise interpretation",
            "",
            "Rank comparison has not been run yet. Execute scripts/39_rank_comparison_kge_vs_phi.py.",
            "",
            f"- total_cells: {len(table)}",
            f"- max_abs_alpha_h_minus_phi_h: {max_diff:.12g}",
            "",
        ]
    )
    INTERPRETATION_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    predictions, master = _load_inputs()
    predictions = predictions[predictions["model"].isin([*TARGET_MODELS, "persistence"])].copy()
    table = kge_horizon_table(predictions)
    table = table[table["model"].isin(TARGET_MODELS)].copy()
    table = _merge_master_columns(table, master)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(KGE_TABLE_PATH, index=False)
    summary = _build_summary(table)
    summary.to_csv(SUMMARY_PATH, index=False)
    _write_interpretation_stub(table)

    print(f"Wrote {KGE_TABLE_PATH.relative_to(ROOT)} with {len(table)} rows")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)} with {len(summary)} rows")
    print("NaNs by column:")
    print(table.isna().sum().to_string())
    print(f"max_abs(alpha_h - phi_h): {table['abs_alpha_h_minus_phi_h'].max(skipna=True):.12g}")
    if "abs_alpha_h_minus_phi_source_sqrt_alpha" in table.columns:
        print(
            "max_abs(alpha_h - sqrt(alpha_source)): "
            f"{table['abs_alpha_h_minus_phi_source_sqrt_alpha'].max(skipna=True):.12g}"
        )
    if len(table) != 595:
        print(f"WARNING: expected 595 station-model-h cells from the current paper; found {len(table)}.")


if __name__ == "__main__":
    main()

