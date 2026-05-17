"""Variance-retention diagnostics for P33."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import VARIANCE_SUMMARY_COLUMNS
from src.data.validation import require_columns


def build_variance_retention_summary(predictions_df: pd.DataFrame, skill_df: pd.DataFrame) -> pd.DataFrame:
    """Build the final diagnostic table with vr (%), skill_vp, and diagnostic flags.

    vr  = 100 * Var_pred / Var_obs  (variance retention, percent scale)
    skill_vp = skill * min(1, vr/100)  (capped: inflation does not boost skill_vp)
    """
    require_columns(
        predictions_df,
        ["dataset", "model", "horizon", "y_true", "y_pred"],
        "predictions_df",
    )
    require_columns(skill_df, ["dataset", "model", "horizon", "skill"], "skill_df")

    rows: list[dict] = []
    for (dataset, model, horizon), group in predictions_df.groupby(["dataset", "model", "horizon"]):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": horizon,
                "vr": _compute_vr(group),
            }
        )
    grouped = pd.DataFrame(rows)

    merged = grouped.merge(skill_df, on=["dataset", "model", "horizon"], how="inner")
    # Cap vr at 100 so variance inflation never inflates skill_vp above skill
    merged["skill_vp"] = merged["skill"] * merged["vr"].clip(upper=100.0) / 100.0
    merged["collapse_flag"] = merged["vr"] < 50.0
    merged["inflation_flag"] = merged["vr"] > 150.0
    merged["near_ideal_flag"] = (merged["skill"] > 0.0) & merged["vr"].between(80.0, 120.0, inclusive="both")

    return merged[VARIANCE_SUMMARY_COLUMNS].sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


def _compute_vr(group: pd.DataFrame) -> float:
    """Compute variance retention (%) = 100 * Var_pred / Var_obs."""
    observed_variance = float(np.var(group["y_true"].to_numpy(dtype=float), ddof=0))
    predicted_variance = float(np.var(group["y_pred"].to_numpy(dtype=float), ddof=0))
    if observed_variance <= 0:
        return 0.0
    return 100.0 * predicted_variance / observed_variance
