"""Variance-retention diagnostics for P33."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import VARIANCE_SUMMARY_COLUMNS
from src.data.validation import require_columns


def build_variance_retention_summary(predictions_df: pd.DataFrame, skill_df: pd.DataFrame) -> pd.DataFrame:
    """Build the final diagnostic table with alpha, skill_vp, and diagnostic flags."""
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
                "alpha": _compute_alpha(group),
            }
        )
    grouped = pd.DataFrame(rows)

    merged = grouped.merge(skill_df, on=["dataset", "model", "horizon"], how="inner")
    merged["skill_vp"] = merged["skill"] * merged["alpha"]
    merged["collapse_flag"] = merged["alpha"] < 0.5
    merged["inflation_flag"] = merged["alpha"] > 1.5
    merged["near_ideal_flag"] = (merged["skill"] > 0.0) & merged["alpha"].between(0.8, 1.2, inclusive="both")

    return merged[VARIANCE_SUMMARY_COLUMNS].sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


def _compute_alpha(group: pd.DataFrame) -> float:
    """Compute the predicted-to-observed variance ratio for one group."""
    observed_variance = float(np.var(group["y_true"].to_numpy(dtype=float), ddof=0))
    predicted_variance = float(np.var(group["y_pred"].to_numpy(dtype=float), ddof=0))
    return predicted_variance / observed_variance if observed_variance > 0 else 0.0
