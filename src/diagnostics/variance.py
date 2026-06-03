"""Variance-retention diagnostics for P33."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schema import VARIANCE_SUMMARY_COLUMNS
from src.data.validation import require_columns

MIN_N_PER_GROUP = 30


@dataclass(frozen=True)
class VarianceDiagnosticFlags:
    collapse_flag: bool
    inflation_flag: bool
    near_ideal_flag: bool


def detect_variance_collapse(alpha: float, threshold: float = 0.5) -> bool:
    """Return True when alpha indicates variance collapse."""
    return bool(np.isfinite(alpha) and alpha < threshold)


def variance_diagnostic_flags(
    alpha: float,
    skill: float = 0.0,
    collapse_threshold: float = 0.5,
    inflation_threshold: float = 1.5,
) -> VarianceDiagnosticFlags:
    """Build simple variance-retention diagnostic flags."""
    collapse = detect_variance_collapse(alpha, threshold=collapse_threshold)
    inflation = bool(np.isfinite(alpha) and alpha > inflation_threshold)
    near_ideal = bool(np.isfinite(alpha) and skill > 0.0 and 0.8 <= alpha <= 1.2)
    return VarianceDiagnosticFlags(
        collapse_flag=collapse,
        inflation_flag=inflation,
        near_ideal_flag=near_ideal,
    )


def build_variance_retention_table(fold_metrics_df: pd.DataFrame, skill_df: pd.DataFrame) -> pd.DataFrame:
    """Build the compact variance-retention table used by diagnostics tests."""
    require_columns(
        fold_metrics_df,
        ["dataset", "model", "horizon", "y_true", "y_pred"],
        "fold_metrics_df",
    )
    skill_col = "skill_vs_baseline" if "skill_vs_baseline" in skill_df.columns else "skill"
    require_columns(skill_df, ["dataset", "model", "horizon", skill_col], "skill_df")

    rows: list[dict[str, object]] = []
    for (dataset, model, horizon), group in fold_metrics_df.groupby(["dataset", "model", "horizon"], sort=True):
        alpha = _compute_alpha(group)
        matched = skill_df[
            skill_df["dataset"].eq(dataset)
            & skill_df["model"].eq(model)
            & skill_df["horizon"].eq(horizon)
        ]
        if matched.empty:
            continue
        skill = float(matched.iloc[0][skill_col])
        flags = variance_diagnostic_flags(alpha=alpha, skill=skill)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": int(horizon),
                "skill": skill,
                "alpha": alpha,
                "skill_vp": skill * min(1.0, max(0.0, alpha)),
                "collapse_flag": flags.collapse_flag,
                "inflation_flag": flags.inflation_flag,
                "near_ideal_flag": flags.near_ideal_flag,
            }
        )

    return pd.DataFrame(rows)[
        [
            "dataset",
            "model",
            "horizon",
            "skill",
            "alpha",
            "skill_vp",
            "collapse_flag",
            "inflation_flag",
            "near_ideal_flag",
        ]
    ].sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


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
        alpha_ci_low, alpha_ci_high = _bootstrap_alpha_ci(group)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": horizon,
                "n": int(len(group)),
                "alpha": _compute_alpha(group),
                "alpha_ci_low": alpha_ci_low,
                "alpha_ci_high": alpha_ci_high,
            }
        )
    grouped = pd.DataFrame(rows)

    merged = grouped.merge(skill_df, on=["dataset", "model", "horizon"], how="inner")
    if "mae_skill" not in merged.columns:
        merged["mae_skill"] = np.nan
    merged["skill_vp"] = merged["skill"] * merged["alpha"]
    merged["collapse_flag"] = merged["alpha"] < 0.5
    merged["inflation_flag"] = merged["alpha"] > 1.5
    merged["near_ideal_flag"] = (merged["skill"] > 0.0) & merged["alpha"].between(0.8, 1.2, inclusive="both")
    merged["low_sample_flag"] = merged["n"] < MIN_N_PER_GROUP

    return merged[VARIANCE_SUMMARY_COLUMNS].sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


def _compute_alpha(group: pd.DataFrame) -> float:
    """Compute the predicted-to-observed variance ratio for one group."""
    observed_variance = float(np.var(group["y_true"].to_numpy(dtype=float), ddof=0))
    predicted_variance = float(np.var(group["y_pred"].to_numpy(dtype=float), ddof=0))
    return predicted_variance / observed_variance if observed_variance > 0 else 0.0


def _bootstrap_alpha_ci(
    group: pd.DataFrame,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    rng    = np.random.default_rng(seed)
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    n      = len(y_true)
    alphas = []
    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        vt   = float(np.var(y_true[idx], ddof=0))
        vp   = float(np.var(y_pred[idx], ddof=0))
        alphas.append(vp / vt if vt > 0 else 0.0)
    lo = float(np.percentile(alphas, (1 - ci) / 2 * 100))
    hi = float(np.percentile(alphas, (1 + ci) / 2 * 100))
    return lo, hi
