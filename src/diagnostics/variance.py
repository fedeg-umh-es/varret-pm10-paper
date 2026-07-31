"""Variance-retention diagnostics for P33."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import VARIANCE_SUMMARY_COLUMNS
from src.data.validation import require_columns

MIN_N_PER_GROUP = 30
BOOTSTRAP_TYPE = "circular_moving_block"
BOOTSTRAP_RESAMPLING_UNIT = "forecast_origin"
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42
BOOTSTRAP_INTERVAL_METHOD = "percentile"


def build_variance_retention_summary(predictions_df: pd.DataFrame, skill_df: pd.DataFrame) -> pd.DataFrame:
    """Build the final diagnostic table with alpha, skill_vp, and diagnostic flags."""
    require_columns(
        predictions_df,
        ["dataset", "model", "horizon", "origin_date", "y_true", "y_pred"],
        "predictions_df",
    )
    require_columns(skill_df, ["dataset", "model", "horizon", "skill"], "skill_df")

    rows: list[dict] = []
    for (dataset, model, horizon), group in predictions_df.groupby(["dataset", "model", "horizon"]):
        group = _validated_temporal_group(group, context=f"{dataset}/{model}/h={horizon}")
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
    return predicted_variance / observed_variance if observed_variance > 0 else np.nan


def _validated_temporal_group(group: pd.DataFrame, *, context: str) -> pd.DataFrame:
    duplicate_origins = group.duplicated(["origin_date"], keep=False)
    if duplicate_origins.any():
        origins = group.loc[duplicate_origins, "origin_date"].head(3).tolist()
        raise ValueError(f"{context}: duplicate forecast origins: {origins}")
    origin_timestamps = pd.to_datetime(group["origin_date"], errors="raise")
    if not origin_timestamps.is_monotonic_increasing:
        raise ValueError(f"{context}: forecast origins are not in chronological order")
    finite = np.isfinite(group[["y_true", "y_pred"]].to_numpy(dtype=float)).all(axis=1)
    valid = group.loc[finite].copy()
    if valid.empty:
        raise ValueError(f"{context}: no finite prediction pairs")
    return valid.reset_index(drop=True)


def _default_block_length(n: int, horizon: int) -> int:
    """Use at least the overlap span and otherwise the cube-root rule."""
    if n < 1:
        raise ValueError("n must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return min(n, max(int(horizon), int(np.ceil(n ** (1.0 / 3.0)))))


def _circular_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw forecast-origin indices while preserving order inside each block."""
    if not 1 <= block_length <= n:
        raise ValueError("block_length must be between 1 and n")
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]


def _bootstrap_alpha_ci(
    group: pd.DataFrame,
    n_boot: int = BOOTSTRAP_REPLICATES,
    ci: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
    block_length: int | None = None,
) -> tuple[float, float]:
    """Percentile CI from a circular moving-block bootstrap over origins."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")
    rng = np.random.default_rng(seed)
    y_true = group["y_true"].to_numpy(dtype=float)
    y_pred = group["y_pred"].to_numpy(dtype=float)
    n = len(y_true)
    horizon = int(group["horizon"].iloc[0])
    resolved_block_length = block_length or _default_block_length(n, horizon)
    alphas = []
    for _ in range(n_boot):
        idx = _circular_block_indices(n, resolved_block_length, rng)
        vt = float(np.var(y_true[idx], ddof=0))
        vp = float(np.var(y_pred[idx], ddof=0))
        if vt > 0 and np.isfinite(vt) and np.isfinite(vp):
            alphas.append(vp / vt)
    if not alphas:
        return np.nan, np.nan
    lo = float(np.percentile(alphas, (1 - ci) / 2 * 100))
    hi = float(np.percentile(alphas, (1 + ci) / 2 * 100))
    return lo, hi
