#!/usr/bin/env python3
"""Diebold-Mariano significance tests against persistence."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pairing import pair_model_and_baseline


DEFAULT_OUTPUT = Path("outputs/tables/dm_significance_all_stations.csv")
HAC_KERNEL = "bartlett"
DM_ALTERNATIVE = "two-sided"


def _autocovariance(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if lag == 0:
        return float(np.mean(x * x))
    return float(np.dot(x[lag:], x[:-lag]) / n) if lag < n else 0.0


def _dm_hln_result(
    loss_diff: np.ndarray,
    horizon: int,
    *,
    hac_lag: int | None = None,
    apply_hln: bool = True,
) -> dict[str, float | int | str | bool]:
    d = np.asarray(loss_diff, dtype=float)
    if not np.isfinite(d).all():
        return _invalid_dm_result("nonfinite_loss_differential", len(d), hac_lag, apply_hln)
    n = len(d)
    lag = horizon - 1 if hac_lag is None else int(hac_lag)
    if lag < 0:
        raise ValueError("hac_lag must be non-negative")
    effective_horizon = lag + 1
    if n <= max(2, effective_horizon):
        return _invalid_dm_result("insufficient_sample", n, lag, apply_hln)
    d_bar = float(np.mean(d))
    if np.allclose(d, d[0], rtol=1e-12, atol=1e-14):
        return _invalid_dm_result("all_loss_differentials_equal", n, lag, apply_hln, d_bar)

    long_run_variance = _autocovariance(d, 0)
    for autocov_lag in range(1, lag + 1):
        weight = 1.0 - autocov_lag / (lag + 1.0)
        long_run_variance += 2.0 * weight * _autocovariance(d, autocov_lag)
    if not np.isfinite(long_run_variance) or long_run_variance < -1e-12:
        return _invalid_dm_result("invalid_hac_estimate", n, lag, apply_hln, d_bar)
    if long_run_variance <= 1e-12:
        return _invalid_dm_result("zero_long_run_variance", n, lag, apply_hln, d_bar)

    var_d_bar = long_run_variance / n
    dm_stat = d_bar / np.sqrt(var_d_bar)
    corrected_stat = float(dm_stat)
    if apply_hln:
        correction_arg = (
            n + 1 - 2 * effective_horizon + effective_horizon * (effective_horizon - 1) / n
        ) / n
        if correction_arg <= 0 or not np.isfinite(correction_arg):
            return _invalid_dm_result("invalid_hln_correction", n, lag, apply_hln, d_bar)
        corrected_stat *= float(np.sqrt(correction_arg))
    pval = float(2 * t.sf(abs(corrected_stat), df=n - 1))
    return {
        "dm_stat": corrected_stat,
        "dm_pval_raw": pval,
        "mean_loss_diff": d_bar,
        "long_run_variance": float(long_run_variance),
        "hac_lag": lag,
        "hln_applied": apply_hln,
        "dm_status": "ok",
    }


def _invalid_dm_result(
    status: str,
    n: int,
    hac_lag: int | None,
    apply_hln: bool,
    mean_loss_diff: float = np.nan,
) -> dict[str, float | int | str | bool]:
    return {
        "dm_stat": np.nan,
        "dm_pval_raw": np.nan,
        "mean_loss_diff": mean_loss_diff,
        "long_run_variance": np.nan,
        "hac_lag": np.nan if hac_lag is None else int(hac_lag),
        "hln_applied": apply_hln,
        "dm_status": status,
    }


def _dm_hln(
    loss_diff: np.ndarray,
    horizon: int,
    *,
    hac_lag: int | None = None,
    apply_hln: bool = True,
) -> tuple[float, float]:
    """Backward-compatible statistic/p-value wrapper."""
    result = _dm_hln_result(loss_diff, horizon, hac_lag=hac_lag, apply_hln=apply_hln)
    return float(result["dm_stat"]), float(result["dm_pval_raw"])


def _overlap_hac_lag(origin_dates: pd.Series, horizon: int) -> tuple[int, float]:
    origins = pd.Series(pd.to_datetime(origin_dates, errors="raise").drop_duplicates().sort_values())
    deltas = origins.diff().dropna().dt.total_seconds().to_numpy(dtype=float) / 86400.0
    positive = deltas[deltas > 0]
    if not len(positive):
        return 0, np.nan
    stride_days = float(np.min(positive))
    lag = max(0, math.ceil(float(horizon) / stride_days) - 1)
    return min(lag, max(0, len(origins) - 1)), stride_days


def _bh_adjust(pvals: pd.Series) -> pd.Series:
    valid = pvals.dropna()
    adjusted = pd.Series(np.nan, index=pvals.index, dtype=float)
    if valid.empty:
        return adjusted
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    m = len(ranked)
    raw_adj = ranked * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adjusted.loc[order] = np.minimum(monotone, 1.0)
    return adjusted


def build_dm_table(predictions: pd.DataFrame, *, apply_hln: bool = True) -> pd.DataFrame:
    required = {"dataset", "model", "fold", "origin_date", "date", "horizon", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions table missing columns: {sorted(missing)}")
    baseline = predictions[predictions["model"].eq("persistence")]
    if baseline.empty:
        raise ValueError("Predictions table has no persistence baseline.")

    rows = []
    for (dataset, model, horizon), group in predictions.groupby(["dataset", "model", "horizon"], sort=True):
        if model == "persistence":
            continue
        base = baseline[(baseline["dataset"].eq(dataset)) & (baseline["horizon"].eq(horizon))]
        merged = pair_model_and_baseline(
            group,
            base,
            context=f"DM {dataset}/{model}/h={horizon}",
        )
        loss_diff = (merged["y_true"] - merged["y_pred_baseline"]) ** 2 - (
            merged["y_true"] - merged["y_pred"]
        ) ** 2
        hac_lag, stride_days = _overlap_hac_lag(merged["origin_date"], int(horizon))
        result = _dm_hln_result(
            loss_diff.to_numpy(dtype=float),
            int(horizon),
            hac_lag=hac_lag,
            apply_hln=apply_hln,
        )
        rows.append({
            "dataset": dataset,
            "model": model,
            "horizon": int(horizon),
            "n_pairs": int(len(merged)),
            "n_candidate_pairs": int(merged.attrs["candidate_pairs"]),
            "n_invalid_pairs": int(merged.attrs["invalid_pairs_dropped"]),
            "origin_stride_days": stride_days,
            "hac_kernel": HAC_KERNEL,
            "alternative": DM_ALTERNATIVE,
            "loss_differential_sign": "persistence_squared_error_minus_model_squared_error",
            **result,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No DM tests generated.")
    table["dm_pval_bh"] = np.nan
    for dataset, idx in table.groupby("dataset").groups.items():
        table.loc[idx, "dm_pval_bh"] = _bh_adjust(table.loc[idx, "dm_pval_raw"])
    table["dm_significant"] = table["dm_status"].eq("ok") & table["dm_pval_bh"].lt(0.05)
    table["dm_model_better"] = table["dm_significant"] & table["mean_loss_diff"].gt(0)
    table["dm_direction"] = np.select(
        [table["mean_loss_diff"].gt(0), table["mean_loss_diff"].lt(0)],
        ["model_better", "persistence_better"],
        default="tie_or_not_evaluable",
    )
    return table.sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DM tests against persistence.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--station", default=None, help="Optional station label retained for compatibility")
    parser.add_argument("--no-hln", action="store_true", help="Disable the HLN finite-sample correction")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    table = build_dm_table(predictions, apply_hln=not args.no_hln)
    if args.station is not None:
        table.insert(1, "station", args.station)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(table)} rows")


if __name__ == "__main__":
    main()
