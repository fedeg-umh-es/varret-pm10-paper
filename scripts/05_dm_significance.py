#!/usr/bin/env python3
"""Diebold-Mariano significance tests against persistence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


DEFAULT_OUTPUT = Path("outputs/tables/dm_significance_all_stations.csv")


def _autocovariance(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if lag == 0:
        return float(np.mean(x * x))
    return float(np.mean(x[lag:] * x[:-lag])) if lag < n else 0.0


def _dm_hln(loss_diff: np.ndarray, horizon: int) -> tuple[float, float]:
    d = np.asarray(loss_diff, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n <= max(2, horizon):
        return np.nan, np.nan
    d_bar = float(np.mean(d))
    gamma_sum = _autocovariance(d, 0)
    for lag in range(1, horizon):
        gamma_sum += 2.0 * _autocovariance(d, lag)
    var_d_bar = gamma_sum / n
    if var_d_bar <= 0 or not np.isfinite(var_d_bar):
        return np.nan, np.nan
    dm_stat = d_bar / np.sqrt(var_d_bar)
    correction_arg = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if correction_arg <= 0:
        return np.nan, np.nan
    dm_hln = float(dm_stat * np.sqrt(correction_arg))
    pval = float(2 * (1 - t.cdf(abs(dm_hln), df=n - 1)))
    return dm_hln, pval


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


def build_dm_table(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "fold", "date", "horizon", "y_true", "y_pred"}
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
        merged = group.merge(
            base[["fold", "date", "y_true", "y_pred"]].rename(
                columns={"y_true": "y_true_baseline", "y_pred": "y_pred_persistence"}
            ),
            on=["fold", "date"],
            how="inner",
        )
        if merged.empty:
            continue
        if not np.allclose(merged["y_true"], merged["y_true_baseline"], equal_nan=True):
            raise ValueError(f"y_true mismatch after merge for {dataset}/{model}/h={horizon}")
        loss_diff = (merged["y_true"] - merged["y_pred_persistence"]) ** 2 - (
            merged["y_true"] - merged["y_pred"]
        ) ** 2
        dm_stat, pval = _dm_hln(loss_diff.to_numpy(dtype=float), int(horizon))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": int(horizon),
                "n_pairs": int(len(merged)),
                "dm_stat": dm_stat,
                "dm_pval_raw": pval,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No DM tests generated.")
    table["dm_pval_bh"] = np.nan
    for dataset, idx in table.groupby("dataset").groups.items():
        table.loc[idx, "dm_pval_bh"] = _bh_adjust(table.loc[idx, "dm_pval_raw"])
    table["dm_significant"] = table["dm_pval_bh"] < 0.05
    return table.sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DM tests against persistence.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--station", default=None, help="Optional station label retained for compatibility")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    table = build_dm_table(predictions)
    if args.station is not None:
        table.insert(1, "station", args.station)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(table)} rows")


if __name__ == "__main__":
    main()
