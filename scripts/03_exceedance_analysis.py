#!/usr/bin/env python3
"""Exceedance-event analysis for rolling-origin prediction tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = (("p75", 75.0), ("p90", 90.0), ("abs_50", 50.0))


def _event_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: np.ndarray) -> dict[str, float]:
    event_true = y_true > threshold
    event_pred = y_pred > threshold
    tp = int(np.sum(event_true & event_pred))
    fn = int(np.sum(event_true & ~event_pred))
    fp = int(np.sum(~event_true & event_pred))
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan
    return {
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "flag_rate": float(np.mean(event_pred)),
        "base_rate": float(np.mean(event_true)),
    }


def _observed_history(predictions: pd.DataFrame) -> pd.Series:
    obs = predictions[["date", "y_true"]].copy()
    obs["date"] = pd.to_datetime(obs["date"])
    obs["y_true"] = pd.to_numeric(obs["y_true"], errors="coerce")
    obs = obs.dropna().drop_duplicates("date").sort_values("date")
    return pd.Series(obs["y_true"].to_numpy(dtype=float), index=obs["date"])


def _history_percentile(history: pd.Series, origin_date: pd.Timestamp, percentile: float) -> float:
    train = history[history.index <= origin_date]
    if train.empty:
        return np.nan
    return float(np.percentile(train.to_numpy(dtype=float), percentile))


def build_exceedance_table(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "horizon", "origin_date", "date", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions table missing columns: {sorted(missing)}")

    df = predictions.copy()
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["origin_date", "y_true", "y_pred"])

    rows = []
    for dataset, dataset_df in df.groupby("dataset", sort=True):
        history = _observed_history(dataset_df)
        for threshold_type, threshold_param in THRESHOLDS:
            with_thresholds = dataset_df.copy()
            if threshold_type == "abs_50":
                with_thresholds["threshold_value"] = threshold_param
            else:
                with_thresholds["threshold_value"] = [
                    _history_percentile(history, origin, threshold_param)
                    for origin in with_thresholds["origin_date"]
                ]
            with_thresholds = with_thresholds.dropna(subset=["threshold_value"])
            for (model, horizon), group in with_thresholds.groupby(["model", "horizon"], sort=True):
                metrics = _event_metrics(
                    group["y_true"].to_numpy(dtype=float),
                    group["y_pred"].to_numpy(dtype=float),
                    group["threshold_value"].to_numpy(dtype=float),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "horizon": int(horizon),
                        "threshold_type": threshold_type,
                        "threshold_value": float(group["threshold_value"].mean()),
                        **metrics,
                    }
                )
    if not rows:
        raise ValueError("No exceedance rows generated.")
    return pd.DataFrame(rows).sort_values(["dataset", "model", "horizon", "threshold_type"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute exceedance metrics from predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--station", default=None, help="Optional station label retained for compatibility")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    table = build_exceedance_table(predictions)
    if args.station is not None:
        table.insert(1, "station", args.station)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(table)} rows")


if __name__ == "__main__":
    main()
