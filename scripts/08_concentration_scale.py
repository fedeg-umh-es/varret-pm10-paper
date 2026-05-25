#!/usr/bin/env python3
"""Concentration-scale variability diagnostics in raw PM10 units."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path("outputs/metrics/predictions.csv")
DEFAULT_METADATA = Path("outputs/tables/variance_retention_all_stations.csv")
DEFAULT_OUTPUT = Path("outputs/tables/concentration_scale_summary.csv")
MODELS = ("hgb_direct", "ridge_direct", "stl_ridge_direct", "sarima", "seasonal_naive")


def _metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols = [c for c in ["dataset", "station_name", "station_type"] if c in df.columns]
    return df[cols].drop_duplicates("dataset") if "dataset" in cols else pd.DataFrame()


def build_concentration_summary(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "horizon", "date", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions table missing columns: {sorted(missing)}")
    rows = []
    for dataset, dataset_df in predictions.groupby("dataset", sort=True):
        observed = dataset_df[["date", "y_true"]].drop_duplicates("date")
        obs_mean = float(observed["y_true"].mean())
        obs_std = float(observed["y_true"].std(ddof=0))
        row = {
            "dataset": dataset,
            "obs_mean_ug": obs_mean,
            "obs_std_ug": obs_std,
            "observed_range_ug": 2 * obs_std,
        }
        for model in MODELS:
            model_df = dataset_df[dataset_df["model"].eq(model)]
            for horizon in (1, 7):
                values = model_df.loc[model_df["horizon"].eq(horizon), "y_pred"].to_numpy(dtype=float)
                pred_std = float(np.std(values, ddof=0)) if len(values) else np.nan
                row[f"pred_std_{model}_h{horizon}"] = pred_std
                row[f"predicted_range_{model}_h{horizon}_ug"] = 2 * pred_std if np.isfinite(pred_std) else np.nan
                row[f"alpha_{model}_h{horizon}"] = (pred_std / obs_std) ** 2 if obs_std > 0 and np.isfinite(pred_std) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not metadata.empty:
        out = out.merge(metadata, on="dataset", how="left")
        front = [c for c in ["dataset", "station_name", "station_type"] if c in out.columns]
        out = out[front + [c for c in out.columns if c not in front]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build concentration-scale diagnostics.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    table = build_concentration_summary(predictions, _metadata(args.metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(table)} rows")


if __name__ == "__main__":
    main()
