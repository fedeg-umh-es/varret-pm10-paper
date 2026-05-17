"""Aggregate all prediction parquets into skill and combined metrics tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import PREDICTIONS_COLUMNS, SKILL_COLUMNS


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def main() -> None:
    pred_dir = Path("outputs/predictions")
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(pred_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No prediction parquets found in {pred_dir}. Run scripts/03-05 first."
        )

    all_preds = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    all_preds.to_csv(metrics_dir / "predictions.csv", index=False)
    print(f"Combined {len(all_preds)} predictions from {len(parquet_files)} models → {metrics_dir/'predictions.csv'}")

    # Persistence reference for each (dataset, horizon, date)
    persist_ref = (
        all_preds[all_preds["model"] == "persistence"]
        [["dataset", "horizon", "date", "y_true", "y_pred"]]
        .rename(columns={"y_pred": "y_persist"})
    )

    skill_rows = []

    # Persistence itself: skill = 0 by definition
    for (dataset, horizon) in all_preds[["dataset", "horizon"]].drop_duplicates().itertuples(index=False):
        skill_rows.append({"dataset": dataset, "model": "persistence", "horizon": horizon, "skill": 0.0})

    for (dataset, model, horizon), group in all_preds.groupby(["dataset", "model", "horizon"]):
        if model == "persistence":
            continue
        ref = persist_ref[(persist_ref["dataset"] == dataset) & (persist_ref["horizon"] == horizon)]
        merged = group.merge(ref[["date", "y_persist"]], on="date", how="inner")
        if merged.empty:
            continue
        rmse_m = _rmse(merged["y_true"].values, merged["y_pred"].values)
        rmse_p = _rmse(merged["y_true"].values, merged["y_persist"].values)
        if rmse_p > 0 and not np.isnan(rmse_m):
            skill = float(1.0 - rmse_m / rmse_p)
        else:
            skill = float("nan")
        skill_rows.append({"dataset": dataset, "model": model, "horizon": horizon, "skill": skill})

    skill_df = (
        pd.DataFrame(skill_rows, columns=SKILL_COLUMNS)
        .sort_values(["dataset", "model", "horizon"])
        .reset_index(drop=True)
    )
    skill_df.to_csv(metrics_dir / "skill_summary.csv", index=False)
    print(f"Skill summary → {metrics_dir / 'skill_summary.csv'}")
    print(skill_df.to_string(index=False))


if __name__ == "__main__":
    main()
