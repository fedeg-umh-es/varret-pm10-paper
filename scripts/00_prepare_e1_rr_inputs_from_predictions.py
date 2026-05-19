"""Prepare E1-RR post-evaluation inputs from a prediction table.

This script consumes a long prediction table that includes both persistence and
non-persistence model forecasts, then writes the two minimal inputs required by
`07_build_variance_retention_table.py`:

- outputs/metrics/predictions.csv
- outputs/metrics/skill_summary.csv

It is intentionally limited to the current E1-RR post-evaluation work package.
It does not run new models and does not consume E2-MET or E3-PROB outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("../p34-variance-retention-api/outputs/predictions/predictions_all_models.csv")
DEFAULT_PREDICTIONS_OUTPUT = Path("outputs/metrics/predictions.csv")
DEFAULT_SKILL_OUTPUT = Path("outputs/metrics/skill_summary.csv")


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    values = (y_true.astype(float) - y_pred.astype(float)).to_numpy()
    return float(np.sqrt(np.mean(values**2)))


def _normalise_predictions(raw: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Map a project-specific prediction table to the canonical contract."""
    required = {"origin", "forecast_timestamp", "horizon", "model", "y_true", "y_pred"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")

    df = raw.copy()
    df["dataset"] = dataset_name
    df["fold"] = df["origin"]
    df["origin_date"] = pd.to_datetime(df["origin"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["date"] = pd.to_datetime(df["forecast_timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    return df[["dataset", "model", "fold", "origin_date", "horizon", "date", "y_true", "y_pred"]]


def _build_skill_summary(predictions: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    """Compute RMSE skill against the persistence baseline by dataset/model/horizon."""
    baseline = predictions[predictions["model"] == baseline_model]
    if baseline.empty:
        available = sorted(predictions["model"].dropna().unique())
        raise ValueError(f"Baseline model '{baseline_model}' not found. Available models: {available}")

    baseline_rmse = (
        baseline.groupby(["dataset", "horizon"], as_index=False)
        .apply(lambda g: pd.Series({"rmse_baseline": _rmse(g["y_true"], g["y_pred"])}), include_groups=False)
    )

    rows: list[dict] = []
    for (dataset, model, horizon), group in predictions.groupby(["dataset", "model", "horizon"]):
        if model == baseline_model:
            continue
        rmse_model = _rmse(group["y_true"], group["y_pred"])
        matched = baseline_rmse[(baseline_rmse["dataset"] == dataset) & (baseline_rmse["horizon"] == horizon)]
        if matched.empty:
            continue
        rmse_baseline = float(matched.iloc[0]["rmse_baseline"])
        skill = 1.0 - (rmse_model / rmse_baseline) if rmse_baseline > 0 else np.nan
        rows.append({"dataset": dataset, "model": model, "horizon": horizon, "skill": skill})

    skill_df = pd.DataFrame(rows)
    if skill_df.empty:
        raise ValueError("No non-baseline model rows were available to compute skill_summary.csv")
    return skill_df.sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare E1-RR variance-retention inputs from prediction table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source predictions CSV")
    parser.add_argument("--dataset", default="e1_rr", help="Dataset identifier to write into canonical outputs")
    parser.add_argument("--baseline-model", default="persistence", help="Model name used as persistence baseline")
    parser.add_argument("--predictions-output", type=Path, default=DEFAULT_PREDICTIONS_OUTPUT)
    parser.add_argument("--skill-output", type=Path, default=DEFAULT_SKILL_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing source predictions table: {args.input}")

    raw = pd.read_csv(args.input)
    predictions = _normalise_predictions(raw, args.dataset)
    skill = _build_skill_summary(predictions, args.baseline_model)

    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.skill_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)
    skill.to_csv(args.skill_output, index=False)

    print(f"Wrote {args.predictions_output} with {len(predictions)} rows")
    print(f"Wrote {args.skill_output} with {len(skill)} rows")
    print("Models:", sorted(predictions["model"].dropna().unique()))
    print("Horizons:", int(predictions["horizon"].min()), "to", int(predictions["horizon"].max()))


if __name__ == "__main__":
    main()
