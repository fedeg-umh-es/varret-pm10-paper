"""Build persistence-relative RMSE skill tables from canonical predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path("outputs/metrics/predictions.csv")
DEFAULT_OUTPUT = Path("outputs/metrics/skill_summary.csv")


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    values = y_true.astype(float).to_numpy() - y_pred.astype(float).to_numpy()
    return float(np.sqrt(np.mean(values**2)))


def build_skill_table(predictions: pd.DataFrame, baseline_model: str = "persistence") -> pd.DataFrame:
    required = {"dataset", "model", "horizon", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions table missing columns: {sorted(missing)}")

    baseline = predictions[predictions["model"].eq(baseline_model)]
    if baseline.empty:
        models = sorted(predictions["model"].dropna().unique())
        raise ValueError(f"Baseline model '{baseline_model}' not found. Available models: {models}")

    baseline_rmse = {
        (dataset, int(horizon)): _rmse(group["y_true"], group["y_pred"])
        for (dataset, horizon), group in baseline.groupby(["dataset", "horizon"], sort=True)
    }

    rows: list[dict[str, object]] = []
    for (dataset, model, horizon), group in predictions.groupby(["dataset", "model", "horizon"], sort=True):
        if model == baseline_model:
            continue
        key = (dataset, int(horizon))
        if key not in baseline_rmse:
            continue
        rmse_baseline = baseline_rmse[key]
        rmse_model = _rmse(group["y_true"], group["y_pred"])
        skill = 1.0 - (rmse_model / rmse_baseline) if rmse_baseline > 0 else np.nan
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": int(horizon),
                "skill": skill,
            }
        )

    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("No non-baseline model rows were available to compute skill.")
    return output.sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-model", default="persistence")
    args = parser.parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(f"Missing predictions table: {args.predictions}")

    predictions = pd.read_csv(args.predictions)
    skill = build_skill_table(predictions, baseline_model=args.baseline_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    skill.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(skill)} rows")


if __name__ == "__main__":
    main()
