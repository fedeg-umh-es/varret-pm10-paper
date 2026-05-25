#!/usr/bin/env python3
"""Murphy-style MSE decomposition for prediction tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PREDICTIONS = Path("outputs/metrics/predictions.csv")
DEFAULT_OUTPUT = Path("outputs/tables/murphy_decomposition_all_stations.csv")
TOL = 1e-6


def _rho(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    std_pred = float(np.std(y_pred, ddof=0))
    std_true = float(np.std(y_true, ddof=0))
    if std_pred == 0 or std_true == 0:
        return 0.0
    return float(np.corrcoef(y_pred, y_true)[0, 1])


def build_murphy_table(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "model", "horizon", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions table missing columns: {sorted(missing)}")

    rows = []
    for (dataset, model, horizon), group in predictions.groupby(["dataset", "model", "horizon"], sort=True):
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        mean_true = float(np.mean(y_true))
        mean_pred = float(np.mean(y_pred))
        std_true = float(np.std(y_true, ddof=0))
        std_pred = float(np.std(y_pred, ddof=0))
        var_true = float(np.var(y_true, ddof=0))
        rho = _rho(y_pred, y_true)

        mse = float(np.mean((y_pred - y_true) ** 2))
        bias_sq = (mean_pred - mean_true) ** 2
        cond_bias_sq = (std_pred - rho * std_true) ** 2
        irreducible_sq = (1 - rho**2) * var_true
        mse_total = bias_sq + cond_bias_sq + irreducible_sq
        if abs(mse_total - mse) > TOL * max(1.0, mse):
            raise ValueError(
                f"Murphy decomposition mismatch for {dataset}/{model}/h={horizon}: "
                f"mse={mse}, components={mse_total}"
            )
        alpha = (std_pred / std_true) ** 2 if std_true > 0 else 0.0
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": int(horizon),
                "mse": mse,
                "bias_sq": bias_sq,
                "cond_bias_sq": cond_bias_sq,
                "irreducible_sq": irreducible_sq,
                "rho": rho,
                "std_pred_ug": std_pred,
                "std_true_ug": std_true,
                "alpha": alpha,
            }
        )
    if not rows:
        raise ValueError("No Murphy decomposition rows generated.")
    return pd.DataFrame(rows).sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Murphy decomposition table.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    table = build_murphy_table(predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(table)} rows")


if __name__ == "__main__":
    main()
