"""Run SARIMA under rolling-origin evaluation with periodic re-fitting.

SARIMA is the time-series linear model for P33. It is re-fitted every
`sarima_refit_every` unique origins (default 30) to keep total runtime tractable
while maintaining a leakage-free expanding-window protocol.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.schema import PREDICTIONS_COLUMNS
from src.models.sarima_model import SarimaForecaster
from src.splits.rolling_origin import generate_rolling_origin_folds


def main() -> None:
    cfg_ds = yaml.safe_load(Path("configs/datasets/pm10.yaml").read_text())["dataset"]
    cfg_ro = yaml.safe_load(Path("configs/evaluation/rolling_origin.yaml").read_text())["rolling_origin"]

    processed_path = Path(cfg_ds["processed_path"])
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset missing: {processed_path}. Run scripts/02 first.")

    df = pd.read_parquet(processed_path)
    folds = generate_rolling_origin_folds(
        df,
        date_column="date",
        min_train_size=cfg_ro["min_train_size"],
        max_horizon=cfg_ro["max_horizon"],
        step_size=cfg_ro.get("step_size", 1),
    )

    refit_every = cfg_ro.get("sarima_refit_every", 30)

    folds_by_origin: dict = defaultdict(list)
    for fold in folds:
        folds_by_origin[fold.origin_date].append(fold)

    unique_origins = sorted(folds_by_origin.keys())
    refit_set = set(unique_origins[::refit_every])
    print(
        f"SARIMA: {len(unique_origins)} origins, "
        f"re-fitting at {len(refit_set)} checkpoints (every {refit_every} origins)"
    )

    sarima_params = dict(order=(1, 0, 1), seasonal_order=(1, 0, 1, 7))
    forecaster: SarimaForecaster | None = None
    dataset_name = cfg_ds["name"]
    rows = []

    for origin in unique_origins:
        origin_folds = folds_by_origin[origin]
        history = df["y"].iloc[origin_folds[0].train_indices].values
        max_h = max(f.horizon for f in origin_folds)

        if origin in refit_set or forecaster is None:
            print(f"  Fitting SARIMA on {len(history)} obs (origin {origin.date()}) …", flush=True)
            forecaster = SarimaForecaster(**sarima_params)
            try:
                forecaster.fit(history)
            except Exception as exc:
                print(f"  WARNING: fit failed ({exc}); filling with NaN")
                for fold in origin_folds:
                    rows.append(
                        {
                            "dataset": dataset_name, "model": "sarima",
                            "fold": fold.fold, "origin_date": str(fold.origin_date.date()),
                            "horizon": fold.horizon, "date": str(fold.target_date.date()),
                            "y_true": float(df["y"].iloc[fold.test_index]), "y_pred": float("nan"),
                        }
                    )
                forecaster = None
                continue

        try:
            preds = forecaster.forecast(steps=max_h)
        except Exception:
            preds = np.full(max_h, float("nan"))

        for fold in origin_folds:
            rows.append(
                {
                    "dataset": dataset_name, "model": "sarima",
                    "fold": fold.fold, "origin_date": str(fold.origin_date.date()),
                    "horizon": fold.horizon, "date": str(fold.target_date.date()),
                    "y_true": float(df["y"].iloc[fold.test_index]),
                    "y_pred": float(preds[fold.horizon - 1]) if fold.horizon <= len(preds) else float("nan"),
                }
            )

    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PREDICTIONS_COLUMNS).to_parquet(out_dir / "sarima.parquet", index=False)
    print(f"sarima: {len(rows)} predictions written → {out_dir / 'sarima.parquet'}")


if __name__ == "__main__":
    main()
