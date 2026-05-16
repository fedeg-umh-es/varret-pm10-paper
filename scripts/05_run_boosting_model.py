"""Run LightGBM boosting model under rolling-origin evaluation.

Uses direct multi-step forecasting (one model per horizon h).
Features are built inline from the daily series: lags [1,2,3,7,14] and
rolling statistics [7,14] — no separate feature engineering step required.
The model is re-fitted every `lgbm_refit_every` unique origins.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.schema import PREDICTIONS_COLUMNS
from src.models.lgbm_model import LGBMMultiHorizon
from src.splits.rolling_origin import generate_rolling_origin_folds


def _features_at(series: np.ndarray, idx: int) -> dict:
    """Lag and rolling features for the observation at position `idx`."""
    feats: dict = {}
    for lag in (1, 2, 3, 7, 14):
        feats[f"lag_{lag}"] = float(series[idx - lag]) if idx >= lag else float("nan")
    for w in (7, 14):
        sl = series[max(0, idx - w + 1): idx + 1]
        feats[f"roll_mean_{w}"] = float(np.mean(sl))
        feats[f"roll_std_{w}"] = float(np.std(sl)) if len(sl) > 1 else 0.0
    return feats


def _feature_matrix(series: np.ndarray, indices: list[int]) -> pd.DataFrame:
    df = pd.DataFrame([_features_at(series, i) for i in indices])
    return df.ffill().bfill().fillna(0.0)


def main() -> None:
    cfg_ds = yaml.safe_load(Path("configs/datasets/pm10.yaml").read_text())["dataset"]
    cfg_ro = yaml.safe_load(Path("configs/evaluation/rolling_origin.yaml").read_text())["rolling_origin"]

    processed_path = Path(cfg_ds["processed_path"])
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset missing: {processed_path}. Run scripts/02 first.")

    df = pd.read_parquet(processed_path)
    series = df["y"].values
    horizons = list(range(1, cfg_ro["max_horizon"] + 1))

    folds = generate_rolling_origin_folds(
        df,
        date_column="date",
        min_train_size=cfg_ro["min_train_size"],
        max_horizon=cfg_ro["max_horizon"],
        step_size=cfg_ro.get("step_size", 1),
    )

    refit_every = cfg_ro.get("lgbm_refit_every", 30)
    folds_by_origin: dict = defaultdict(list)
    for fold in folds:
        folds_by_origin[fold.origin_date].append(fold)

    unique_origins = sorted(folds_by_origin.keys())
    refit_set = set(unique_origins[::refit_every])
    print(
        f"LightGBM: {len(unique_origins)} origins, "
        f"re-fitting at {len(refit_set)} checkpoints (every {refit_every} origins)"
    )

    lgbm_params = {
        "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
        "num_leaves": 31, "random_state": 42, "verbose": -1,
    }
    model: LGBMMultiHorizon | None = None
    dataset_name = cfg_ds["name"]
    rows = []

    for origin in unique_origins:
        origin_folds = folds_by_origin[origin]
        train_indices = origin_folds[0].train_indices
        origin_idx = train_indices[-1]

        if origin in refit_set or model is None:
            print(f"  Fitting LightGBM on {len(train_indices)} obs (origin {origin.date()}) …", flush=True)
            valid = [i for i in train_indices if i + max(horizons) < len(series)]
            if len(valid) < 20:
                model = None
            else:
                X_tr = _feature_matrix(series, valid)
                y_tr = pd.DataFrame({f"h{h}": [series[i + h] for i in valid] for h in horizons})
                model = LGBMMultiHorizon(horizons=horizons, lgb_params=lgbm_params)
                model.fit(X_tr, y_tr)

        if model is None:
            for fold in origin_folds:
                rows.append(
                    {
                        "dataset": dataset_name, "model": "lgbm",
                        "fold": fold.fold, "origin_date": str(fold.origin_date.date()),
                        "horizon": fold.horizon, "date": str(fold.target_date.date()),
                        "y_true": float(df["y"].iloc[fold.test_index]), "y_pred": float("nan"),
                    }
                )
            continue

        X_pred = _feature_matrix(series, [origin_idx])
        preds = model.predict(X_pred)[0]  # shape: (n_horizons,)

        for fold in origin_folds:
            h_idx = horizons.index(fold.horizon)
            rows.append(
                {
                    "dataset": dataset_name, "model": "lgbm",
                    "fold": fold.fold, "origin_date": str(fold.origin_date.date()),
                    "horizon": fold.horizon, "date": str(fold.target_date.date()),
                    "y_true": float(df["y"].iloc[fold.test_index]),
                    "y_pred": float(preds[h_idx]),
                }
            )

    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PREDICTIONS_COLUMNS).to_parquet(out_dir / "lgbm.parquet", index=False)
    print(f"lgbm: {len(rows)} predictions written → {out_dir / 'lgbm.parquet'}")


if __name__ == "__main__":
    main()
