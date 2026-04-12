"""
Train SARIMA baseline
=====================
Trains a fixed-order SARIMA on each rolling-origin fold and generates
multi-step recursive forecasts.

KEY CONSTRAINTS (do not relax without updating the paper):
  - Trained exclusively on train_idx of each fold (no test leakage).
  - Operates on the raw normalized PM10 series, NOT on tabular lag features.
  - Forecast is purely recursive from the fitted model.
  - Order (p,d,q)(P,D,Q,s) is fixed globally from config — no per-fold search.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import argparse
import yaml
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.sarima_model import SarimaForecaster


def _sarima_order_from_cfg(cfg: dict):
    """Extract SARIMA order tuples from config."""
    scfg = cfg['models']['sarima']
    order = tuple(scfg['order'])
    seasonal_order = tuple(scfg['seasonal_order'])
    trend = scfg.get('trend', None)
    enforce_stationarity = scfg.get('enforce_stationarity', False)
    enforce_invertibility = scfg.get('enforce_invertibility', False)
    return order, seasonal_order, trend, enforce_stationarity, enforce_invertibility


def train_rolling_origin(
    pm10_series: np.ndarray,
    splits: dict,
    horizons: list,
    cfg: dict,
) -> pd.DataFrame:
    """Fit SARIMA per fold, forecast recursively, return long-format predictions."""
    order, seasonal_order, trend, enf_s, enf_i = _sarima_order_from_cfg(cfg)
    predictions_list = []

    for fold_info in splits['folds']:
        fold = fold_info['fold']
        train_idx = fold_info['train_idx']
        test_idx = fold_info['test_idx']

        # --- training series: only train_idx, chronological ---
        y_train = pm10_series[train_idx]

        print(f"  Fold {fold}: fitting SARIMA on {len(y_train)} observations...", flush=True)
        try:
            forecaster = SarimaForecaster(
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=enf_s,
                enforce_invertibility=enf_i,
            )
            forecaster.fit(y_train)
        except Exception as exc:
            print(f"    WARNING: SARIMA fit failed on fold {fold}: {exc}")
            print(f"    Filling fold {fold} with NaN predictions.")
            for global_idx in test_idx:
                for h in horizons:
                    predictions_list.append({
                        'fold': fold,
                        'sample_idx': global_idx,
                        'horizon': h,
                        'y_pred': np.nan,
                    })
            continue

        max_h = max(horizons)

        # Each test observation requires its own forecast origin.
        # The origin shifts as we move through the test set.
        # We re-fit only at the fold level (fixed-origin within fold),
        # which is the correct interpretation for a rolling-origin scheme
        # where the test window is evaluated from a single model fit.
        #
        # If you want a fully rolling re-fit (one fit per test step),
        # replace the block below. For Paper A this is intentionally
        # kept simple: one fit per fold origin.

        all_forecasts = forecaster.forecast(steps=len(test_idx) + max_h)
        # all_forecasts[k] = prediction for time (end_of_train + k + 1)

        for step, global_idx in enumerate(test_idx):
            # The test point at position `step` relative to end of train:
            # h=1 forecast corresponds to all_forecasts[step],
            # h=h forecast corresponds to all_forecasts[step + h - 1].
            for h in horizons:
                fc_index = step + h - 1
                if fc_index < len(all_forecasts):
                    y_pred = float(all_forecasts[fc_index])
                else:
                    y_pred = np.nan
                predictions_list.append({
                    'fold': fold,
                    'sample_idx': global_idx,
                    'horizon': h,
                    'y_pred': y_pred,
                })

        print(f"    Done. {len(test_idx)} test points × {len(horizons)} horizons recorded.")

    return pd.DataFrame(predictions_list)


def train_holdout(
    pm10_series: np.ndarray,
    splits: dict,
    horizons: list,
    cfg: dict,
) -> pd.DataFrame:
    """Fit SARIMA on holdout train, forecast recursively for holdout test."""
    order, seasonal_order, trend, enf_s, enf_i = _sarima_order_from_cfg(cfg)

    train_idx = splits['train_idx']
    test_idx = splits['test_idx']

    y_train = pm10_series[train_idx]

    print(f"  Fitting SARIMA on {len(y_train)} holdout train observations...", flush=True)
    try:
        forecaster = SarimaForecaster(
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enf_s,
            enforce_invertibility=enf_i,
        )
        forecaster.fit(y_train)
    except Exception as exc:
        print(f"  WARNING: SARIMA holdout fit failed: {exc}")
        rows = [
            {'sample_idx': idx, 'horizon': h, 'y_pred': np.nan}
            for idx in test_idx
            for h in horizons
        ]
        return pd.DataFrame(rows)

    max_h = max(horizons)
    all_forecasts = forecaster.forecast(steps=len(test_idx) + max_h)

    predictions_list = []
    for step, global_idx in enumerate(test_idx):
        for h in horizons:
            fc_index = step + h - 1
            y_pred = float(all_forecasts[fc_index]) if fc_index < len(all_forecasts) else np.nan
            predictions_list.append({
                'sample_idx': global_idx,
                'horizon': h,
                'y_pred': y_pred,
            })

    print(f"  Done. {len(test_idx)} test points × {len(horizons)} horizons recorded.")
    return pd.DataFrame(predictions_list)


def main():
    parser = argparse.ArgumentParser(description="Train SARIMA baseline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    with open("config/horizons.yaml") as f:
        horizons_cfg = yaml.safe_load(f)
    horizons = horizons_cfg['horizons']

    processed_dir = Path(cfg['paths']['processed_dir'])
    pred_dir = Path(cfg['paths']['predictions_dir'])
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Load normalized series (SARIMA works on the time series directly)
    print("Loading normalized PM10 series...")
    df_pre = pd.read_parquet(processed_dir / "pm10_preprocessed.parquet")
    pm10_series = df_pre['pm10_normalized'].values

    print("Loading splits...")
    with open(processed_dir / "splits_rolling_origin.json") as f:
        splits_ro = json.load(f)
    with open(processed_dir / "splits_holdout.json") as f:
        splits_ho = json.load(f)

    # Rolling-origin
    print("\nTraining SARIMA (rolling-origin)...")
    pred_ro = train_rolling_origin(pm10_series, splits_ro, horizons, cfg)
    path_ro = pred_dir / "sarima_rolling_origin.parquet"
    pred_ro.to_parquet(path_ro)
    print(f"Saved → {path_ro} ({len(pred_ro)} records)")

    # Holdout
    print("\nTraining SARIMA (holdout)...")
    pred_ho = train_holdout(pm10_series, splits_ho, horizons, cfg)
    path_ho = pred_dir / "sarima_holdout.parquet"
    pred_ho.to_parquet(path_ho)
    print(f"Saved → {path_ho} ({len(pred_ho)} records)")


if __name__ == "__main__":
    main()
