#!/usr/bin/env python3
"""Leakage-free empirical rerun for Paper A on the recovered Madrid series.

The implementation keeps preprocessing causal, fits every transform and event
threshold on the corresponding training window, and evaluates only validated
targets. Models are refitted at each expanding-window fold, then evaluated
sequentially at hourly origins inside that fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


HORIZONS = (1, 6, 24, 48)
LAGS = (0, 1, 6, 24, 48, 168)
ROLLING_WINDOWS = (6, 24, 48, 168)


def causal_inputs(observed: pd.Series) -> pd.Series:
    """Fill an unavailable input only from an earlier validated observation."""
    return observed.ffill()


def make_features(observed: pd.Series, timestamps: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """Features available at the origin plus calendar values known in advance."""
    causal = causal_inputs(observed)
    frame = pd.DataFrame(index=timestamps)
    for lag in LAGS:
        frame[f"pm10_lag_{lag}"] = causal.shift(lag)
    for window in ROLLING_WINDOWS:
        roll = causal.rolling(window, min_periods=window)
        frame[f"pm10_mean_{window}"] = roll.mean()
        frame[f"pm10_std_{window}"] = roll.std()
    verification_time = timestamps + pd.to_timedelta(horizon, unit="h")
    frame["target_hour_sin"] = np.sin(2 * np.pi * verification_time.hour / 24)
    frame["target_hour_cos"] = np.cos(2 * np.pi * verification_time.hour / 24)
    frame["target_dow_sin"] = np.sin(2 * np.pi * verification_time.dayofweek / 7)
    frame["target_dow_cos"] = np.cos(2 * np.pi * verification_time.dayofweek / 7)
    frame["target_month_sin"] = np.sin(2 * np.pi * (verification_time.month - 1) / 12)
    frame["target_month_cos"] = np.cos(2 * np.pi * (verification_time.month - 1) / 12)
    return frame


def expanding_folds(n: int) -> list[tuple[int, int, int]]:
    initial_train = n // 2
    test_size = (n - initial_train) // 5
    return [
        (fold, initial_train + fold * test_size, initial_train + (fold + 1) * test_size)
        for fold in range(5)
    ]


def fit_lightgbm_predictions(
    observed: pd.Series, timestamps: pd.DatetimeIndex, folds: list[tuple[int, int, int]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    causal = causal_inputs(observed)
    for fold, train_end, test_end in folds:
        threshold = float(observed.iloc[:train_end].dropna().quantile(0.75))
        for horizon in HORIZONS:
            features = make_features(observed, timestamps, horizon)
            target = observed.shift(-horizon)
            train_mask = np.arange(len(observed)) + horizon < train_end
            test_mask = (
                (np.arange(len(observed)) >= train_end)
                & (np.arange(len(observed)) + horizon < test_end)
            )
            usable_train = train_mask & features.notna().all(axis=1).to_numpy() & target.notna().to_numpy()
            usable_test = test_mask & features.notna().all(axis=1).to_numpy() & target.notna().to_numpy()
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                deterministic=True,
                force_col_wise=True,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(features.loc[usable_train], target.loc[usable_train])
            prediction = model.predict(features.loc[usable_test])
            origins = np.flatnonzero(usable_test)
            for origin, y_pred in zip(origins, prediction, strict=True):
                rows.append(
                    prediction_row(
                        "lightgbm", fold, horizon, origin, timestamps, observed,
                        causal, float(y_pred), threshold,
                    )
                )
    return rows


def fit_sarima_predictions(
    observed: pd.Series, timestamps: pd.DatetimeIndex, folds: list[tuple[int, int, int]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    causal = causal_inputs(observed)
    for fold, train_end, test_end in folds:
        started = time.monotonic()
        train = causal.iloc[:train_end].copy()
        if train.isna().any():
            train = train.fillna(float(observed.iloc[:train_end].dropna().median()))
        threshold = float(observed.iloc[:train_end].dropna().quantile(0.75))
        fitted = SARIMAX(
            train.to_numpy(),
            order=(1, 0, 1),
            seasonal_order=(1, 0, 0, 24),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=100)
        for origin in range(train_end, test_end):
            fitted = fitted.append([float(causal.iloc[origin])], refit=False)
            forecasts = np.asarray(fitted.forecast(max(HORIZONS)), dtype=float)
            for horizon in HORIZONS:
                target_idx = origin + horizon
                if target_idx < test_end and pd.notna(observed.iloc[target_idx]):
                    rows.append(
                        prediction_row(
                            "sarima", fold, horizon, origin, timestamps, observed,
                            causal, float(forecasts[horizon - 1]), threshold,
                        )
                    )
        print(f"SARIMA fold {fold} completed in {time.monotonic() - started:.1f}s", flush=True)
    return rows


def prediction_row(
    model: str,
    fold: int,
    horizon: int,
    origin: int,
    timestamps: pd.DatetimeIndex,
    observed: pd.Series,
    causal: pd.Series,
    y_pred: float,
    threshold: float,
) -> dict[str, object]:
    target_idx = origin + horizon
    return {
        "protocol": "rolling_origin",
        "model": model,
        "fold": fold,
        "origin_time": timestamps[origin],
        "target_time": timestamps[target_idx],
        "horizon": horizon,
        "y_true": float(observed.iloc[target_idx]),
        "y_pred": y_pred,
        "y_persistence": float(causal.iloc[origin]),
        "p75_train": threshold,
    }


def aggregate_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    event_rows = []
    for (model, horizon), group in predictions.groupby(["model", "horizon"], sort=True):
        error = group["y_pred"] - group["y_true"]
        baseline_error = group["y_persistence"] - group["y_true"]
        rmse = float(np.sqrt(np.mean(np.square(error))))
        rmse_persistence = float(np.sqrt(np.mean(np.square(baseline_error))))
        skill = 1 - rmse / rmse_persistence
        variance_observed = float(group["y_true"].var(ddof=1))
        variance_predicted = float(group["y_pred"].var(ddof=1))
        retention = variance_predicted / variance_observed
        metric_rows.append(
            {
                "model": model,
                "horizon": horizon,
                "n": len(group),
                "rmse": rmse,
                "rmse_persistence": rmse_persistence,
                "skill_rmse": skill,
                "variance_observed": variance_observed,
                "variance_predicted": variance_predicted,
                "variance_retention_pct": 100 * retention,
                "skill_vp": skill * min(1.0, retention),
            }
        )
        actual_event = group["y_true"] > group["p75_train"]
        predicted_event = group["y_pred"] > group["p75_train"]
        tp = int((actual_event & predicted_event).sum())
        event_rows.append(
            {
                "model": model,
                "horizon": horizon,
                "n": len(group),
                "events": int(actual_event.sum()),
                "threshold_policy": "fold_train_p75",
                "recall": tp / int(actual_event.sum()) if actual_event.any() else np.nan,
                "precision": tp / int(predicted_event.sum()) if predicted_event.any() else np.nan,
                "flag_rate": float(predicted_event.mean()),
                "base_rate": float(actual_event.mean()),
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(event_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/casa_de_campo_pm10_2023.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--models", default="lightgbm,sarima")
    parser.add_argument("--protocol", choices=("rolling_origin", "holdout"), default="rolling_origin")
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["timestamp"])
    timestamps = pd.DatetimeIndex(data["timestamp"])
    observed = pd.Series(data["pm10"].to_numpy(dtype=float), index=timestamps)
    folds = (
        expanding_folds(len(observed))
        if args.protocol == "rolling_origin"
        else [(0, int(0.8 * len(observed)), len(observed))]
    )
    selected = {item.strip() for item in args.models.split(",")}
    rows: list[dict[str, object]] = []
    if "lightgbm" in selected:
        rows.extend(fit_lightgbm_predictions(observed, timestamps, folds))
    if "sarima" in selected:
        rows.extend(fit_sarima_predictions(observed, timestamps, folds))
    if not rows:
        raise ValueError("No supported models selected")

    predictions = pd.DataFrame(rows).sort_values(["model", "fold", "horizon", "origin_time"])
    predictions["protocol"] = args.protocol
    metrics, events = aggregate_metrics(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output_dir / f"predictions_{args.protocol}.parquet", index=False)
    metrics.to_csv(args.output_dir / f"metrics_{args.protocol}.csv", index=False)
    events.to_csv(args.output_dir / f"events_p75_{args.protocol}.csv", index=False)
    manifest = {
        "protocol": (
            "five expanding folds; initial train 50%; five disjoint 10% test windows"
            if args.protocol == "rolling_origin"
            else "single chronological 80/20 blocked holdout"
        ),
        "horizons_hours": list(HORIZONS),
        "models": sorted(selected),
        "preprocessing": "validated targets only; input gaps forward-filled causally; no scaling",
        "event_threshold": "P75 estimated separately from each fold training window",
        "lightgbm": "one direct model per fold and horizon",
        "sarima": "(1,0,1)(1,0,0)[24], fit per fold; state updated sequentially without refit",
        "random_seed": 42,
    }
    (args.output_dir / f"run_manifest_{args.protocol}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(events.to_string(index=False))


if __name__ == "__main__":
    main()
