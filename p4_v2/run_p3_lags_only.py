#!/usr/bin/env python3
"""Execute the frozen P3 daily PM10 lags-only protocol.

This runner creates row-level predictions only. It deliberately does not calculate
forecast metrics or scientific diagnostics. It is restartable at station-shard
boundaries and refuses to run when frozen input hashes do not match the protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = ROOT / "data" / "processed"
MANIFESTS = ROOT / "manifests"
RESULTS = ROOT / "results"

DAILY_PATH = DATA / "pm10_daily_canonical.parquet"
PANEL_PATH = DATA / "station_panel_frozen.csv"
PANEL_MANIFEST_PATH = MANIFESTS / "station_panel_manifest.json"
PROTOCOL_PATH = ROOT / "docs" / "P3_MULTISTATION_PROTOCOL_FREEZE.md"
PROTOCOL_MANIFEST_PATH = MANIFESTS / "p3_multistation_protocol_manifest.json"
RAW_MANIFEST_PATH = MANIFESTS / "raw_data_manifest.json"

PROTOCOL_HASH = "2f776008e8fd23b307beac405513bcf06ada1cf5006a8826304b49458bd96516"
PANEL_HASH = "bc2a921e610dc60976f0f9609de09a7d5733e367102f6fe14a2c11361ebb0d27"
DAILY_HASH = "1d4fbf031ecece900803303944d0d15e64e5bbe22260823ea57e3f6bcb2f8a34"
PANEL_MANIFEST_HASH = "f05d78c062fbdf4f7a8d30b57815893ecb19d411f8d5b7a52eee6d6bbd1a3924"
PROTOCOL_MANIFEST_HASH = "8c457ddf94891f410d317bcfd4361842723777e0f214e68ccae45a1d21beb328"

CONDITION = "lags_only"
HORIZONS = tuple(range(1, 8))
LAGS = (1, 2, 3, 6, 7, 14, 28)
MIN_TRAIN_DAYS = 365
MIN_TRAIN_ROWS = 300
REFIT_BLOCK_DAYS = 28
EVENT_QUANTILE = 0.75
MODEL_NAMES = ("xgboost_direct", "sarima")

LAG_COLUMNS = [f"pm10_lag_{lag}" for lag in LAGS]
CALENDAR_COLUMNS = [
    "target_dow_sin",
    "target_dow_cos",
    "target_doy_sin",
    "target_doy_cos",
    "target_month_sin",
    "target_month_cos",
]
FEATURE_COLUMNS = LAG_COLUMNS + CALENDAR_COLUMNS

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 1,
    "tree_method": "hist",
}
SARIMA_ORDER = (1, 0, 1)
SARIMA_SEASONAL_ORDER = (1, 0, 0, 7)
SARIMA_KWARGS: dict[str, Any] = {
    "trend": "n",
    "enforce_stationarity": False,
    "enforce_invertibility": False,
}
SARIMA_MAXITER = 120

ROW_COLUMNS = [
    "station_id",
    "condition",
    "model",
    "fold",
    "refit_block",
    "origin",
    "target_time",
    "horizon",
    "y_true",
    "y_pred",
    "y_persistence",
    "event_threshold_p75",
    "run_id",
    "protocol_hash",
    "source_panel_hash",
]
FAILURE_COLUMNS = [
    "timestamp_utc",
    "station_id",
    "model",
    "failure_type",
    "fold",
    "refit_block",
    "origin_start",
    "origin_end",
    "horizon",
    "target_time",
    "exception",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def verify_frozen_inputs() -> dict[str, str]:
    expected = {
        DAILY_PATH: DAILY_HASH,
        PANEL_PATH: PANEL_HASH,
        PANEL_MANIFEST_PATH: PANEL_MANIFEST_HASH,
        PROTOCOL_PATH: PROTOCOL_HASH,
        PROTOCOL_MANIFEST_PATH: PROTOCOL_MANIFEST_HASH,
    }
    actual: dict[str, str] = {}
    mismatches: list[str] = []
    for path, expected_hash in expected.items():
        if not path.exists():
            mismatches.append(f"missing: {path}")
            continue
        actual[str(path)] = sha256(path)
        if actual[str(path)] != expected_hash:
            mismatches.append(f"hash mismatch: {path}: {actual[str(path)]} != {expected_hash}")
    if mismatches:
        raise RuntimeError("FROZEN_INPUT_MISMATCH\n" + "\n".join(mismatches))

    protocol_manifest = json.loads(PROTOCOL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if protocol_manifest.get("protocol", {}).get("sha256") != PROTOCOL_HASH:
        raise RuntimeError("Frozen protocol manifest does not reference the required protocol hash")
    if protocol_manifest.get("model_training_authorized") is not False:
        raise RuntimeError("Frozen protocol manifest does not keep model training authorization closed")
    if protocol_manifest.get("condition") != CONDITION:
        raise RuntimeError("Frozen protocol manifest condition is not lags_only")
    return actual


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    panel = pd.read_csv(PANEL_PATH, dtype={"station_id": str})
    panel["station_id"] = panel["station_id"].astype(str)
    panel = panel.sort_values("station_id", kind="mergesort").reset_index(drop=True)
    if len(panel) != 324 or panel["station_id"].nunique() != 324:
        raise RuntimeError("Frozen station panel is not the expected unique 324-station panel")

    daily = pd.read_parquet(DAILY_PATH, columns=["station_id", "date", "pm10_daily"])
    daily["station_id"] = daily["station_id"].astype(str)
    daily["date"] = pd.to_datetime(daily["date"], format="%Y-%m-%d")
    daily["pm10_daily"] = pd.to_numeric(daily["pm10_daily"], errors="coerce")
    if daily.duplicated(["station_id", "date"]).any():
        raise RuntimeError("Duplicate station/date rows in canonical daily data")
    source_panel_hash = sha256(PANEL_PATH)
    return panel, daily, source_panel_hash


def calendar_features(target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    dow = target_dates.dayofweek.to_numpy(dtype=float) / 7.0
    doy = (target_dates.dayofyear.to_numpy(dtype=float) - 1.0) / 365.25
    month = (target_dates.month.to_numpy(dtype=float) - 1.0) / 12.0
    return pd.DataFrame(
        {
            "target_dow_sin": np.sin(2.0 * np.pi * dow),
            "target_dow_cos": np.cos(2.0 * np.pi * dow),
            "target_doy_sin": np.sin(2.0 * np.pi * doy),
            "target_doy_cos": np.cos(2.0 * np.pi * doy),
            "target_month_sin": np.sin(2.0 * np.pi * month),
            "target_month_cos": np.cos(2.0 * np.pi * month),
        },
        index=target_dates,
    )


def prepare_station(
    station_id: str, panel_row: pd.Series, daily: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame, pd.DatetimeIndex, list[dict[str, Any]], dict[int, int]]:
    first = pd.Timestamp(panel_row["first_date"])
    last = pd.Timestamp(panel_row["last_date"])
    index = pd.date_range(first, last, freq="D")
    source = daily[daily["station_id"] == station_id][["date", "pm10_daily"]].copy()
    source = source.set_index("date")["pm10_daily"].astype(float)
    if source.index.has_duplicates:
        raise RuntimeError(f"Duplicate station/date rows for {station_id}")
    series = source.reindex(index)
    lag_df = pd.DataFrame({f"pm10_lag_{lag}": series.shift(lag) for lag in LAGS}, index=index)
    lag_ok = lag_df.notna().all(axis=1)
    candidate_origins = index[lag_ok.to_numpy()]
    target_ok = {h: series.shift(-h).notna() for h in HORIZONS}

    first_origin: pd.Timestamp | None = None
    for origin in candidate_origins:
        if (origin - first).days < MIN_TRAIN_DAYS:
            continue
        enough = True
        for h in HORIZONS:
            complete = lag_ok & target_ok[h] & ((index + pd.Timedelta(days=h)) < origin)
            if int(complete.sum()) < MIN_TRAIN_ROWS:
                enough = False
                break
        if enough:
            first_origin = origin
            break
    if first_origin is None:
        raise RuntimeError(f"No eligible first origin for {station_id}")

    fold_specs: list[dict[str, Any]] = []
    fold = 0
    fold_start = first_origin
    while fold_start <= last:
        fold_end = fold_start + pd.Timedelta(days=REFIT_BLOCK_DAYS)
        origins = candidate_origins[(candidate_origins >= fold_start) & (candidate_origins < fold_end)]
        key_count = sum(int(target_ok[h].loc[origins].sum()) for h in HORIZONS)
        if len(origins) and key_count:
            train_counts = {
                h: int((lag_ok & target_ok[h] & ((index + pd.Timedelta(days=h)) < fold_start)).sum())
                for h in HORIZONS
            }
            fold_specs.append(
                {
                    "fold": fold,
                    "refit_block": fold,
                    "fold_start": fold_start,
                    "fold_end": fold_end,
                    "origins": origins,
                    "key_count": key_count,
                    "train_counts": train_counts,
                }
            )
            fold += 1
        fold_start = fold_start + pd.Timedelta(days=REFIT_BLOCK_DAYS)

    baseline = series.ffill()
    if baseline.loc[candidate_origins].isna().any():
        raise RuntimeError(f"Persistence baseline unavailable for an eligible origin at {station_id}")
    return series, lag_df, candidate_origins, fold_specs, {h: int(target_ok[h].sum()) for h in HORIZONS}


def failure_row(
    station_id: str,
    model: str,
    failure_type: str,
    fold: Any = "",
    refit_block: Any = "",
    origin_start: Any = "",
    origin_end: Any = "",
    horizon: Any = "",
    target_time: Any = "",
    exception: str = "",
) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now(),
        "station_id": station_id,
        "model": model,
        "failure_type": failure_type,
        "fold": fold,
        "refit_block": refit_block,
        "origin_start": str(origin_start)[:10] if origin_start != "" else "",
        "origin_end": str(origin_end)[:10] if origin_end != "" else "",
        "horizon": horizon,
        "target_time": str(target_time)[:10] if target_time != "" else "",
        "exception": exception[:2000],
    }


def build_rows_for_station(
    station_id: str,
    panel_row: pd.Series,
    daily: pd.DataFrame,
    run_id: str,
    smoke: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    series, lag_df, candidate_origins, fold_specs, target_counts = prepare_station(station_id, panel_row, daily)
    index = series.index
    baseline = series.ffill()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    expected_keys = 0
    xgb_keys = 0
    sarima_keys = 0
    model_status = {"xgboost_direct": "not_started", "sarima": "not_started"}

    for spec in fold_specs:
        fold = int(spec["fold"])
        fold_start = pd.Timestamp(spec["fold_start"])
        fold_end = pd.Timestamp(spec["fold_end"])
        origins = pd.DatetimeIndex(spec["origins"])
        expected_keys += int(spec["key_count"])
        train_end = fold_start - pd.Timedelta(days=1)

        train_values = series.loc[:train_end].dropna().to_numpy(dtype=float)
        if len(train_values) == 0:
            failures.append(failure_row(station_id, "all", "insufficient_training_data", fold, fold, fold_start, fold_end, exception="No nonmissing training values"))
            continue
        event_threshold = float(np.quantile(train_values, EVENT_QUANTILE, method="linear"))

        xgb_models: dict[int, XGBRegressor] = {}
        xgb_fit_ok = True
        model_status["xgboost_direct"] = "running"
        for h in HORIZONS:
            complete = lag_df.notna().all(axis=1) & series.shift(-h).notna() & ((index + pd.Timedelta(days=h)) < fold_start)
            if int(complete.sum()) < MIN_TRAIN_ROWS:
                failures.append(failure_row(station_id, "xgboost_direct", "insufficient_training_data", fold, fold, fold_start, fold_end, h, exception=f"{int(complete.sum())} complete rows < {MIN_TRAIN_ROWS}"))
                xgb_fit_ok = False
                continue
            train_origins = index[complete.to_numpy()]
            x_lag = lag_df.loc[train_origins, LAG_COLUMNS].reset_index(drop=True)
            target_dates = train_origins + pd.Timedelta(days=h)
            x_cal = calendar_features(pd.DatetimeIndex(target_dates)).reset_index(drop=True)
            x_train = pd.concat([x_lag, x_cal], axis=1)[FEATURE_COLUMNS]
            y_train = series.loc[target_dates].to_numpy(dtype=float)
            try:
                model = XGBRegressor(**XGB_PARAMS)
                model.fit(x_train, y_train, verbose=False)
                xgb_models[h] = model
            except Exception as exc:  # pragma: no cover - runtime-dependent
                failures.append(failure_row(station_id, "xgboost_direct", "fit_failure", fold, fold, fold_start, fold_end, h, exception=traceback.format_exc()))
                xgb_fit_ok = False
        if xgb_fit_ok:
            model_status["xgboost_direct"] = "complete"
        else:
            model_status["xgboost_direct"] = "partial_or_failed"

        for h in HORIZONS:
            if h not in xgb_models:
                continue
            valid_origins = origins[series.shift(-h).loc[origins].notna().to_numpy()]
            for origin in valid_origins:
                target_time = origin + pd.Timedelta(days=h)
                x_lag = lag_df.loc[[origin], LAG_COLUMNS].reset_index(drop=True)
                x_cal = calendar_features(pd.DatetimeIndex([target_time])).reset_index(drop=True)
                x_future = pd.concat([x_lag, x_cal], axis=1)[FEATURE_COLUMNS]
                try:
                    pred = float(xgb_models[h].predict(x_future)[0])
                    rows.append({
                        "station_id": station_id,
                        "condition": CONDITION,
                        "model": "xgboost_direct",
                        "fold": fold,
                        "refit_block": fold,
                        "origin": origin.strftime("%Y-%m-%d"),
                        "target_time": target_time.strftime("%Y-%m-%d"),
                        "horizon": h,
                        "y_true": float(series.loc[target_time]),
                        "y_pred": pred,
                        "y_persistence": float(baseline.loc[origin]),
                        "event_threshold_p75": event_threshold,
                        "run_id": run_id,
                        "protocol_hash": PROTOCOL_HASH,
                        "source_panel_hash": PANEL_HASH,
                    })
                    xgb_keys += 1
                except Exception as exc:  # pragma: no cover - runtime-dependent
                    failures.append(failure_row(station_id, "xgboost_direct", "forecast_failure", fold, fold, origin, origin, h, target_time, traceback.format_exc()))

        model_status["sarima"] = "running"
        sarima_result = None
        try:
            sarima_train = series.loc[:train_end].copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sarima_model = SARIMAX(
                    sarima_train,
                    order=SARIMA_ORDER,
                    seasonal_order=SARIMA_SEASONAL_ORDER,
                    **SARIMA_KWARGS,
                )
                sarima_result = sarima_model.fit(disp=False, maxiter=SARIMA_MAXITER)
        except Exception as exc:  # pragma: no cover - runtime-dependent
            failures.append(failure_row(station_id, "sarima", "fit_failure", fold, fold, fold_start, fold_end, exception=traceback.format_exc()))
            model_status["sarima"] = "partial_or_failed"

        if sarima_result is not None:
            last_update = train_end
            sarima_ok = True
            for origin in origins:
                try:
                    update_index = pd.date_range(last_update + pd.Timedelta(days=1), origin, freq="D")
                    if len(update_index):
                        update_values = series.reindex(update_index)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            sarima_result = sarima_result.append(update_values, refit=False)
                        last_update = origin
                    forecast = np.asarray(sarima_result.get_forecast(steps=max(HORIZONS)).predicted_mean, dtype=float)
                    if len(forecast) < max(HORIZONS) or not np.isfinite(forecast).all():
                        raise RuntimeError("SARIMA returned non-finite or incomplete forecast")
                    for h in HORIZONS:
                        target_time = origin + pd.Timedelta(days=h)
                        if target_time not in series.index or pd.isna(series.loc[target_time]):
                            continue
                        rows.append({
                            "station_id": station_id,
                            "condition": CONDITION,
                            "model": "sarima",
                            "fold": fold,
                            "refit_block": fold,
                            "origin": origin.strftime("%Y-%m-%d"),
                            "target_time": target_time.strftime("%Y-%m-%d"),
                            "horizon": h,
                            "y_true": float(series.loc[target_time]),
                            "y_pred": float(forecast[h - 1]),
                            "y_persistence": float(baseline.loc[origin]),
                            "event_threshold_p75": event_threshold,
                            "run_id": run_id,
                            "protocol_hash": PROTOCOL_HASH,
                            "source_panel_hash": PANEL_HASH,
                        })
                        sarima_keys += 1
                except Exception as exc:  # pragma: no cover - runtime-dependent
                    sarima_ok = False
                    failures.append(failure_row(station_id, "sarima", "forecast_failure", fold, fold, origin, origin, exception=traceback.format_exc()))
                    break
            model_status["sarima"] = "complete" if sarima_ok else "partial_or_failed"

        for origin in origins:
            if not lag_df.loc[origin, LAG_COLUMNS].notna().all():
                failures.append(failure_row(station_id, "all", "missing_lag", fold, fold, origin, origin))
            for h in HORIZONS:
                target_time = origin + pd.Timedelta(days=h)
                if target_time not in series.index or pd.isna(series.loc[target_time]):
                    failures.append(failure_row(station_id, "all", "missing_target", fold, fold, origin, origin, h, target_time))

    # Missing-lag dates before the first eligible origin are still logged as data exclusions.
    # This is a ledger operation only and does not affect the frozen support.
    for origin in candidate_origins:
        if not lag_df.loc[origin, LAG_COLUMNS].notna().all():
            failures.append(failure_row(station_id, "all", "missing_lag", exception="candidate-origin check"))

    rows.sort(key=lambda r: (r["station_id"], r["fold"], r["origin"], r["horizon"], r["model"]))
    metadata = {
        "station_id": station_id,
        "expected_keys": expected_keys,
        "xgboost_keys": xgb_keys,
        "sarima_keys": sarima_keys,
        "models": model_status,
        "folds": len(fold_specs),
        "target_counts": target_counts,
    }
    return rows, failures, metadata


def rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            ("station_id", pa.string()),
            ("condition", pa.string()),
            ("model", pa.string()),
            ("fold", pa.int32()),
            ("refit_block", pa.int32()),
            ("origin", pa.string()),
            ("target_time", pa.string()),
            ("horizon", pa.int16()),
            ("y_true", pa.float64()),
            ("y_pred", pa.float64()),
            ("y_persistence", pa.float64()),
            ("event_threshold_p75", pa.float64()),
            ("run_id", pa.string()),
            ("protocol_hash", pa.string()),
            ("source_panel_hash", pa.string()),
        ]
    )
    if not rows:
        return pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
    return pa.Table.from_pydict({name: [row.get(name) for row in rows] for name in ROW_COLUMNS}, schema=schema)


def write_shard(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(rows_to_table(rows), tmp, compression="zstd")
    os.replace(tmp, path)
    return sha256(path)


def write_failure_shard(path: Path, failures: list[dict[str, Any]]) -> str:
    atomic_csv(path, FAILURE_COLUMNS, failures)
    return sha256(path)


def read_progress(path: Path, run_id: str | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"run_id": run_id, "protocol_hash": PROTOCOL_HASH, "stations": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_hash") != PROTOCOL_HASH:
        raise RuntimeError("Existing execution progress belongs to another protocol hash")
    if run_id is not None and payload.get("run_id") != run_id:
        raise RuntimeError("Existing execution progress belongs to another run_id")
    return payload


def verify_completed_shard(path: Path, expected_sha: str) -> bool:
    if not path.exists() or sha256(path) != expected_sha:
        return False
    table = pq.read_table(path, columns=["condition", "protocol_hash"])
    if table.num_rows == 0:
        return False
    return set(table["condition"].to_pylist()) == {CONDITION} and set(table["protocol_hash"].to_pylist()) == {PROTOCOL_HASH}


def process_one_station(
    station_id: str,
    panel_row: pd.Series,
    daily: pd.DataFrame,
    run_id: str,
    shard_dir: Path,
    failure_dir: Path,
    progress: dict[str, Any],
    smoke: bool = False,
) -> dict[str, Any]:
    station_entry = progress.setdefault("stations", {}).get(station_id, {})
    shard_path = shard_dir / f"station_{station_id}.parquet"
    failure_path = failure_dir / f"station_{station_id}.csv"
    if station_entry.get("status") == "complete" and verify_completed_shard(shard_path, station_entry.get("checksum", "")):
        return station_entry

    started = time.time()
    try:
        rows, failures, metadata = build_rows_for_station(station_id, panel_row, daily, run_id, smoke=smoke)
        checksum = write_shard(shard_path, rows)
        failure_checksum = write_failure_shard(failure_path, failures)
        station_entry = {
            "status": "complete",
            "station_id": station_id,
            "expected_keys": metadata["expected_keys"],
            "completed_keys": len(rows),
            "model_status": metadata["models"],
            "checksum": checksum,
            "failure_ledger_checksum": failure_checksum,
            "failure_count": len(failures),
            "folds": metadata["folds"],
            "execution_timestamp": utc_now(),
            "elapsed_seconds": round(time.time() - started, 3),
            "protocol_hash": PROTOCOL_HASH,
            "run_id": run_id,
        }
    except Exception as exc:  # pragma: no cover - runtime-dependent
        failure = failure_row(station_id, "all", "station_execution_failure", exception=traceback.format_exc())
        checksum = write_failure_shard(failure_path, [failure])
        station_entry = {
            "status": "failed",
            "station_id": station_id,
            "expected_keys": 0,
            "completed_keys": 0,
            "model_status": {"xgboost_direct": "failed", "sarima": "failed"},
            "checksum": "",
            "failure_ledger_checksum": checksum,
            "failure_count": 1,
            "execution_timestamp": utc_now(),
            "elapsed_seconds": round(time.time() - started, 3),
            "protocol_hash": PROTOCOL_HASH,
            "run_id": run_id,
            "exception": str(exc),
        }
    progress.setdefault("stations", {})[station_id] = station_entry
    atomic_json(PROGRESS_PATH, progress)
    return station_entry


def collect_failure_ledger(failure_dir: Path, station_ids: list[str], output: Path) -> str:
    rows: list[dict[str, Any]] = []
    for station_id in station_ids:
        path = failure_dir / f"station_{station_id}.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as fh:
            rows.extend(dict(row) for row in csv.DictReader(fh))
    rows.sort(key=lambda r: (r["station_id"], r["failure_type"], r["fold"], r["origin_start"], r["horizon"]))
    return write_failure_shard(output, rows)


def run_smoke(panel: pd.DataFrame, daily: pd.DataFrame, run_id: str, source_panel_hash: str) -> dict[str, Any]:
    smoke_root = RESULTS / "smoke_test"
    shard_dir = smoke_root / "prediction_shards"
    failure_dir = smoke_root / "failure_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    progress_path = smoke_root / "execution_progress.json"
    progress = read_progress(progress_path)
    progress.update({"run_id": run_id, "protocol_hash": PROTOCOL_HASH, "source_panel_hash": source_panel_hash})
    smoke_ids = panel["station_id"].sort_values(kind="mergesort").head(3).tolist()
    entries = []
    for station_id in smoke_ids:
        panel_row = panel.loc[panel["station_id"] == station_id].iloc[0]
        entries.append(process_one_station(station_id, panel_row, daily, run_id, shard_dir, failure_dir, progress, smoke=True))
    atomic_json(progress_path, progress)
    failures = []
    for entry in entries:
        if entry.get("status") != "complete":
            failures.append(f"station {entry.get('station_id')} status {entry.get('status')}")
    schema_ok = True
    timestamps_ok = True
    persistence_ok = True
    condition_ok = True
    protocol_ok = True
    duplicate_count = 0
    rows_total = 0
    for station_id in smoke_ids:
        table = pq.read_table(shard_dir / f"station_{station_id}.parquet")
        df = table.to_pandas()
        rows_total += len(df)
        schema_ok = schema_ok and set(ROW_COLUMNS).issubset(df.columns)
        condition_ok = condition_ok and set(df["condition"]) == {CONDITION}
        protocol_ok = protocol_ok and set(df["protocol_hash"]) == {PROTOCOL_HASH}
        keys = ["station_id", "model", "origin", "target_time", "horizon"]
        duplicate_count += int(df.duplicated(keys).sum())
        timestamps_ok = timestamps_ok and bool((pd.to_datetime(df["target_time"]) - pd.to_datetime(df["origin"]) == pd.to_timedelta(df["horizon"], unit="D")).all())
        daily_station = daily[daily["station_id"] == station_id].copy()
        daily_station["date"] = pd.to_datetime(daily_station["date"])
        s = daily_station.set_index("date")["pm10_daily"].sort_index()
        expected_persistence = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).ffill()
        expected = [expected_persistence.loc[pd.Timestamp(o)] for o in df["origin"]]
        persistence_ok = persistence_ok and bool(np.allclose(df["y_persistence"].to_numpy(), np.asarray(expected, dtype=float), equal_nan=False))
    report = {
        "status": "PASS" if not failures and schema_ok and timestamps_ok and persistence_ok and condition_ok and protocol_ok and duplicate_count == 0 else "FAIL",
        "run_id": run_id,
        "protocol_hash": PROTOCOL_HASH,
        "station_ids": smoke_ids,
        "rows": rows_total,
        "schema_ok": schema_ok,
        "timestamps_ok": timestamps_ok,
        "persistence_alignment_ok": persistence_ok,
        "condition_ok": condition_ok,
        "protocol_hash_ok": protocol_ok,
        "duplicate_keys": duplicate_count,
        "checkpoint_files": [str(shard_dir / f"station_{sid}.parquet") for sid in smoke_ids],
        "failures": failures,
        "scientific_metrics_calculated": False,
        "created_at_utc": utc_now(),
    }
    atomic_json(RESULTS / "smoke_test_execution_report.json", report)
    return report


def run_full(panel: pd.DataFrame, daily: pd.DataFrame, run_id: str, source_panel_hash: str) -> dict[str, Any]:
    shard_dir = RESULTS / "prediction_shards"
    failure_dir = RESULTS / "failure_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    progress = read_progress(PROGRESS_PATH)
    progress.update({
        "run_id": run_id,
        "protocol_hash": PROTOCOL_HASH,
        "source_panel_hash": source_panel_hash,
        "condition": CONDITION,
        "station_count_expected": len(panel),
        "updated_at_utc": utc_now(),
    })
    atomic_json(PROGRESS_PATH, progress)
    for idx, panel_row in panel.iterrows():
        station_id = str(panel_row["station_id"])
        entry = process_one_station(station_id, panel_row, daily, run_id, shard_dir, failure_dir, progress)
        if (idx + 1) % 10 == 0 or entry.get("status") != "complete":
            print(json.dumps({"station_progress": idx + 1, "station_count": len(panel), "station_id": station_id, "status": entry.get("status"), "rows": entry.get("completed_keys")}, ensure_ascii=False), flush=True)

    failure_hash = collect_failure_ledger(failure_dir, panel["station_id"].tolist(), RESULTS / "model_execution_failure_ledger.csv")
    completed = [sid for sid in panel["station_id"] if progress.get("stations", {}).get(sid, {}).get("status") == "complete"]
    return {"progress": progress, "completed": completed, "failure_hash": failure_hash}


def consolidate_predictions(panel: pd.DataFrame, run_id: str) -> tuple[str, int, int]:
    output = RESULTS / "predictions_row_level_lags_only.parquet"
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    station_count = 0
    for station_id in panel["station_id"].tolist():
        path = RESULTS / "prediction_shards" / f"station_{station_id}.parquet"
        if not path.exists():
            raise RuntimeError(f"Missing prediction shard for {station_id}")
        table = pq.read_table(path)
        if set(table["condition"].to_pylist()) != {CONDITION} or set(table["protocol_hash"].to_pylist()) != {PROTOCOL_HASH}:
            raise RuntimeError(f"Frozen metadata mismatch in shard {path}")
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        elif table.schema != writer.schema:
            raise RuntimeError(f"Schema mismatch in shard {path}")
        writer.write_table(table)
        total_rows += table.num_rows
        station_count += 1
    if writer is None:
        raise RuntimeError("No prediction shards found")
    writer.close()
    os.replace(tmp, output)
    return sha256(output), total_rows, station_count


def expected_keys_for_station(panel_row: pd.Series, daily_station: pd.DataFrame) -> list[dict[str, Any]]:
    station_id = str(panel_row["station_id"])
    series, lag_df, _, fold_specs, _ = prepare_station(station_id, panel_row, daily_station)
    keys: list[dict[str, Any]] = []
    for spec in fold_specs:
        fold = int(spec["fold"])
        for origin in spec["origins"]:
            if not lag_df.loc[origin, LAG_COLUMNS].notna().all():
                continue
            for h in HORIZONS:
                target_time = origin + pd.Timedelta(days=h)
                if target_time in series.index and pd.notna(series.loc[target_time]):
                    keys.append({
                        "station_id": station_id,
                        "fold": fold,
                        "origin": origin.strftime("%Y-%m-%d"),
                        "target_time": target_time.strftime("%Y-%m-%d"),
                        "horizon": h,
                    })
    return keys


def common_support_audit(panel: pd.DataFrame, daily: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    any_integrity_failure = False
    total_common = total_expected = duplicate_total = 0
    for _, panel_row in panel.iterrows():
        station_id = str(panel_row["station_id"])
        expected = expected_keys_for_station(panel_row, daily[daily["station_id"] == station_id])
        expected_set = {(x["fold"], x["origin"], x["target_time"], x["horizon"]) for x in expected}
        shard = RESULTS / "prediction_shards" / f"station_{station_id}.parquet"
        df = pq.read_table(shard).to_pandas()
        key_cols = ["fold", "origin", "target_time", "horizon"]
        xgb = df[(df["model"] == "xgboost_direct") & np.isfinite(df["y_pred"])]
        sar = df[(df["model"] == "sarima") & np.isfinite(df["y_pred"])]
        def key_set(frame: pd.DataFrame) -> set[tuple[Any, ...]]:
            return set(map(tuple, frame[key_cols].itertuples(index=False, name=None)))
        xgb_set = key_set(xgb)
        sar_set = key_set(sar)
        xgb_dup = int(xgb.duplicated(key_cols).sum())
        sar_dup = int(sar.duplicated(key_cols).sum())
        common_set = expected_set & xgb_set & sar_set
        y_true_mismatch = 0
        y_persistence_mismatch = 0
        xgb_map = {tuple(r[key_cols]): (float(r["y_true"]), float(r["y_persistence"])) for _, r in xgb.iterrows()}
        sar_map = {tuple(r[key_cols]): (float(r["y_true"]), float(r["y_persistence"])) for _, r in sar.iterrows()}
        for key in common_set:
            if not np.isclose(xgb_map[key][0], sar_map[key][0], rtol=0, atol=0):
                y_true_mismatch += 1
            if not np.isclose(xgb_map[key][1], sar_map[key][1], rtol=0, atol=0):
                y_persistence_mismatch += 1
        removed = len(expected_set - common_set)
        reason_parts = []
        if removed:
            reason_parts.append("missing_model_prediction_keys")
        if xgb_dup or sar_dup:
            reason_parts.append("duplicate_keys")
        if y_true_mismatch:
            reason_parts.append("y_true_mismatch")
        if y_persistence_mismatch:
            reason_parts.append("y_persistence_mismatch")
        if y_true_mismatch or y_persistence_mismatch or xgb_dup or sar_dup:
            any_integrity_failure = True
        rows.append({
            "station_id": station_id,
            "horizon": "all",
            "expected_keys": len(expected_set),
            "xgboost_keys": len(xgb_set & expected_set),
            "sarima_keys": len(sar_set & expected_set),
            "persistence_keys": len(expected_set),
            "common_keys": len(common_set),
            "removed_keys": removed,
            "duplicate_xgboost_keys": xgb_dup,
            "duplicate_sarima_keys": sar_dup,
            "y_true_mismatch_keys": y_true_mismatch,
            "y_persistence_mismatch_keys": y_persistence_mismatch,
            "reason": ";".join(reason_parts) if reason_parts else "OK",
        })
        for h in HORIZONS:
            eh = {tuple((x["fold"], x["origin"], x["target_time"], x["horizon"])) for x in expected if x["horizon"] == h}
            xc = xgb_set & eh
            sc = sar_set & eh
            cc = eh & xgb_set & sar_set
            rows.append({
                "station_id": station_id,
                "horizon": h,
                "expected_keys": len(eh),
                "xgboost_keys": len(xc),
                "sarima_keys": len(sc),
                "persistence_keys": len(eh),
                "common_keys": len(cc),
                "removed_keys": len(eh - cc),
                "duplicate_xgboost_keys": int(xgb[xgb["horizon"] == h].duplicated(key_cols).sum()),
                "duplicate_sarima_keys": int(sar[sar["horizon"] == h].duplicated(key_cols).sum()),
                "y_true_mismatch_keys": 0,
                "y_persistence_mismatch_keys": 0,
                "reason": "OK" if len(eh - cc) == 0 else "missing_model_prediction_keys",
            })
        total_expected += len(expected_set)
        total_common += len(common_set)
        duplicate_total += xgb_dup + sar_dup
    fields = [
        "station_id", "horizon", "expected_keys", "xgboost_keys", "sarima_keys", "persistence_keys",
        "common_keys", "removed_keys", "duplicate_xgboost_keys", "duplicate_sarima_keys",
        "y_true_mismatch_keys", "y_persistence_mismatch_keys", "reason",
    ]
    atomic_csv(RESULTS / "common_support_audit.csv", fields, rows)
    audit_hash = sha256(RESULTS / "common_support_audit.csv")
    return audit_hash, {
        "status": "FAIL" if any_integrity_failure else "PASS",
        "total_expected_keys": total_expected,
        "total_common_keys": total_common,
        "duplicate_keys": duplicate_total,
        "rows": len(rows),
    }


def validate_prediction_file(panel: pd.DataFrame) -> dict[str, Any]:
    path = RESULTS / "predictions_row_level_lags_only.parquet"
    df = pq.read_table(path).to_pandas()
    required_ok = set(ROW_COLUMNS).issubset(df.columns)
    duplicate_keys = int(df.duplicated(["station_id", "model", "origin", "target_time", "horizon"]).sum())
    null_y_true = int(df["y_true"].isna().sum())
    null_y_pred = int(df["y_pred"].isna().sum())
    null_persistence = int(df["y_persistence"].isna().sum())
    elapsed_ok = bool((pd.to_datetime(df["target_time"]) - pd.to_datetime(df["origin"]) == pd.to_timedelta(df["horizon"], unit="D")).all())
    report = {
        "required_schema": required_ok,
        "station_count": int(df["station_id"].nunique()),
        "model_count": int(df["model"].nunique()),
        "models": sorted(df["model"].unique().tolist()),
        "horizons": sorted(df["horizon"].unique().tolist()),
        "condition_values": sorted(df["condition"].unique().tolist()),
        "duplicate_keys": duplicate_keys,
        "null_y_true": null_y_true,
        "null_y_pred": null_y_pred,
        "null_y_persistence": null_persistence,
        "elapsed_horizon_logic": elapsed_ok,
        "protocol_hash_values": sorted(df["protocol_hash"].unique().tolist()),
        "source_panel_hash_values": sorted(df["source_panel_hash"].unique().tolist()),
        "rows": len(df),
    }
    return report


def package_versions() -> dict[str, str]:
    import numpy
    import pandas
    import pyarrow
    import statsmodels
    import xgboost

    return {
        "python": sys.version,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "statsmodels": statsmodels.__version__,
        "xgboost": xgboost.__version__,
    }


def make_run_manifest(
    run_id: str,
    source_panel_hash: str,
    prediction_hash: str,
    common_hash: str,
    failure_hash: str,
    validation_report: dict[str, Any],
    common_report: dict[str, Any],
    progress: dict[str, Any],
) -> str:
    payload = {
        "manifest_type": "P3 lags-only forecasting run manifest",
        "project": "P3",
        "technical_namespace": "p4_v2",
        "condition": CONDITION,
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "protocol_hash": PROTOCOL_HASH,
        "panel_hash": source_panel_hash,
        "daily_data_hash": DAILY_HASH,
        "raw_data_manifest_hash": sha256(RAW_MANIFEST_PATH),
        "code_commit": git_head(),
        "package_versions": package_versions(),
        "platform": {"os": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "models": {"xgboost_direct": XGB_PARAMS, "sarima": {"order": SARIMA_ORDER, "seasonal_order": SARIMA_SEASONAL_ORDER, **SARIMA_KWARGS, "maxiter": SARIMA_MAXITER}},
        "features": {"lags": LAGS, "calendar": CALENDAR_COLUMNS, "condition": CONDITION},
        "event_threshold": {"quantile": EVENT_QUANTILE, "method": "numpy.quantile method=linear", "training_only": True},
        "station_count_expected": len(panel_ids_from_progress(progress)),
        "station_count_completed": len([v for v in progress.get("stations", {}).values() if v.get("status") == "complete"]),
        "horizons": list(HORIZONS),
        "prediction_file": {"path": str(RESULTS / "predictions_row_level_lags_only.parquet"), "sha256": prediction_hash, "rows": validation_report["rows"]},
        "common_support_audit": {"path": str(RESULTS / "common_support_audit.csv"), "sha256": common_hash, **common_report},
        "failure_ledger": {"path": str(RESULTS / "model_execution_failure_ledger.csv"), "sha256": failure_hash},
        "validation": validation_report,
        "scientific_metrics_calculated": False,
    }
    path = MANIFESTS / "p3_lags_only_run_manifest.json"
    atomic_json(path, payload)
    return sha256(path)


def panel_ids_from_progress(progress: dict[str, Any]) -> list[str]:
    return sorted(progress.get("stations", {}).keys())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.smoke == args.full:
        parser.error("choose exactly one of --smoke or --full")

    actual = verify_frozen_inputs()
    panel, daily, source_panel_hash = load_inputs()
    run_id = os.environ.get("P3_RUN_ID") or f"p3_lags_only_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(json.dumps({"run_id": run_id, "protocol_hash": PROTOCOL_HASH, "condition": CONDITION, "stations": len(panel), "mode": "smoke" if args.smoke else "full"}, ensure_ascii=False), flush=True)

    global PROGRESS_PATH
    PROGRESS_PATH = (RESULTS / "smoke_test" / "execution_progress.json") if args.smoke else (MANIFESTS / "execution_progress.json")
    if args.smoke:
        report = run_smoke(panel, daily, run_id, source_panel_hash)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["status"] == "PASS" else 2

    smoke_report_path = RESULTS / "smoke_test_execution_report.json"
    if not smoke_report_path.exists():
        raise RuntimeError("Smoke test report is missing; run --smoke before --full")
    smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    if smoke_report.get("status") != "PASS" or smoke_report.get("protocol_hash") != PROTOCOL_HASH:
        raise RuntimeError("Smoke test did not pass for the current protocol")
    full_result = run_full(panel, daily, run_id, source_panel_hash)
    if len(full_result["completed"]) != len(panel):
        print(json.dumps({"full_run": "PARTIAL", "stations_completed": len(full_result["completed"])}, ensure_ascii=False), flush=True)
        return 3
    prediction_hash, _, _ = consolidate_predictions(panel, run_id)
    validation_report = validate_prediction_file(panel)
    common_hash, common_report = common_support_audit(panel, daily)
    failure_hash = sha256(RESULTS / "model_execution_failure_ledger.csv")
    manifest_hash = make_run_manifest(run_id, source_panel_hash, prediction_hash, common_hash, failure_hash, validation_report, common_report, full_result["progress"])
    final_report = {
        "PROJECT": "P3",
        "CONDITION": CONDITION,
        "PROTOCOL_HASH_MATCH": "YES",
        "SMOKE_TEST": "PASS",
        "FULL_RUN": "COMPLETE",
        "STATIONS_EXPECTED": len(panel),
        "STATIONS_COMPLETED": len(full_result["completed"]),
        "PREDICTION_ROWS": validation_report["rows"],
        "COMMON_SUPPORT_KEYS": common_report["total_common_keys"],
        "DUPLICATE_KEYS": validation_report["duplicate_keys"],
        "SARIMA_FAILURES": sum(1 for r in csv.DictReader((RESULTS / "model_execution_failure_ledger.csv").open(encoding="utf-8")) if r["model"] == "sarima" and r["failure_type"] in {"fit_failure", "forecast_failure"}),
        "XGBOOST_FAILURES": sum(1 for r in csv.DictReader((RESULTS / "model_execution_failure_ledger.csv").open(encoding="utf-8")) if r["model"] == "xgboost_direct" and r["failure_type"] in {"fit_failure", "forecast_failure"}),
        "PERSISTENCE_FAILURES": sum(1 for r in csv.DictReader((RESULTS / "model_execution_failure_ledger.csv").open(encoding="utf-8")) if r["model"] == "persistence"),
        "COMMON_SUPPORT_EMPIRICAL": common_report["status"],
        "ROW_LEVEL_GRADE_A": "PASS" if validation_report["required_schema"] and validation_report["station_count"] == len(panel) and validation_report["model_count"] == 2 and validation_report["horizons"] == list(HORIZONS) and validation_report["condition_values"] == [CONDITION] and validation_report["duplicate_keys"] == 0 and validation_report["null_y_true"] == 0 and validation_report["null_y_pred"] == 0 and validation_report["null_y_persistence"] == 0 and validation_report["elapsed_horizon_logic"] and common_report["status"] == "PASS" else "FAIL",
        "SCIENTIFIC_METRICS_AUTHORIZED": "NO",
        "prediction_file_hash": prediction_hash,
        "run_manifest_hash": manifest_hash,
    }
    atomic_json(RESULTS / "p3_lags_only_execution_report.json", final_report)
    print(json.dumps(final_report, ensure_ascii=False), flush=True)
    return 0 if final_report["ROW_LEVEL_GRADE_A"] == "PASS" else 4


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    raise SystemExit(main())
