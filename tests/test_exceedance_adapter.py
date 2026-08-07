"""Unit tests for exceedance adapter, schema validation, and rank reversal engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.exceedance_adapter import (
    check_case_alignment,
    check_duplicates,
    classify_rank_reversal,
    compute_contingency_metrics,
    compute_kendall_taub,
    normalize_predictions_schema,
)


def test_schema_adapter_canonical_columns() -> None:
    df = pd.DataFrame({
        "origin_time": ["2023-01-01 00:00:00"],
        "target_time": ["2023-01-01 01:00:00"],
        "horizon": [1],
        "fold": [0],
        "model": ["lightgbm"],
        "y_true": [15.0],
        "y_pred": [14.5],
        "y_persistence": [16.0],
    })
    out, meta = normalize_predictions_schema(df)
    assert meta["station_status"] == "MISSING_FROM_SOURCE"
    assert meta["alias_used"] is None
    assert not meta["fallback_used"]
    assert out["horizon"].iloc[0] == 1


def test_schema_adapter_aliases_origin_target_date() -> None:
    df = pd.DataFrame({
        "origin_date": ["2023-01-01 00:00:00"],
        "target_date": ["2023-01-01 06:00:00"],
        "horizon": [6],
        "fold": [0],
        "model": ["sarima"],
        "y_true": [20.0],
        "y_pred": [18.0],
        "y_persistence": [22.0],
    })
    out, meta = normalize_predictions_schema(df)
    assert "origin_time" in out.columns
    assert "target_time" in out.columns
    assert "origin_date" in meta["alias_used"]


def test_schema_adapter_fallback_date() -> None:
    df = pd.DataFrame({
        "origin_time": ["2023-01-01 00:00:00"],
        "date": ["2023-01-02 00:00:00"],
        "horizon": [24],
        "fold": [0],
        "model": ["lightgbm"],
        "y_true": [30.0],
        "y_pred": [28.0],
        "y_persistence": [25.0],
    })
    out, meta = normalize_predictions_schema(df)
    assert meta["fallback_used"]
    assert out["target_time"].iloc[0] == pd.Timestamp("2023-01-02 00:00:00")


def test_schema_adapter_temporal_coherence_validation() -> None:
    # Invalid horizon mismatch (horizon=24 but delta is 1 hour)
    df = pd.DataFrame({
        "origin_time": ["2023-01-01 00:00:00"],
        "target_time": ["2023-01-01 01:00:00"],
        "horizon": [24],
        "fold": [0],
        "model": ["lightgbm"],
        "y_true": [15.0],
        "y_pred": [14.5],
        "y_persistence": [16.0],
    })
    _, meta = normalize_predictions_schema(df)
    assert len(meta["validation_errors"]) > 0


def test_check_duplicates() -> None:
    df = pd.DataFrame({
        "model": ["m1", "m1"],
        "fold": [0, 0],
        "origin_time": pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 00:00:00"]),
        "target_time": pd.to_datetime(["2023-01-01 01:00:00", "2023-01-01 01:00:00"]),
        "horizon": [1, 1],
        "y_true": [10.0, 10.0],
        "y_pred": [11.0, 12.0],
        "y_persistence": [10.0, 10.0],
    })
    dups = check_duplicates(df)
    assert len(dups) == 2


def test_case_alignment_aligned_and_misaligned() -> None:
    # Exact aligned cases
    df_aligned = pd.DataFrame({
        "fold": [0, 0, 0, 0],
        "origin_time": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"]),
        "target_time": pd.to_datetime(["2023-01-02", "2023-01-02", "2023-01-02", "2023-01-02"]),
        "horizon": [24, 24, 24, 24],
        "model": ["lightgbm", "lightgbm", "sarima", "sarima"],
        "y_true": [10.0, 20.0, 10.0, 20.0],
    })
    aligned, report = check_case_alignment(df_aligned)
    assert aligned

    # Misaligned cases
    df_misaligned = pd.DataFrame({
        "fold": [0, 0, 0],
        "origin_time": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-01"]),
        "target_time": pd.to_datetime(["2023-01-02", "2023-01-02", "2023-01-03"]),
        "horizon": [24, 24, 48],
        "model": ["lightgbm", "sarima", "sarima"],
        "y_true": [10.0, 10.0, 30.0],
    })
    aligned, report = check_case_alignment(df_misaligned)
    assert not aligned


def test_contingency_metrics_calculation() -> None:
    y_true = np.array([10.0, 60.0, 70.0, 20.0])
    y_pred = np.array([15.0, 55.0, 40.0, 65.0])
    thresh = 50.0

    res = compute_contingency_metrics(y_true, y_pred, thresh)
    # TP: y_true > 50 & y_pred > 50 -> (60, 55) -> 1
    # FP: y_true <= 50 & y_pred > 50 -> (20, 65) -> 1
    # FN: y_true > 50 & y_pred <= 50 -> (70, 40) -> 1
    # TN: y_true <= 50 & y_pred <= 50 -> (10, 15) -> 1
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["fn"] == 1
    assert res["tn"] == 1
    assert res["pod"] == 0.5
    assert res["far"] == 0.5
    assert res["csi"] == 1 / 3
    # Intensity error for events (60, 70): mean((55-60) + (40-70)) / 2 = (-5 -30)/2 = -17.5
    assert res["exceedance_intensity_error"] == -17.5


def test_kendall_taub_with_ties() -> None:
    x = np.array([1.0, 2.0, 2.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 3.0, 5.0])
    tau = compute_kendall_taub(x, y)
    assert not np.isnan(tau)
    assert tau > 0.5


def test_rank_reversal_classifications() -> None:
    # YES: continuous skill prefers M1 (+0.1), CSI prefers M2 (-0.2)
    assert classify_rank_reversal(0.1, -0.2, 0.5) == "YES"
    # NO: both prefer M1 (+0.1, +0.2)
    assert classify_rank_reversal(0.1, 0.2, 0.9) == "NO"
    # TRADE_OFF_ONLY: skill prefers M1 (+0.1), CSI is tie (0.0)
    assert classify_rank_reversal(0.1, 0.0, 0.5) == "TRADE_OFF_ONLY"
    # NOT_EVALUABLE: missing metric
    assert classify_rank_reversal(np.nan, 0.2, 0.5) == "NOT_EVALUABLE"
