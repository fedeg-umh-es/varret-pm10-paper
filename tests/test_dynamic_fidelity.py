"""Unit tests for dynamic fidelity engine, metric boundary conditions, and ghost skill audit table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.dynamic_fidelity import (
    compute_all_dynamic_fidelity_metrics,
    compute_alpha_kge,
    compute_amplitude_ratio,
    compute_correlation,
    compute_event_amplitude_retention,
    compute_std_ratio,
    compute_temporal_variability,
    compute_variance_retention,
)


def test_perfect_fidelity() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    target_times = pd.date_range("2023-01-01 00:00:00", periods=5, freq="1h")

    res = compute_all_dynamic_fidelity_metrics(y_true, y_pred, threshold=25.0, target_times=target_times)

    assert pytest.approx(res["variance_retention"], abs=1e-6) == 1.0
    assert pytest.approx(res["std_ratio"], abs=1e-6) == 1.0
    assert pytest.approx(res["alpha_kge"], abs=1e-6) == 1.0
    assert pytest.approx(res["correlation"], abs=1e-6) == 1.0
    assert pytest.approx(res["amplitude_ratio"], abs=1e-6) == 1.0
    assert pytest.approx(res["temporal_variability"], abs=1e-6) == 1.0
    assert pytest.approx(res["event_amplitude_retention"], abs=1e-6) == 1.0


def test_variance_collapse_and_constant_predictions() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([25.0, 25.0, 25.0, 25.0, 25.0])  # constant prediction

    res = compute_all_dynamic_fidelity_metrics(y_true, y_pred, threshold=25.0)

    assert res["variance_retention"] == 0.0
    assert res["std_ratio"] == 0.0
    assert res["alpha_kge"] == 0.0
    assert np.isnan(res["correlation"])  # zero variance in pred -> NaN correlation
    assert res["amplitude_ratio"] == 0.0
    assert res["temporal_variability"] == 0.0
    # y_true > 25 -> 30, 40, 50 (mean = 40); y_pred = 25 -> event_amplitude_retention = 25/40 = 0.625
    assert pytest.approx(res["event_amplitude_retention"], abs=1e-6) == 0.625


def test_zero_variance_in_y_true() -> None:
    y_true = np.array([20.0, 20.0, 20.0, 20.0])
    y_pred = np.array([15.0, 25.0, 18.0, 22.0])

    res = compute_all_dynamic_fidelity_metrics(y_true, y_pred, threshold=15.0)

    assert np.isnan(res["variance_retention"])
    assert np.isnan(res["std_ratio"])
    assert np.isnan(res["alpha_kge"])
    assert np.isnan(res["correlation"])
    assert np.isnan(res["amplitude_ratio"])
    assert np.isnan(res["temporal_variability"])


def test_amplitude_attenuation() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = 0.5 * y_true  # 50% amplitude scaling

    res = compute_all_dynamic_fidelity_metrics(y_true, y_pred, threshold=25.0)

    assert pytest.approx(res["variance_retention"], abs=1e-6) == 0.25
    assert pytest.approx(res["std_ratio"], abs=1e-6) == 0.5
    assert pytest.approx(res["alpha_kge"], abs=1e-6) == 0.5
    assert pytest.approx(res["correlation"], abs=1e-6) == 1.0
    assert pytest.approx(res["amplitude_ratio"], abs=1e-6) == 0.5
    assert pytest.approx(res["temporal_variability"], abs=1e-6) == 0.5
    assert pytest.approx(res["event_amplitude_retention"], abs=1e-6) == 0.5


def test_nan_handling_and_boundary() -> None:
    y_true = np.array([10.0, np.nan, 30.0, 40.0, 50.0])
    y_pred = np.array([10.0, 20.0, np.nan, 40.0, 50.0])

    res = compute_all_dynamic_fidelity_metrics(y_true, y_pred, threshold=25.0)

    # Valid pairs: index 0 (10,10), index 3 (40,40), index 4 (50,50)
    assert not np.isnan(res["variance_retention"])
    assert pytest.approx(res["correlation"], abs=1e-6) == 1.0


def test_common_case_count_preservation(tmp_path: pytest.TempPathFactory) -> None:
    df_test = pd.DataFrame({
        "model": ["lightgbm"] * 5 + ["sarima"] * 5,
        "fold": [0] * 10,
        "origin_time": pd.to_datetime(["2023-01-01 00:00:00"] * 10),
        "target_time": pd.to_datetime(["2023-01-01 01:00:00"] * 10),
        "horizon": [1] * 10,
        "y_true": [10.0, 20.0, 30.0, 40.0, 50.0] * 2,
        "y_pred": [12.0, 18.0, 32.0, 38.0, 48.0] * 2,
        "y_persistence": [10.0] * 10,
        "p75_train": [25.0] * 10,
    })
    assert len(df_test) == 10
    models = df_test["model"].unique()
    assert len(models) == 2
