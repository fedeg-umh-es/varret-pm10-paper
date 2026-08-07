"""Unit tests for fold stability evaluator and contiguous step boundary isolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.fold_stability import compute_fold_level_metrics, summarize_sarima_fold_stability


def test_fold_level_metrics_calculation() -> None:
    # 2 folds, 2 models, 1 horizon
    df = pd.DataFrame({
        "model": ["sarima"] * 10 + ["lightgbm"] * 10,
        "horizon": [48] * 20,
        "fold": ([0] * 5 + [1] * 5) * 2,
        "origin_time": pd.to_datetime(["2023-01-01 00:00:00"] * 20),
        "target_time": pd.concat([
            pd.Series(pd.date_range("2023-01-03 00:00:00", periods=5, freq="1h")),
            pd.Series(pd.date_range("2023-02-01 00:00:00", periods=5, freq="1h")),
            pd.Series(pd.date_range("2023-01-03 00:00:00", periods=5, freq="1h")),
            pd.Series(pd.date_range("2023-02-01 00:00:00", periods=5, freq="1h")),
        ]).reset_index(drop=True),
        "y_true": [10.0, 20.0, 30.0, 40.0, 50.0] * 4,
        "y_pred": [25.0, 25.0, 25.0, 25.0, 25.0] * 2 + [11.0, 19.0, 31.0, 39.0, 51.0] * 2,  # SARIMA constant, LightGBM good
        "y_persistence": [10.0, 10.0, 10.0, 10.0, 10.0] * 4,
        "p75_train": [25.0] * 20,
    })

    res = compute_fold_level_metrics(df)
    assert len(res) == 4  # 2 models x 2 folds

    sarima_f0 = res[(res["model"] == "sarima") & (res["fold"] == 0)].iloc[0]
    assert sarima_f0["variance_retention"] == 0.0
    assert sarima_f0["POD"] == 0.0

    lgbm_f0 = res[(res["model"] == "lightgbm") & (res["fold"] == 0)].iloc[0]
    assert lgbm_f0["variance_retention"] > 0.9
    assert lgbm_f0["POD"] > 0.5


def test_fold_boundary_isolation_temporal_variability() -> None:
    # Verify that temporal variability does NOT difference across fold boundary gap
    df = pd.DataFrame({
        "model": ["sarima"] * 4,
        "horizon": [48] * 4,
        "fold": [0, 0, 1, 1],
        "origin_time": pd.to_datetime(["2023-01-01 00:00:00"] * 4),
        "target_time": pd.to_datetime([
            "2023-01-03 00:00:00", "2023-01-03 01:00:00",  # Fold 0 contiguous
            "2023-02-01 00:00:00", "2023-02-01 01:00:00",  # Fold 1 contiguous (gap between fold 0 and 1)
        ]),
        "y_true": [10.0, 20.0, 100.0, 110.0],
        "y_pred": [10.0, 20.0, 100.0, 110.0],
        "y_persistence": [10.0] * 4,
        "p75_train": [25.0] * 4,
    })

    res = compute_fold_level_metrics(df)
    # Fold 0: diff_true = |20-10| = 10; diff_pred = |20-10| = 10 -> ratio = 1.0
    # Fold 1: diff_true = |110-100| = 10; diff_pred = |110-100| = 10 -> ratio = 1.0
    for _, row in res.iterrows():
        assert pytest.approx(row["temporal_variability"], abs=1e-6) == 1.0


def test_sarima_summary_aggregation() -> None:
    df_fold = pd.DataFrame({
        "model": ["sarima"] * 5,
        "horizon": [48] * 5,
        "fold": [0, 1, 2, 3, 4],
        "rmse_skill": [0.1, 0.12, 0.15, 0.08, 0.14],
        "variance_retention": [0.003, 0.004, 0.002, 0.005, 0.003],
        "correlation": [-0.1, -0.05, -0.15, 0.0, -0.08],
        "amplitude_ratio": [0.06, 0.07, 0.05, 0.06, 0.07],
        "temporal_variability": [0.02, 0.03, 0.02, 0.02, 0.03],
        "POD": [0.0] * 5,
        "CSI": [0.0] * 5,
        "positive_skill": [True] * 5,
        "concordant_degradation": [True] * 5,
    })

    summary = summarize_sarima_fold_stability(df_fold)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["folds_with_positive_skill"] == 5
    assert row["stability_pattern"] == "GHOST_PATTERN_REPLICATED_5_OF_5_FOLDS"
    assert bool(row["complete_event_failure_all_folds"]) is True
    assert bool(row["degraded_event_representation_all_folds"]) is True
