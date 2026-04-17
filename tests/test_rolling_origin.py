"""Conceptual tests for leakage-free rolling-origin generation."""

import pandas as pd

from src.splits.rolling_origin import generate_rolling_origin_folds


def test_rolling_origin_respects_temporal_order() -> None:
    """Each test index must occur strictly after the training window."""
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20, freq="D"), "y": range(20)})
    folds = generate_rolling_origin_folds(df, min_train_size=5, max_horizon=3)
    assert folds
    assert all(fold.test_index > max(fold.train_indices) for fold in folds)
