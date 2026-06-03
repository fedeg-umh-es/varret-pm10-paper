"""Train-only preprocessing helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Iterable[int],
) -> pd.DataFrame:
    """Create lagged features from historical observations only."""
    out = df.copy()
    for lag in sorted(set(lags)):
        if lag <= 0:
            raise ValueError("Lag values must be positive integers.")
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def drop_incomplete_rows(df: pd.DataFrame, required_cols: Iterable[str]) -> pd.DataFrame:
    """Drop rows with missing values in required columns."""
    return df.dropna(subset=list(required_cols)).reset_index(drop=True)
