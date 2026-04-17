"""Lag feature construction for daily tabular forecasting models."""

import pandas as pd


def add_lag_features(df: pd.DataFrame, target_column: str, lags: list[int]) -> pd.DataFrame:
    """Add lagged versions of the target column."""
    enriched = df.copy()
    for lag in lags:
        enriched[f"{target_column}_lag_{lag}"] = enriched[target_column].shift(lag)
    return enriched
