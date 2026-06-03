"""Minimal wrappers for tabular forecasting models."""

from __future__ import annotations

from xgboost import XGBRegressor


def build_default_xgboost(random_state: int = 42) -> XGBRegressor:
    """Return a conservative XGBoost regressor for lag-based benchmarks."""
    return XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
    )
