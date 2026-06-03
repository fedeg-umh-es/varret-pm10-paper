"""Baseline forecasting models."""

from __future__ import annotations

import numpy as np


def persistence_forecast(last_observation: float, horizon: int) -> np.ndarray:
    """Repeat the latest observed value across the forecast horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    return np.repeat(float(last_observation), horizon)


def seasonal_persistence_forecast(history: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
    """Project the latest seasonal pattern forward."""
    hist = np.asarray(history, dtype=float)
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if season_length <= 0:
        raise ValueError("season_length must be positive.")
    if hist.size < season_length:
        raise ValueError("history is shorter than season_length.")
    template = hist[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    return np.tile(template, reps)[:horizon]


def seasonal_persistence_7_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Weekly seasonal persistence for daily series."""
    return seasonal_persistence_forecast(
        history=history,
        horizon=horizon,
        season_length=7,
    )
