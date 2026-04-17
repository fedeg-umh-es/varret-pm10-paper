"""Seasonal persistence baseline for daily forecasting."""

from __future__ import annotations

import numpy as np


class SeasonalPersistenceModel:
    """Repeat the last observed seasonal analogue."""

    def __init__(self, season_length: int = 7) -> None:
        self.season_length = season_length

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Predict each step using the value observed one season earlier when available."""
        if history.size == 0:
            raise ValueError("history must contain at least one observation.")
        predictions = []
        for step in range(horizon):
            index = -(self.season_length - step)
            if abs(index) <= history.size:
                predictions.append(float(history[index]))
            else:
                predictions.append(float(history[-1]))
        return np.asarray(predictions, dtype=float)
