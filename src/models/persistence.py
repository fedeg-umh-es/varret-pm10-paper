"""Persistence baseline for daily multi-step forecasting in P33."""

from __future__ import annotations

import numpy as np


class PersistenceModel:
    """Repeat the last observed value across the requested forecast horizon."""

    def fit(self, history: np.ndarray) -> "PersistenceModel":
        """Store the observed history for interface completeness."""
        values = np.asarray(history, dtype=float)
        if values.size == 0:
            raise ValueError("history must contain at least one observation.")
        self.last_value_ = float(values[-1])
        return self

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Return a multi-step persistence forecast."""
        values = np.asarray(history, dtype=float)
        if values.size == 0:
            raise ValueError("history must contain at least one observation.")
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")
        return np.repeat(float(values[-1]), horizon).astype(float)
