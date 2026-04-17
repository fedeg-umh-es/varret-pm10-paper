"""Minimal linear autoregressive model placeholder for P33."""

from __future__ import annotations

import numpy as np


class LinearARModel:
    """Linear autoregressive model stub with a restrained interface."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearARModel":
        """Store training metadata for later implementation."""
        self.n_features_ = X.shape[1] if X.ndim > 1 else 1
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a zero vector placeholder until model fitting is implemented."""
        if not getattr(self, "is_fitted_", False):
            raise ValueError("Model must be fitted before predicting.")
        return np.zeros(len(X), dtype=float)
