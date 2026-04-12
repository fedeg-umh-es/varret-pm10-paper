"""
Persistence Model: Baseline forecaster
======================================
Predicts: y_hat[t+h] = y[t] for all horizons h

This is the explicit baseline for computing skill metrics.
"""

import numpy as np
from typing import Union


class PersistenceModel:
    """
    Persistence baseline: repeats last observed value for all horizons.
    
    No parameters to fit. Serves as RMSE reference for skill computation.
    
    skill = 1 - rmse_model / rmse_persistence
    """
    
    def __init__(self, horizons: list = None):
        """
        Args:
            horizons: list of horizons (only for interface consistency)
        """
        self.horizons = horizons or []
        self.is_fitted = True  # No fitting needed
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        No-op. Persistence requires no fitting.
        X and y are accepted for interface consistency with other models.
        
        Args:
            X: features (ignored)
            y: targets (ignored)
        
        Returns:
            self
        """
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using persistence.
        
        Args:
            X: (n_samples, n_features) or (n_samples, context_len, 1) for LSTM format
               Last column/row is y[t-1]
        
        Returns:
            (n_samples, n_horizons) where each row repeats the last value
        """
        if len(X.shape) == 3:
            # LSTM format: (n_samples, context_len, 1)
            last_val = X[:, -1, 0]  # (n_samples,)
        else:
            # LightGBM format: (n_samples, n_features)
            last_val = X[:, -1]  # Last feature is y[t-1]
        
        # Repeat for all horizons
        n_horizons = len(self.horizons) if self.horizons else 48
        predictions = np.tile(last_val[:, np.newaxis], (1, n_horizons))
        
        return predictions
    
    def predict_single_horizon(self, X: np.ndarray, horizon_idx: int) -> np.ndarray:
        """
        Predict single horizon.
        
        Args:
            X: features
            horizon_idx: which horizon (0-indexed into self.horizons)
        
        Returns:
            (n_samples,) predictions
        """
        if len(X.shape) == 3:
            return X[:, -1, 0]
        else:
            return X[:, -1]
