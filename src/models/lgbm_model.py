"""
LightGBM Multi-Horizon Model
=============================
Trains separate LightGBM model per horizon (h=1,6,24,48).
No multi-output; explicit leakage prevention via separate models.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict, Any


class LGBMMultiHorizon:
    """
    Wrapper for LightGBM with explicit per-horizon training.
    Trains independent models to prevent information leakage across horizons.
    """
    
    def __init__(self, horizons: list, lgb_params: Dict[str, Any] = None):
        """
        Args:
            horizons: list of horizons [1, 6, 24, 48]
            lgb_params: LightGBM hyperparameters dict
        """
        self.horizons = horizons
        self.lgb_params = lgb_params or {}
        self.models: Dict[int, lgb.LGBMRegressor] = {}
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "LGBMMultiHorizon":
        """
        Train independent model per horizon.
        
        Args:
            X: features (n_samples, n_features)
            y: targets (n_samples, n_horizons)
        
        Returns:
            self
        """
        assert len(X) == len(y), "X and y must have same length"
        
        for idx, h in enumerate(self.horizons):
            y_h = y.iloc[:, idx] if isinstance(y, pd.DataFrame) else y[:, idx]
            
            model = lgb.LGBMRegressor(
                n_estimators=self.lgb_params.get("n_estimators", 200),
                max_depth=self.lgb_params.get("max_depth", 8),
                learning_rate=self.lgb_params.get("learning_rate", 0.05),
                num_leaves=self.lgb_params.get("num_leaves", 31),
                subsample=self.lgb_params.get("subsample", 0.9),
                colsample_bytree=self.lgb_params.get("colsample_bytree", 0.9),
                random_state=self.lgb_params.get("random_state", 42),
                verbose=-1
            )
            model.fit(X, y_h)
            self.models[h] = model
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict all horizons.
        
        Args:
            X: features (n_samples, n_features)
        
        Returns:
            (n_samples, n_horizons) predictions
        """
        assert self.is_fitted, "Model not fitted"
        
        predictions = []
        for h in self.horizons:
            pred_h = self.models[h].predict(X)
            predictions.append(pred_h)
        
        return np.column_stack(predictions)
    
    def predict_single_horizon(self, X: pd.DataFrame, h: int) -> np.ndarray:
        """
        Predict single horizon.
        
        Args:
            X: features
            h: horizon value (from self.horizons)
        
        Returns:
            (n_samples,) predictions
        """
        assert self.is_fitted, "Model not fitted"
        assert h in self.models, f"Horizon {h} not in fitted models"
        return self.models[h].predict(X)
    
    def save(self, dirpath: Path) -> None:
        """Save all models to directory."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        for h, model in self.models.items():
            model.booster_.save_model(str(dirpath / f"lgbm_h{h}.txt"))
    
    def load(self, dirpath: Path) -> "LGBMMultiHorizon":
        """Load all models from directory."""
        dirpath = Path(dirpath)
        
        for h in self.horizons:
            model_path = dirpath / f"lgbm_h{h}.txt"
            if model_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                # Wrap in LGBMRegressor-like interface
                lgbm_model = lgb.LGBMRegressor(random_state=42)
                lgbm_model.booster_ = model
                self.models[h] = lgbm_model
        
        self.is_fitted = True
        return self
