"""STL plus Ridge direct forecaster for daily PM10.

The model decomposes the train-only daily series with STL, fits direct Ridge
models to the residual component, and adds back a seasonal-naive raw-series
component at prediction time. It is intended as a leakage-safe diagnostic
contrast: explicit weekly seasonality is handled before residual modelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


class STLRidgeForecaster:
    """Seasonal-Trend decomposition plus Ridge direct residual forecaster."""

    def __init__(self, season_length: int = 7, ridge_alpha: float = 1.0, n_lags: int = 7):
        if season_length < 2:
            raise ValueError("season_length must be >= 2")
        if n_lags < 1:
            raise ValueError("n_lags must be >= 1")
        self.season_length = int(season_length)
        self.ridge_alpha = float(ridge_alpha)
        self.n_lags = int(n_lags)
        self._models: dict[int, object] = {}
        self._residual: pd.Series | None = None
        self._raw: pd.Series | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, train_df: pd.DataFrame) -> "STLRidgeForecaster":
        """Fit one direct residual Ridge model per feasible horizon.

        Parameters
        ----------
        train_df
            Chronological DataFrame with `date` and `pm10` columns. Only rows
            available up to the forecast origin should be supplied by callers.
        """
        missing = {"date", "pm10"} - set(train_df.columns)
        if missing:
            raise ValueError(f"train_df missing required columns: {sorted(missing)}")

        frame = train_df[["date", "pm10"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["pm10"] = pd.to_numeric(frame["pm10"], errors="coerce")
        frame = frame.sort_values("date").dropna(subset=["pm10"])
        if len(frame) < 2 * self.season_length:
            raise ValueError(f"STL requires at least {2 * self.season_length} observations.")

        y = pd.Series(frame["pm10"].to_numpy(dtype=float), index=frame["date"])
        stl = STL(y, period=self.season_length, robust=True).fit()
        residual = y - stl.seasonal

        residual_frame = pd.DataFrame({"residual": residual})
        for lag in range(self.n_lags):
            residual_frame[f"lag_{lag}"] = residual_frame["residual"].shift(lag)

        feature_cols = [f"lag_{lag}" for lag in range(self.n_lags)]
        self._models = {}
        max_horizon = self.season_length
        for horizon in range(1, max_horizon + 1):
            supervised = residual_frame.copy()
            supervised["target"] = supervised["residual"].shift(-horizon)
            supervised = supervised.dropna(subset=feature_cols + ["target"])
            if supervised.empty:
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=self.ridge_alpha))
            model.fit(supervised[feature_cols], supervised["target"])
            self._models[horizon] = model

        if not self._models:
            raise ValueError("No STL+Ridge horizon models were fitted.")
        self._residual = residual
        self._raw = y
        self._last_date = y.index[-1]
        return self

    def predict_horizon(self, h: int) -> float:
        """Predict PM10 at horizon h from the last fitted train origin."""
        if self._residual is None or self._raw is None or self._last_date is None:
            raise RuntimeError("Call fit() before predict_horizon().")
        if h not in self._models:
            raise ValueError(f"No fitted STL+Ridge model for horizon {h}.")

        lag_values = [self._residual.iloc[-1 - lag] for lag in range(self.n_lags)]
        if any(pd.isna(lag_values)):
            raise ValueError("Residual lag vector contains missing values.")
        feature_cols = [f"lag_{lag}" for lag in range(self.n_lags)]
        x_pred = pd.DataFrame([lag_values], columns=feature_cols)
        residual_pred = float(self._models[h].predict(x_pred)[0])

        seasonal_source_date = self._last_date - pd.Timedelta(days=self.season_length - h)
        if seasonal_source_date not in self._raw.index or pd.isna(self._raw.loc[seasonal_source_date]):
            raise ValueError(f"Missing seasonal source observation for {seasonal_source_date.date()}.")
        seasonal_pred = float(self._raw.loc[seasonal_source_date])
        return residual_pred + seasonal_pred
