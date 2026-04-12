"""
SARIMA forecaster
=================
Statistical baseline for Paper A.

Wraps statsmodels SARIMAX with a fixed (p,d,q)(P,D,Q,s) order.
Trained once per rolling-origin fold on the raw (normalized) series.
Forecast is purely recursive — never touches test observations.
"""

import numpy as np
from typing import List


class SarimaForecaster:
    """Fixed-order SARIMA forecaster.

    Parameters
    ----------
    order : tuple (p, d, q)
    seasonal_order : tuple (P, D, Q, s)
    trend : str or None  — passed to SARIMAX
    enforce_stationarity : bool
    enforce_invertibility : bool
    """

    def __init__(
        self,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        trend=None,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._fitted = None

    def fit(self, y_train: np.ndarray) -> "SarimaForecaster":
        """Fit SARIMA on the training series.

        Parameters
        ----------
        y_train : 1-D array of floats (normalized PM10 values, chronological)
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        self._fitted = model.fit(disp=False)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        """Recursive multi-step forecast from the end of the training series.

        Parameters
        ----------
        steps : int  — maximum horizon needed

        Returns
        -------
        1-D array of length `steps`; index 0 = h=1, index h-1 = h=h.
        """
        if self._fitted is None:
            raise RuntimeError("Call fit() before forecast().")
        result = self._fitted.get_forecast(steps=steps)
        pm = result.predicted_mean
        return pm.values if hasattr(pm, 'values') else np.array(pm)

    def predict_horizons(self, horizons: List[int]) -> np.ndarray:
        """Return forecasts only for the requested horizons.

        Parameters
        ----------
        horizons : list of int, e.g. [1, 6, 24, 48]

        Returns
        -------
        1-D array aligned to `horizons`
        """
        max_h = max(horizons)
        all_forecasts = self.forecast(steps=max_h)
        # all_forecasts[0] is h=1, all_forecasts[h-1] is h=h
        return np.array([all_forecasts[h - 1] for h in horizons])
