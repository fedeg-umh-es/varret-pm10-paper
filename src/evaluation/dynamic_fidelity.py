"""Canonical Dynamic Fidelity Metrics Engine for PM10 Forecasting Evaluation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

EPS = 1e-12


def compute_variance_retention(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes predicted-to-observed variance ratio: Var(y_pred) / Var(y_true).
    Uses sample variance with ddof=1.
    If Var(y_true) <= 1e-12, returns NaN. If Var(y_pred) <= 1e-12, returns 0.0.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    if len(yt) < 2:
        return np.nan

    var_true = float(np.var(yt, ddof=1))
    var_pred = float(np.var(yp, ddof=1))

    if var_true <= EPS:
        return np.nan
    if var_pred <= EPS:
        return 0.0
    return float(var_pred / var_true)


def compute_std_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes predicted-to-observed standard deviation ratio: SD(y_pred) / SD(y_true).
    Uses sample standard deviation with ddof=1.
    If SD(y_true) <= 1e-12, returns NaN. If SD(y_pred) <= 1e-12, returns 0.0.
    """
    vr = compute_variance_retention(y_true, y_pred)
    if np.isnan(vr):
        return np.nan
    return float(np.sqrt(vr))


def compute_alpha_kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Kling-Gupta Efficiency variability ratio component alpha = SD(y_pred) / SD(y_true).
    Identical to std_ratio by definition (Gupta et al., 2009).
    """
    return compute_std_ratio(y_true, y_pred)


def compute_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Pearson correlation coefficient r between y_pred and y_true.
    If SD(y_true) <= 1e-12 or SD(y_pred) <= 1e-12, returns NaN.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    if len(yt) < 2:
        return np.nan

    sd_true = float(np.std(yt, ddof=1))
    sd_pred = float(np.std(yp, ddof=1))

    if sd_true <= EPS or sd_pred <= EPS:
        return np.nan

    r, _ = pearsonr(yp, yt)
    return float(r) if not np.isnan(r) else np.nan


def compute_amplitude_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes inter-quantile 95th-5th percentile amplitude ratio:
    IQR_95_5(y_pred) / IQR_95_5(y_true).
    Quantifies robust dynamic spread without extreme outlier vulnerability.
    If IQR_95_5(y_true) <= 1e-12, returns NaN.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    if len(yt) < 2:
        return np.nan

    iqr_true = float(np.percentile(yt, 95) - np.percentile(yt, 5))
    iqr_pred = float(np.percentile(yp, 95) - np.percentile(yp, 5))

    if iqr_true <= EPS:
        return np.nan
    if iqr_pred <= EPS:
        return 0.0
    return float(iqr_pred / iqr_true)


def compute_temporal_variability(
    y_true: np.ndarray, y_pred: np.ndarray, target_times: np.ndarray | None = None
) -> float:
    """
    Computes mean absolute step-to-step first-difference ratio:
    mean(|yp[t] - yp[t-1]|) / mean(|yt[t] - yt[t-1]|).
    If target_times is provided, differences are computed ONLY between contiguous hourly steps
    where target_time[t] - target_time[t-1] == 1 hour, preventing gap/boundary leakage.
    If mean(|yt[t] - yt[t-1]|) <= 1e-12, returns NaN.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]
    if len(yt) < 2:
        return np.nan

    if target_times is not None:
        tt = pd.to_datetime(target_times[valid])
        time_diffs = (tt[1:] - tt[:-1]).total_seconds() / 3600.0
        contiguous_mask = np.abs(time_diffs - 1.0) < 1e-3
        if not contiguous_mask.any():
            return np.nan
        diff_true = np.abs(yt[1:] - yt[:-1])[contiguous_mask]
        diff_pred = np.abs(yp[1:] - yp[:-1])[contiguous_mask]
    else:
        diff_true = np.abs(np.diff(yt))
        diff_pred = np.abs(np.diff(yp))

    mean_diff_true = float(np.mean(diff_true))
    mean_diff_pred = float(np.mean(diff_pred))

    if mean_diff_true <= EPS:
        return np.nan
    if mean_diff_pred <= EPS:
        return 0.0
    return float(mean_diff_pred / mean_diff_true)


def compute_event_amplitude_retention(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: np.ndarray | float
) -> float:
    """
    Computes ratio of mean predicted amplitude to mean observed amplitude during peak episodes
    where y_true exceeds threshold: mean(y_pred[y_true > threshold]) / mean(y_true[y_true > threshold]).
    Formerly referred to as peak_retention.
    If no events exist or mean(y_true[y_true > threshold]) <= 1e-12, returns NaN.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[valid], y_pred[valid]

    if isinstance(threshold, (int, float)):
        thresh_arr = np.full_like(yt, float(threshold))
    else:
        thresh_arr = np.asarray(threshold, dtype=float)[valid]

    peak_mask = yt > thresh_arr
    if not peak_mask.any():
        return np.nan

    mean_peak_true = float(np.mean(yt[peak_mask]))
    mean_peak_pred = float(np.mean(yp[peak_mask]))

    if mean_peak_true <= EPS:
        return np.nan
    return float(mean_peak_pred / mean_peak_true)


def compute_all_dynamic_fidelity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: np.ndarray | float,
    target_times: np.ndarray | None = None,
) -> Dict[str, float]:
    """Computes all 7 mandatory dynamic fidelity metrics as a dictionary."""
    return {
        "variance_retention": compute_variance_retention(y_true, y_pred),
        "std_ratio": compute_std_ratio(y_true, y_pred),
        "alpha_kge": compute_alpha_kge(y_true, y_pred),
        "correlation": compute_correlation(y_true, y_pred),
        "amplitude_ratio": compute_amplitude_ratio(y_true, y_pred),
        "temporal_variability": compute_temporal_variability(y_true, y_pred, target_times),
        "event_amplitude_retention": compute_event_amplitude_retention(y_true, y_pred, threshold),
    }
