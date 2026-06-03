"""Forecast accuracy metrics used in P33."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def rmse(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Compute root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Compute mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def relative_skill_vs_persistence(model_error: float, persistence_error: float) -> float:
    """Compute baseline-relative skill as 1 - model_error / persistence_error."""
    if persistence_error == 0 or not np.isfinite(persistence_error):
        return float("nan")
    return float(1.0 - (model_error / persistence_error))


def skill_rmse(
    y_true: Sequence[float] | np.ndarray,
    y_pred_model: Sequence[float] | np.ndarray,
    y_pred_persistence: Sequence[float] | np.ndarray,
) -> float:
    """Compute RMSE skill relative to persistence."""
    model_rmse = rmse(y_true, y_pred_model)
    persistence_rmse = rmse(y_true, y_pred_persistence)
    return relative_skill_vs_persistence(model_rmse, persistence_rmse)


def variance_retention(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Compute variance retention as var(y_pred) / var(y_true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    true_variance = float(np.var(y_true, ddof=0))
    if true_variance == 0:
        return float("nan")
    return float(np.var(y_pred, ddof=0) / true_variance)


def variance_ratio_alpha(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Alias for the paper's alpha variance-retention ratio."""
    return variance_retention(y_true, y_pred)


def skill_vp(skill: float, alpha: float) -> float:
    """Auxiliary skill adjusted by variance retention.

    Positive skill is multiplied by `min(1, alpha)`. This preserves skill when
    variance is retained or inflated, and penalizes variance collapse.
    """
    if not np.isfinite(skill) or not np.isfinite(alpha):
        return float("nan")
    return float(skill * min(1.0, max(0.0, alpha)))


def kge_components(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Compute KGE components r, alpha, beta and KGE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    std_true = float(np.std(y_true, ddof=0))
    std_pred = float(np.std(y_pred, ddof=0))

    if mean_true == 0 or std_true == 0 or std_pred == 0:
        return {"kge": float("nan"), "r": float("nan"), "alpha": float("nan"), "beta": float("nan")}

    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = (std_pred / mean_pred) / (std_true / mean_true)
    beta = mean_pred / mean_true
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return {"kge": float(kge), "r": float(r), "alpha": float(alpha), "beta": float(beta)}


def ranking_from_metric_dict(
    metric_by_model: Mapping[str, float] | None = None,
    ascending: bool = True,
    **kwargs: object,
) -> list[tuple[str, float]]:
    """Rank models by metric value, defaulting to lower-is-better."""
    if metric_by_model is None:
        metric_by_model = kwargs.get("metric_by_model")  # supports explicit keyword use
    if metric_by_model is None:
        raise ValueError("metric_by_model is required")
    return sorted(metric_by_model.items(), key=lambda item: (item[1], item[0]), reverse=not ascending)
