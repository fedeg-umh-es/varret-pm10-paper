"""Yule–Walker finite-memory optimal linear predictability reference for P2.

Mathematical contract: docs/p2/P2_MATHEMATICAL_CONTRACT.md

Key design decisions:
- Autocovariance from calendar-aligned valid pairs (NaN preserved on grid).
- scipy.linalg.solve instead of explicit matrix inversion.
- Full numerical diagnostics per (station, horizon, p).
- No silent regularization; regularization is recorded and flagged.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import solve, LinAlgError

OMP_LIMIT = 4  # respect environment thread limits


@dataclass
class YWResult:
    station: str
    horizon: int
    p: int
    gamma0: float
    autocovariances: np.ndarray  # γ(0), γ(1), ..., γ(p+h-2)   shape = (p+h-1,)
    cross_covariance: np.ndarray  # c_h = [γ(h), ..., γ(h+p-1)]  shape = (p,)
    coefficients: Optional[np.ndarray]  # β_h shape = (p,)  or None if failed
    yw_mse: float
    yw_rmse: float
    persistence_mse: float
    persistence_rmse: float
    yw_skill: float
    valid_pair_counts: np.ndarray  # n_k for k = 0, 1, ..., p+h-2
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    psd_status: str
    regularization: float
    numerical_status: str
    warnings: list[str] = field(default_factory=list)


def _estimate_autocovariance(
    y_calendar: pd.Series,
    max_lag: int,
    min_pairs: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate autocovariances γ(0)..γ(max_lag) from calendar-aligned valid pairs.

    Args:
        y_calendar: PM10 series indexed by complete daily DatetimeIndex (NaN for missing).
        max_lag: Maximum lag needed.
        min_pairs: Minimum required valid pairs to estimate a lag.

    Returns:
        gamma: array of shape (max_lag+1,) with γ(0), γ(1), ..., γ(max_lag)
        n_pairs: array of valid pair counts for each lag
    """
    arr = y_calendar.values.astype(float)
    n = len(arr)
    mu = np.nanmean(arr)

    gamma = np.full(max_lag + 1, np.nan)
    n_pairs = np.zeros(max_lag + 1, dtype=int)

    for k in range(max_lag + 1):
        y_t = arr[: n - k]
        y_tk = arr[k:]
        valid = ~np.isnan(y_t) & ~np.isnan(y_tk)
        nv = int(valid.sum())
        n_pairs[k] = nv
        if nv >= min_pairs:
            gamma[k] = np.sum((y_t[valid] - mu) * (y_tk[valid] - mu)) / nv

    return gamma, n_pairs


def _build_toeplitz(gamma: np.ndarray, p: int) -> np.ndarray:
    """Build p×p Toeplitz autocovariance matrix Γ_p from γ(0),...,γ(p-1)."""
    mat = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            lag = abs(i - j)
            mat[i, j] = gamma[lag]
    return mat


def _cross_covariance_vector(gamma: np.ndarray, h: int, p: int) -> np.ndarray:
    """Cross-covariance c_h = [γ(h), γ(h+1), ..., γ(h+p-1)] shape (p,)."""
    return gamma[h : h + p].copy()


def _check_psd(mat: np.ndarray, tol: float = 1e-10) -> tuple[float, float, float, str]:
    """Eigenvalue diagnostics and PSD status."""
    eigvals = np.linalg.eigvalsh(mat)
    min_ev = float(eigvals.min())
    max_ev = float(eigvals.max())
    cond = max_ev / max(abs(min_ev), 1e-30) if max_ev > 0 else np.inf
    if min_ev >= -tol:
        status = "PSD"
    elif min_ev >= -1e-6:
        status = "PSD_MARGINAL"
    else:
        status = "NOT_PSD"
    return min_ev, max_ev, cond, status


def _solve_yw(
    gamma_mat: np.ndarray,
    c_h: np.ndarray,
    tol_reg: float = 1e-8,
) -> tuple[Optional[np.ndarray], float, str, list[str]]:
    """Solve Γ_p β = c_h.

    Returns (β, regularization_applied, numerical_status, warnings_list).
    """
    warns: list[str] = []
    reg = 0.0

    if np.any(np.isnan(gamma_mat)) or np.any(np.isnan(c_h)):
        return None, 0.0, "MISSING_AUTOCOVARIANCES", ["nan in inputs"]

    try:
        beta = solve(gamma_mat, c_h, assume_a="sym")
        return beta, 0.0, "OK", warns
    except LinAlgError as exc:
        warns.append(f"scipy.linalg.solve failed: {exc}")

    # Try minimal ridge regularization
    reg = tol_reg * max(float(np.abs(np.diag(gamma_mat)).mean()), 1.0)
    mat_reg = gamma_mat + reg * np.eye(len(gamma_mat))
    warns.append(f"Applying ridge regularization λ={reg:.3e}")
    try:
        beta = solve(mat_reg, c_h, assume_a="sym")
        return beta, reg, "REGULARIZED", warns
    except LinAlgError as exc2:
        warns.append(f"Regularized solve also failed: {exc2}")
        return None, reg, "NUMERICALLY_UNSTABLE", warns


def compute_yw_reference(
    y_calendar: pd.Series,
    station: str,
    horizon: int,
    p: int,
    persistence_mse: float,
    persistence_rmse: float,
    min_pairs: int = 20,
    tol_neg_mse: float = 1e-8,
    tol_reg: float = 1e-8,
) -> YWResult:
    """Compute Yule–Walker linear reference for one (station, horizon, p).

    Args:
        y_calendar: PM10 indexed by complete daily DatetimeIndex (NaN for missing days).
        station: Station label.
        horizon: Forecast horizon h (1–7).
        p: AR order.
        persistence_mse: MSE of persistence baseline for this (station, horizon).
        persistence_rmse: RMSE of persistence baseline for this (station, horizon).
        min_pairs: Minimum valid pairs required per lag.
        tol_neg_mse: Tolerance for clamping small negative MSE to 0.
        tol_reg: Ridge regularization scale for ill-conditioned Gamma.

    Returns:
        YWResult with all diagnostics.
    """
    warns: list[str] = []
    max_lag = p + horizon - 1

    gamma, n_pairs = _estimate_autocovariance(y_calendar, max_lag, min_pairs)

    gamma0 = float(gamma[0]) if not np.isnan(gamma[0]) else np.nan

    if np.isnan(gamma0):
        return YWResult(
            station=station,
            horizon=horizon,
            p=p,
            gamma0=gamma0,
            autocovariances=gamma,
            cross_covariance=np.full(p, np.nan),
            coefficients=None,
            yw_mse=np.nan,
            yw_rmse=np.nan,
            persistence_mse=persistence_mse,
            persistence_rmse=persistence_rmse,
            yw_skill=np.nan,
            valid_pair_counts=n_pairs,
            min_eigenvalue=np.nan,
            max_eigenvalue=np.nan,
            condition_number=np.nan,
            psd_status="MISSING_GAMMA0",
            regularization=0.0,
            numerical_status="MISSING_AUTOCOVARIANCES",
            warnings=["gamma(0) is NaN — insufficient valid pairs"],
        )

    gamma_mat = _build_toeplitz(gamma, p)

    if np.any(np.isnan(gamma_mat)):
        missing_lags = [k for k in range(p) if np.isnan(gamma[k])]
        return YWResult(
            station=station,
            horizon=horizon,
            p=p,
            gamma0=gamma0,
            autocovariances=gamma,
            cross_covariance=np.full(p, np.nan),
            coefficients=None,
            yw_mse=np.nan,
            yw_rmse=np.nan,
            persistence_mse=persistence_mse,
            persistence_rmse=persistence_rmse,
            yw_skill=np.nan,
            valid_pair_counts=n_pairs,
            min_eigenvalue=np.nan,
            max_eigenvalue=np.nan,
            condition_number=np.nan,
            psd_status="MISSING_LAGS",
            regularization=0.0,
            numerical_status="MISSING_AUTOCOVARIANCES",
            warnings=[f"NaN autocovariances at lags: {missing_lags}"],
        )

    c_h = _cross_covariance_vector(gamma, horizon, p)

    if np.any(np.isnan(c_h)):
        return YWResult(
            station=station,
            horizon=horizon,
            p=p,
            gamma0=gamma0,
            autocovariances=gamma,
            cross_covariance=c_h,
            coefficients=None,
            yw_mse=np.nan,
            yw_rmse=np.nan,
            persistence_mse=persistence_mse,
            persistence_rmse=persistence_rmse,
            yw_skill=np.nan,
            valid_pair_counts=n_pairs,
            min_eigenvalue=np.nan,
            max_eigenvalue=np.nan,
            condition_number=np.nan,
            psd_status="MISSING_CROSS_COV",
            regularization=0.0,
            numerical_status="MISSING_AUTOCOVARIANCES",
            warnings=["NaN in cross-covariance vector c_h"],
        )

    min_ev, max_ev, cond, psd_status = _check_psd(gamma_mat)

    beta, reg, num_status, solve_warns = _solve_yw(gamma_mat, c_h, tol_reg)
    warns.extend(solve_warns)

    if beta is None:
        return YWResult(
            station=station,
            horizon=horizon,
            p=p,
            gamma0=gamma0,
            autocovariances=gamma,
            cross_covariance=c_h,
            coefficients=None,
            yw_mse=np.nan,
            yw_rmse=np.nan,
            persistence_mse=persistence_mse,
            persistence_rmse=persistence_rmse,
            yw_skill=np.nan,
            valid_pair_counts=n_pairs,
            min_eigenvalue=min_ev,
            max_eigenvalue=max_ev,
            condition_number=cond,
            psd_status=psd_status,
            regularization=reg,
            numerical_status=num_status,
            warnings=warns,
        )

    yw_mse_raw = gamma0 - float(c_h @ beta)

    if yw_mse_raw < -tol_neg_mse:
        warns.append(f"yw_mse_raw={yw_mse_raw:.6e} significantly negative — NUMERICAL_INSTABILITY")
        num_status = "NUMERICALLY_UNSTABLE"
        yw_mse = np.nan
        yw_rmse = np.nan
        yw_skill = np.nan
    elif yw_mse_raw < 0.0:
        warns.append(f"yw_mse_raw={yw_mse_raw:.6e} small negative; clamped to 0")
        yw_mse = 0.0
        yw_rmse = 0.0
        yw_skill = 1.0 - yw_rmse / persistence_rmse if persistence_rmse > 0 else np.nan
    else:
        yw_mse = float(yw_mse_raw)
        yw_rmse = float(np.sqrt(yw_mse))
        yw_skill = 1.0 - yw_rmse / persistence_rmse if persistence_rmse > 0 else np.nan

    if psd_status != "PSD" and num_status == "OK":
        warns.append(f"Gamma_p PSD status: {psd_status}")

    return YWResult(
        station=station,
        horizon=horizon,
        p=p,
        gamma0=gamma0,
        autocovariances=gamma,
        cross_covariance=c_h,
        coefficients=beta,
        yw_mse=yw_mse,
        yw_rmse=yw_rmse,
        persistence_mse=persistence_mse,
        persistence_rmse=persistence_rmse,
        yw_skill=yw_skill,
        valid_pair_counts=n_pairs,
        min_eigenvalue=min_ev,
        max_eigenvalue=max_ev,
        condition_number=cond,
        psd_status=psd_status,
        regularization=reg,
        numerical_status=num_status,
        warnings=warns,
    )


def load_calendar_series(csv_path: str, date_col: str = "date", pm10_col: str = "pm10") -> pd.Series:
    """Load daily PM10 CSV and reindex to complete calendar (NaN for missing days)."""
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Reindex to complete daily calendar
    full_idx = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    df = df.set_index(date_col)
    df = df.reindex(full_idx)

    return df[pm10_col].astype(float)


def compute_persistence_metrics(
    pred_df: pd.DataFrame,
    station: str,
    horizon: int,
) -> tuple[float, float]:
    """Extract persistence MSE and RMSE for (station, horizon) from predictions CSV.

    Args:
        pred_df: predictions DataFrame with columns dataset, model, horizon, y_true, y_pred.
        station: Station label used as dataset identifier prefix.
        horizon: Forecast horizon.

    Returns:
        (persistence_mse, persistence_rmse)
    """
    mask = (pred_df["model"] == "persistence") & (pred_df["horizon"] == horizon)
    if station:
        # Match dataset by station keyword
        mask = mask & pred_df["dataset"].str.contains(station, case=False)
    sub = pred_df[mask].dropna(subset=["y_true", "y_pred"])
    if sub.empty:
        return np.nan, np.nan
    residuals = sub["y_true"].values - sub["y_pred"].values
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    return mse, rmse
