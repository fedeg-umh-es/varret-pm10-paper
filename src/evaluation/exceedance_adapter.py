"""Canonical Schema Adapter and Exceedance Evaluation Engine for PM10 Forecasting."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


REQUIRED_CANONICAL_COLUMNS = {
    "origin_time",
    "target_time",
    "horizon",
    "fold",
    "model",
    "y_true",
    "y_pred",
    "y_persistence",
}


def normalize_predictions_schema(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalizes schema of prediction table to canonical columns.
    Canonical: origin_time, target_time, horizon, fold, model, y_true, y_pred, y_persistence.
    Aliases: origin_date -> origin_time, target_date -> target_time.
    Fallback: date -> target_time (only if target_time and target_date missing).
    """
    out = df.copy()
    metadata: Dict[str, Any] = {
        "station_status": "MISSING_FROM_SOURCE",
        "alias_used": None,
        "fallback_used": False,
        "validation_errors": [],
    }

    # Handle aliases
    if "origin_time" not in out.columns and "origin_date" in out.columns:
        out["origin_time"] = out["origin_date"]
        metadata["alias_used"] = "origin_date -> origin_time"

    if "target_time" not in out.columns and "target_date" in out.columns:
        out["target_time"] = out["target_date"]
        metadata["alias_used"] = (
            f"{metadata['alias_used']}; target_date -> target_time"
            if metadata["alias_used"]
            else "target_date -> target_time"
        )

    # Handle date fallback strictly for target_time
    if "target_time" not in out.columns and "date" in out.columns:
        out["target_time"] = out["date"]
        metadata["fallback_used"] = True

    # Check missing canonical columns
    missing = REQUIRED_CANONICAL_COLUMNS - set(out.columns)
    if missing:
        raise ValueError(
            f"Schema validation failed. Missing canonical columns: {sorted(missing)}"
        )

    # Convert datatypes
    out["origin_time"] = pd.to_datetime(out["origin_time"])
    out["target_time"] = pd.to_datetime(out["target_time"])
    out["horizon"] = out["horizon"].astype(int)
    out["fold"] = out["fold"].astype(int)
    out["model"] = out["model"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    out["y_persistence"] = pd.to_numeric(out["y_persistence"], errors="coerce")

    # Validate timestamps and horizon coherence
    invalid_time = out["target_time"] <= out["origin_time"]
    if invalid_time.any():
        metadata["validation_errors"].append(
            f"Found {invalid_time.sum()} rows where target_time <= origin_time"
        )

    invalid_h = out["horizon"] <= 0
    if invalid_h.any():
        metadata["validation_errors"].append(
            f"Found {invalid_h.sum()} rows where horizon <= 0"
        )

    # Validate hourly horizon delta coherence
    time_diff_hours = (
        out["target_time"] - out["origin_time"]
    ).dt.total_seconds() / 3600.0
    diff_mismatch = np.abs(time_diff_hours - out["horizon"]) > 1e-3
    if diff_mismatch.any():
        metadata["validation_errors"].append(
            f"Found {diff_mismatch.sum()} rows where (target_time - origin_time) in hours != horizon"
        )

    return out, metadata


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Checks for duplicate forecast cases by (model, fold, origin_time, target_time, horizon)."""
    keys = ["model", "fold", "origin_time", "target_time", "horizon"]
    dups = df[df.duplicated(subset=keys, keep=False)].copy()
    if not dups.empty:
        dups = dups.sort_values(keys)
    return dups


def check_case_alignment(df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    """
    Checks whether all models share the exact same set of forecast cases:
    (fold, origin_time, target_time, horizon, y_true).
    """
    case_keys = ["fold", "origin_time", "target_time", "horizon", "y_true"]
    models = sorted(df["model"].unique())

    if len(models) < 2:
        report = pd.DataFrame(
            [{
                "status": "SINGLE_MODEL",
                "n_models": len(models),
                "aligned": True,
                "n_cases": len(df),
            }]
        )
        return True, report

    # Extract case set per model
    model_cases = {}
    for m in models:
        sub = df[df["model"] == m][case_keys].drop_duplicates()
        sub_tuples = set(tuple(x) for x in sub.to_numpy())
        model_cases[m] = sub_tuples

    # Compare exact sets
    first_model = models[0]
    base_set = model_cases[first_model]

    all_aligned = True
    report_rows = []

    for m in models:
        case_set = model_cases[m]
        diff_from_base = len(base_set ^ case_set)
        is_exact = (diff_from_base == 0)
        if not is_exact:
            all_aligned = False
        report_rows.append({
            "model": m,
            "n_cases": len(case_set),
            "exact_match_with_base": is_exact,
            "symmetric_diff_count": diff_from_base,
        })

    report_df = pd.DataFrame(report_rows)
    return all_aligned, report_df


def compute_contingency_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: np.ndarray | float
) -> Dict[str, float]:
    """
    Computes contingency table and event metrics given y_true, y_pred, and threshold.
    Event condition: value > threshold (strictly greater).
    """
    if isinstance(threshold, (int, float)):
        thresh_arr = np.full_like(y_true, float(threshold))
    else:
        thresh_arr = np.asarray(threshold, dtype=float)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(thresh_arr)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    thresh_arr = thresh_arr[valid_mask]

    n = len(y_true)
    if n == 0:
        return {
            "n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "pod": np.nan, "recall": np.nan, "precision": np.nan,
            "far": np.nan, "pofd": np.nan, "csi": np.nan,
            "event_bias": np.nan, "exceedance_intensity_error": np.nan,
        }

    actual_event = y_true > thresh_arr
    pred_event = y_pred > thresh_arr

    tp = int(np.sum(actual_event & pred_event))
    fp = int(np.sum(~actual_event & pred_event))
    fn = int(np.sum(actual_event & ~pred_event))
    tn = int(np.sum(~actual_event & ~pred_event))

    pod = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    recall = pod
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    pofd = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    event_bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else np.nan

    # Exceedance intensity error: mean(y_pred - y_true) where y_true > threshold
    if tp + fn > 0:
        exceedance_intensity_error = float(
            np.mean(y_pred[actual_event] - y_true[actual_event])
        )
    else:
        exceedance_intensity_error = np.nan

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pod": float(pod) if not np.isnan(pod) else np.nan,
        "recall": float(recall) if not np.isnan(recall) else np.nan,
        "precision": float(precision) if not np.isnan(precision) else np.nan,
        "far": float(far) if not np.isnan(far) else np.nan,
        "pofd": float(pofd) if not np.isnan(pofd) else np.nan,
        "csi": float(csi) if not np.isnan(csi) else np.nan,
        "event_bias": float(event_bias) if not np.isnan(event_bias) else np.nan,
        "exceedance_intensity_error": float(exceedance_intensity_error) if not np.isnan(exceedance_intensity_error) else np.nan,
    }


def compute_kendall_taub(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes Kendall tau-b rank correlation coefficient between two time-series vectors
    x and y (e.g. y_pred for Model 1 vs Model 2 across hourly test steps at a given horizon),
    handling ties explicitly via tau-b formulation.
    """
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_v = x[valid_mask]
    y_v = y[valid_mask]
    if len(x_v) < 2:
        return np.nan
    res = kendalltau(x_v, y_v, variant="b")
    return float(res.statistic) if not np.isnan(res.statistic) else np.nan


def classify_rank_reversal(
    skill_diff: float, event_metric_diff: float, tau: float
) -> str:
    """
    Classifies rank reversal between continuous skill and event metric:
    - YES: continuous skill prefers model A, event metric prefers model B (strict opposite direction).
    - NO: both agree on model ordering.
    - TRADE_OFF_ONLY: metrics disagree but tau indicates partial/tie trade-off without full reversal.
    - NOT_EVALUABLE: missing values or zero denominator.
    """
    if np.isnan(skill_diff) or np.isnan(event_metric_diff):
        return "NOT_EVALUABLE"
    if skill_diff == 0.0 or event_metric_diff == 0.0:
        return "TRADE_OFF_ONLY" if skill_diff != event_metric_diff else "NO"
    if (skill_diff > 0 and event_metric_diff < 0) or (skill_diff < 0 and event_metric_diff > 0):
        return "YES"
    return "NO"
