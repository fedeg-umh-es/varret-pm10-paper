"""Fold Stability Evaluator for PM10 Forecasting Evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.evaluation.dynamic_fidelity import compute_all_dynamic_fidelity_metrics
from src.evaluation.exceedance_adapter import compute_contingency_metrics


def compute_fold_level_metrics(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Computes dynamic fidelity and event metrics for every (model, horizon, fold) group.
    Enforces intra-fold contiguous step differences for temporal_variability.
    """
    rows: List[Dict[str, Any]] = []

    # Sort strictly by model, horizon, fold, target_time to ensure temporal continuity within fold
    df_sorted = df_norm.sort_values(["model", "horizon", "fold", "target_time"]).reset_index(drop=True)

    grouped = df_sorted.groupby(["model", "horizon", "fold"], sort=True)

    for (model, horizon, fold), group in grouped:
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=float)
        y_pers = group["y_persistence"].to_numpy(dtype=float)
        p75_thresh = group["p75_train"].to_numpy(dtype=float)
        target_times = group["target_time"].to_numpy()

        n_cases = len(y_true)

        # RMSE Skill
        rmse_m = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        rmse_p = float(np.sqrt(np.mean((y_pers - y_true) ** 2)))
        rmse_skill = float(1.0 - (rmse_m / rmse_p))

        # Dynamic Fidelity Metrics with intra-fold contiguous target_times
        fid_dict = compute_all_dynamic_fidelity_metrics(y_true, y_pred, p75_thresh, target_times)

        # Contingency Event Metrics under PRIMARY_FIXED_THRESHOLD (fold_train_p75)
        ev_dict = compute_contingency_metrics(y_true, y_pred, p75_thresh)

        # Dynamic fidelity degradation check (non-redundant dimensions)
        fid_degraded = (
            (fid_dict["variance_retention"] < 0.25)
            or (fid_dict["amplitude_ratio"] < 0.5)
            or (fid_dict["temporal_variability"] < 0.5)
        )

        # Event representation degradation check
        ev_degraded = (
            (ev_dict["pod"] < 0.1)
            or (ev_dict["csi"] < 0.1)
            or (ev_dict["event_bias"] < 0.5)
        )

        concordant_degradation = (rmse_skill > 0.0) and (fid_degraded or ev_degraded)

        rows.append({
            "model": model,
            "horizon": horizon,
            "fold": fold,
            "N": n_cases,
            "rmse_skill": rmse_skill,
            "variance_retention": fid_dict["variance_retention"],
            "std_ratio": fid_dict["std_ratio"],
            "alpha_kge": fid_dict["alpha_kge"],
            "correlation": fid_dict["correlation"],
            "amplitude_ratio": fid_dict["amplitude_ratio"],
            "temporal_variability": fid_dict["temporal_variability"],
            "event_amplitude_retention": fid_dict["event_amplitude_retention"],
            "tp": ev_dict["tp"],
            "fp": ev_dict["fp"],
            "fn": ev_dict["fn"],
            "tn": ev_dict["tn"],
            "POD": ev_dict["pod"],
            "CSI": ev_dict["csi"],
            "event_bias": ev_dict["event_bias"],
            "positive_skill": rmse_skill > 0.0,
            "concordant_degradation": concordant_degradation,
        })

    return pd.DataFrame(rows)


def summarize_sarima_fold_stability(df_fold: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes fold-level stability for target models (specifically SARIMA at h=24 and h=48).
    Returns median, range, positive skill count, and concordant degradation count across 5 folds.
    """
    rows: List[Dict[str, Any]] = []

    for (model, horizon), group in df_fold.groupby(["model", "horizon"], sort=True):
        n_folds = len(group)
        n_pos_skill = int(group["positive_skill"].sum())
        n_concordant_deg = int(group["concordant_degradation"].sum())

        # Dynamic collapse across all folds (variance retention < 0.25 or temporal variability < 0.5)
        dyn_collapse_all = bool(((group["variance_retention"] < 0.25) | (group["temporal_variability"] < 0.5)).all())
        
        # Complete event failure across all folds strictly requires POD == 0 and CSI == 0 in EVERY fold
        complete_event_fail_all = bool(((group["POD"] == 0.0) & (group["CSI"] == 0.0)).all())
        
        # Degraded event representation across all folds (POD < 0.15 or CSI < 0.15 in all folds)
        event_degraded_all = bool(((group["POD"] < 0.15) | (group["CSI"] < 0.15)).all())

        stability_pattern = f"GHOST_PATTERN_REPLICATED_{n_concordant_deg}_OF_{n_folds}_FOLDS"

        metrics = ["rmse_skill", "variance_retention", "correlation", "amplitude_ratio", "temporal_variability", "POD", "CSI"]

        summary_row: Dict[str, Any] = {
            "model": model,
            "horizon": horizon,
            "total_folds": n_folds,
            "folds_with_positive_skill": n_pos_skill,
            "folds_with_concordant_degradation": n_concordant_deg,
            "dynamic_collapse_all_folds": dyn_collapse_all,
            "complete_event_failure_all_folds": complete_event_fail_all,
            "degraded_event_representation_all_folds": event_degraded_all,
            "stability_pattern": stability_pattern,
        }

        for m in metrics:
            vals = group[m].dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                summary_row[f"{m}_median"] = float(np.median(vals))
                summary_row[f"{m}_min"] = float(np.min(vals))
                summary_row[f"{m}_max"] = float(np.max(vals))
            else:
                summary_row[f"{m}_median"] = np.nan
                summary_row[f"{m}_min"] = np.nan
                summary_row[f"{m}_max"] = np.nan

        rows.append(summary_row)

    return pd.DataFrame(rows)
