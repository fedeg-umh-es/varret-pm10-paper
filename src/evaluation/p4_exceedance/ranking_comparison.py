"""Ranking comparison between a continuous metric and an event metric (Fase 4 / B2).

Fixes the original bug: calling ``pd.DataFrame([]).sort_values("horizon")``
on a schema-less empty frame raises ``KeyError: 'horizon'`` whenever there
is a single model, no models, or no comparable horizon. This module always
constructs the output with an explicit column list, so the "nothing to
compare" case degrades to an empty-but-well-formed DataFrame instead of an
exception.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

EVALUATION_STATUSES = (
    "EVALUATED",
    "NOT_EVALUABLE_SINGLE_MODEL",
    "NOT_EVALUABLE_NO_COMMON_CASES",
    "NOT_EVALUABLE_INSUFFICIENT_MODELS",
    "NOT_EVALUABLE_INCOMPLETE_RANKING",
)

RANKING_COMPARISON_COLUMNS = (
    "station",
    "horizon",
    "metric_continuous",
    "metric_event",
    "kendall_tau",
    "kendall_pvalue",
    "n_models",
    "n_pairs",
    "n_reversals",
    "evaluation_status",
)


def _count_reversals(continuous: np.ndarray, event: np.ndarray) -> int:
    n = len(continuous)
    n_reversals = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_c = np.sign(continuous[i] - continuous[j])
            sign_e = np.sign(event[i] - event[j])
            if sign_c != 0 and sign_e != 0 and sign_c != sign_e:
                n_reversals += 1
    return n_reversals


def ranking_comparison(
    metrics_df: pd.DataFrame,
    *,
    metric_continuous_col: str,
    metric_event_col: str,
    metric_continuous_name: str | None = None,
    metric_event_name: str | None = None,
    group_columns: list | None = None,
    misaligned_groups: set | None = None,
) -> pd.DataFrame:
    """Compare model rankings under a continuous metric vs. an event metric.

    Parameters
    ----------
    metrics_df:
        One row per (group..., model) with numeric columns
        `metric_continuous_col` and `metric_event_col`.
    group_columns:
        Defaults to ['station', 'horizon'] restricted to columns present.
        Both must exist in metrics_df, along with 'model'.
    misaligned_groups:
        Set of group-key tuples (same order as group_columns) that failed
        the case-alignment check upstream (see integrity_checks). Those
        groups are reported as NOT_EVALUABLE_NO_COMMON_CASES without
        computing Kendall tau.

    Never raises for "nothing to compare" (0 groups, single model, no
    models); always returns a DataFrame with RANKING_COMPARISON_COLUMNS
    (plus any extra group columns), even when empty.
    """
    metric_continuous_name = metric_continuous_name or metric_continuous_col
    metric_event_name = metric_event_name or metric_event_col
    misaligned_groups = misaligned_groups or set()

    if len(metrics_df) == 0:
        # Nothing to compare at all — the literal B2 regression case
        # (pd.DataFrame([]) has neither rows nor columns). Degrade to the
        # stable empty schema instead of validating columns that were
        # never going to be used.
        fallback_group_columns = group_columns or ["station", "horizon"]
        output_columns = [*fallback_group_columns, *RANKING_COMPARISON_COLUMNS[2:]]
        return pd.DataFrame(columns=output_columns)

    if group_columns is None:
        group_columns = [c for c in ("station", "horizon") if c in metrics_df.columns]

    required = {"station", "horizon", "model"}
    missing = required - set(metrics_df.columns)
    if missing:
        raise ValueError(f"metrics_df is missing required column(s): {sorted(missing)}")
    if metric_continuous_col not in metrics_df.columns:
        raise ValueError(f"metrics_df is missing metric_continuous_col='{metric_continuous_col}'")
    if metric_event_col not in metrics_df.columns:
        raise ValueError(f"metrics_df is missing metric_event_col='{metric_event_col}'")

    output_columns = [*group_columns, *RANKING_COMPARISON_COLUMNS[2:]]
    rows = []

    for group_key, group_df in metrics_df.groupby(group_columns, dropna=False):
        group_key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)

        n_models_total = group_df["model"].nunique()

        if group_key_tuple in misaligned_groups:
            status = "NOT_EVALUABLE_NO_COMMON_CASES"
            tau = np.nan
            pvalue = np.nan
            n_pairs = 0
            n_reversals = np.nan
            n_models = n_models_total
        elif n_models_total == 0:
            status = "NOT_EVALUABLE_INSUFFICIENT_MODELS"
            tau = np.nan
            pvalue = np.nan
            n_pairs = 0
            n_reversals = np.nan
            n_models = 0
        elif n_models_total == 1:
            status = "NOT_EVALUABLE_SINGLE_MODEL"
            tau = np.nan
            pvalue = np.nan
            n_pairs = 0
            n_reversals = np.nan
            n_models = 1
        else:
            valid = group_df.dropna(subset=[metric_continuous_col, metric_event_col])
            n_models_valid = valid["model"].nunique()
            n_models = n_models_total

            if n_models_valid < n_models_total:
                status = "NOT_EVALUABLE_INCOMPLETE_RANKING"
                tau = np.nan
                pvalue = np.nan
                n_pairs = 0
                n_reversals = np.nan
            elif n_models_valid < 2:
                status = "NOT_EVALUABLE_INSUFFICIENT_MODELS"
                tau = np.nan
                pvalue = np.nan
                n_pairs = 0
                n_reversals = np.nan
            else:
                by_model = valid.groupby("model")[[metric_continuous_col, metric_event_col]].mean()
                continuous = by_model[metric_continuous_col].to_numpy()
                event = by_model[metric_event_col].to_numpy()
                tau, pvalue = kendalltau(continuous, event, variant="b")
                n_pairs = int(n_models_valid * (n_models_valid - 1) / 2)
                n_reversals = _count_reversals(continuous, event)
                status = "EVALUATED"

        row = dict(zip(group_columns, group_key_tuple))
        row.update(
            {
                "metric_continuous": metric_continuous_name,
                "metric_event": metric_event_name,
                "kendall_tau": tau,
                "kendall_pvalue": pvalue,
                "n_models": n_models,
                "n_pairs": n_pairs,
                "n_reversals": n_reversals,
                "evaluation_status": status,
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows, columns=output_columns) if rows else pd.DataFrame(columns=output_columns)

    if "horizon" in result.columns:
        result = result.sort_values("horizon", kind="stable").reset_index(drop=True)

    return result
