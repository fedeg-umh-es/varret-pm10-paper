"""Validated temporal pairing for model-versus-baseline comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


PAIRING_KEYS = ["fold", "origin_date", "date", "horizon"]


def pair_model_and_baseline(
    model_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    """Return finite, one-to-one model/baseline pairs in temporal order."""
    required = set(PAIRING_KEYS) | {"y_true", "y_pred"}
    for label, frame in (("model", model_predictions), ("baseline", baseline_predictions)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{context}: {label} predictions missing columns: {sorted(missing)}")
        duplicates = frame.duplicated(PAIRING_KEYS, keep=False)
        if duplicates.any():
            duplicate_keys = frame.loc[duplicates, PAIRING_KEYS].head(3).to_dict("records")
            raise ValueError(f"{context}: duplicate {label} pairing keys: {duplicate_keys}")
        origins = pd.to_datetime(frame["origin_date"], errors="raise")
        if not origins.is_monotonic_increasing:
            raise ValueError(f"{context}: {label} origins are not in chronological order")

    baseline = baseline_predictions[PAIRING_KEYS + ["y_true", "y_pred"]].rename(
        columns={"y_true": "y_true_baseline", "y_pred": "y_pred_baseline"}
    )
    paired = model_predictions.merge(
        baseline,
        on=PAIRING_KEYS,
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if paired.empty:
        raise ValueError(f"{context}: no common model/baseline prediction pairs")
    if not np.allclose(paired["y_true"], paired["y_true_baseline"], equal_nan=True):
        raise ValueError(f"{context}: model and baseline y_true values differ")

    finite = np.isfinite(
        paired[["y_true", "y_true_baseline", "y_pred", "y_pred_baseline"]].to_numpy(dtype=float)
    ).all(axis=1)
    paired = paired.loc[finite].copy()
    if paired.empty:
        raise ValueError(f"{context}: no finite common model/baseline prediction pairs")

    paired["_origin_timestamp"] = pd.to_datetime(paired["origin_date"], errors="raise")
    paired = paired.sort_values(["_origin_timestamp", "date"], kind="stable").drop(columns="_origin_timestamp")
    paired.attrs["candidate_pairs"] = int(len(finite))
    paired.attrs["invalid_pairs_dropped"] = int((~finite).sum())
    return paired.reset_index(drop=True)
