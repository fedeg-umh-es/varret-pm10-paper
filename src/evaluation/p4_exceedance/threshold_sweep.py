"""Threshold handling (Fase 7).

Three threshold modes are kept structurally separate and are never allowed
to collapse into each other:

- ``regulatory_threshold_result``: a fixed, externally defined threshold
  (e.g. a legal PM10 limit). threshold_mode = "FIXED".
- ``diagnostic_sweep``: scans candidate thresholds against the evaluation
  (test) data itself. Always labeled threshold_mode = "POST_HOC_DIAGNOSTIC"
  and always carries usable_as_primary_estimate=False — the best threshold
  it finds must never be used to produce the headline performance number.
- ``calibrated_threshold_result``: selects a threshold using only
  calibration/train data that is temporally prior to the evaluation
  period. usable_as_primary_estimate is only True when that temporal
  ordering is verified; otherwise it raises rather than silently
  mislabeling a post-hoc selection as a valid calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import compute_full_event_metrics


@dataclass
class ThresholdResult:
    threshold: float
    threshold_source: str
    threshold_mode: str
    calibration_period: tuple | None
    evaluation_period: tuple | None
    usable_as_primary_estimate: bool
    sweep_table: pd.DataFrame
    selection_metric: str | None = None

    def to_manifest_fields(self) -> dict:
        return {
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "threshold_mode": self.threshold_mode,
            "calibration_period": list(self.calibration_period) if self.calibration_period else None,
            "evaluation_period": list(self.evaluation_period) if self.evaluation_period else None,
        }


def _sweep_table(y_true, y_pred, thresholds) -> pd.DataFrame:
    rows = [compute_full_event_metrics(y_true, y_pred, t) for t in thresholds]
    return pd.DataFrame(rows)


def regulatory_threshold_result(
    y_true,
    y_pred,
    threshold: float,
    *,
    threshold_source: str = "regulatory_limit",
    evaluation_period: tuple | None = None,
) -> ThresholdResult:
    """A fixed regulatory threshold, evaluated but never selected from data."""
    table = _sweep_table(y_true, y_pred, [threshold])
    return ThresholdResult(
        threshold=float(threshold),
        threshold_source=threshold_source,
        threshold_mode="FIXED",
        calibration_period=None,
        evaluation_period=evaluation_period,
        usable_as_primary_estimate=True,
        sweep_table=table,
        selection_metric=None,
    )


def diagnostic_sweep(
    y_true,
    y_pred,
    thresholds,
    *,
    selection_metric: str = "csi",
    evaluation_period: tuple | None = None,
) -> ThresholdResult:
    """Scan thresholds against the evaluation data itself.

    The result is exploratory by construction: it is always
    POST_HOC_DIAGNOSTIC and must never feed the primary performance
    estimate, regardless of which threshold looks best.
    """
    table = _sweep_table(y_true, y_pred, thresholds)
    if selection_metric not in table.columns:
        raise ValueError(f"Unknown selection_metric '{selection_metric}'.")
    best_row = table.loc[table[selection_metric].idxmax()]
    return ThresholdResult(
        threshold=float(best_row["threshold"]),
        threshold_source="post_hoc_diagnostic_sweep",
        threshold_mode="POST_HOC_DIAGNOSTIC",
        calibration_period=None,
        evaluation_period=evaluation_period,
        usable_as_primary_estimate=False,
        sweep_table=table,
        selection_metric=selection_metric,
    )


def calibrated_threshold_result(
    y_true_calibration,
    y_pred_calibration,
    thresholds,
    *,
    calibration_period: tuple,
    evaluation_period: tuple,
    selection_metric: str = "csi",
    threshold_source: str = "train_or_validation_calibration",
) -> ThresholdResult:
    """Select a threshold using calibration data only.

    Requires calibration_period to end strictly before evaluation_period
    begins; otherwise the caller is attempting to calibrate on the test
    window, which this function refuses to label as evaluative.
    """
    calib_start, calib_end = calibration_period
    eval_start, eval_end = evaluation_period
    calib_end_ts = pd.Timestamp(calib_end)
    eval_start_ts = pd.Timestamp(eval_start)
    if not (calib_end_ts <= eval_start_ts):
        raise ValueError(
            "calibration_period must end at or before evaluation_period begins; "
            f"got calibration_period ending {calib_end} and evaluation_period "
            f"starting {eval_start}. A threshold selected on data overlapping "
            "or after the evaluation window cannot be labeled a valid "
            "calibration; use diagnostic_sweep instead and accept the "
            "POST_HOC_DIAGNOSTIC label."
        )

    table = _sweep_table(y_true_calibration, y_pred_calibration, thresholds)
    if selection_metric not in table.columns:
        raise ValueError(f"Unknown selection_metric '{selection_metric}'.")
    best_row = table.loc[table[selection_metric].idxmax()]

    return ThresholdResult(
        threshold=float(best_row["threshold"]),
        threshold_source=threshold_source,
        threshold_mode="CALIBRATED",
        calibration_period=calibration_period,
        evaluation_period=evaluation_period,
        usable_as_primary_estimate=True,
        sweep_table=table,
        selection_metric=selection_metric,
    )
