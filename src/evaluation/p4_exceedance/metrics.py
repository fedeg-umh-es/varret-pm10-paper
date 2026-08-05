"""Full exceedance contingency metrics.

Reuses src.evaluation.compute_event_metrics.compute_event_metrics (the
existing repo implementation of recall/precision/f1/flag_rate/base_rate)
for the metrics it already defines, and adds the remaining ones this
diagnostic module needs (hit rate, false alarm rate, CSI, event bias,
exceedance intensity error) rather than redefining recall/precision from
scratch.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.compute_event_metrics import compute_event_metrics as _base_event_metrics

EVENT_METRIC_COLUMNS = (
    "threshold",
    "n_events_true",
    "n_events_pred",
    "hits",
    "misses",
    "false_alarms",
    "correct_negatives",
    "hit_rate",
    "false_alarm_rate",
    "precision",
    "recall",
    "csi",
    "event_bias",
    "exceedance_intensity_error",
)


def compute_full_event_metrics(y_true, y_pred, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    base = _base_event_metrics(y_true, y_pred, threshold)

    event_true = y_true > threshold
    event_pred = y_pred > threshold

    hits = int(np.sum(event_true & event_pred))
    misses = int(np.sum(event_true & ~event_pred))
    false_alarms = int(np.sum(~event_true & event_pred))
    correct_negatives = int(np.sum(~event_true & ~event_pred))

    denom_far = false_alarms + correct_negatives
    false_alarm_rate = float(false_alarms / denom_far) if denom_far > 0 else 0.0

    denom_csi = hits + misses + false_alarms
    csi = float(hits / denom_csi) if denom_csi > 0 else 0.0

    n_events_true = int(np.sum(event_true))
    n_events_pred = int(np.sum(event_pred))
    event_bias = float(n_events_pred / n_events_true) if n_events_true > 0 else float("nan")

    if n_events_true > 0:
        exceedance_intensity_error = float(np.mean(y_pred[event_true] - y_true[event_true]))
    else:
        exceedance_intensity_error = float("nan")

    return {
        "threshold": float(threshold),
        "n_events_true": n_events_true,
        "n_events_pred": n_events_pred,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "hit_rate": base["recall"],
        "false_alarm_rate": false_alarm_rate,
        "precision": base["precision"],
        "recall": base["recall"],
        "csi": csi,
        "event_bias": event_bias,
        "exceedance_intensity_error": exceedance_intensity_error,
    }
