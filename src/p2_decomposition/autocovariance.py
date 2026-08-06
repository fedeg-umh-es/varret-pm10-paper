"""Train-only, missingness-aware autocovariance estimation.

Contract: ``P2_PAIRED_DECOMPOSITION_CONTRACT.md`` section 4.

Primary estimator::

    gamma_hat(k) = (1 / n_k) * sum_{t in V_k} (y_t - mu_hat) (y_{t-k} - mu_hat)

where ``V_k`` is the set of training dates for which both calendar-aligned
observations at ``t`` and ``t - k`` days are present, and ``n_k = |V_k|``.
``mu_hat`` is the mean of the observed training values only.

Nothing in this module may read an observation at or after the origin date.
The caller supplies the exclusive training end position; see
:func:`training_end_position`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calendar import DailySeries, compress_time_dropping_gaps

__all__ = [
    "AutocovarianceEstimate",
    "estimate_autocovariances",
    "estimate_autocovariances_compressed_time",
    "training_end_position",
]


@dataclass(frozen=True)
class AutocovarianceEstimate:
    """Train-only mean and autocovariances with per-lag valid-pair counts."""

    mu: float
    gamma: np.ndarray
    n_pairs: np.ndarray
    n_train_observed: int
    train_start_position: int
    train_end_position: int
    max_lag: int
    estimator: str = "CALENDAR_ALIGNED_TRAIN_ONLY"
    status: str = "OK"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_usable(self) -> bool:
        return self.status == "OK"

    def pair_count_min(self, up_to_lag: int) -> int:
        """Smallest valid-pair count among lags ``0..up_to_lag``."""
        return int(np.min(self.n_pairs[: up_to_lag + 1]))


def training_end_position(origin_position: int, train_end_offset_days: int) -> int:
    """Inclusive end position of the training window for a given origin.

    The contract requires ``max(train_date) < origin_date``, so the default
    offset of one day makes the training window end on the day before the
    origin. The lag vector ``x_t`` evaluated *at* the origin is a separate
    object: it is information available at the origin, exactly like the value
    persistence uses, and it is not part of the estimation sample.
    """
    if train_end_offset_days < 1:
        raise ValueError(
            "train_end_offset_days must be >= 1 so that max(train_date) < origin_date; "
            f"got {train_end_offset_days}"
        )
    return origin_position - train_end_offset_days


def estimate_autocovariances(
    series: DailySeries,
    *,
    train_end_position: int,
    max_lag: int,
    train_start_position: int = 0,
    min_pairs_per_lag: int = 1,
    min_train_observations: int = 1,
) -> AutocovarianceEstimate:
    """Estimate ``mu_hat`` and ``gamma_hat(0..max_lag)`` from training data only.

    Parameters
    ----------
    series:
        Complete-calendar daily series. Positions beyond
        ``train_end_position`` are never touched.
    train_end_position:
        Inclusive end position of the training window.
    max_lag:
        Largest lag required, ``max(p) + max(h) - 1`` for the AR(p) systems.
    min_pairs_per_lag:
        A lag with fewer calendar-aligned observed pairs than this is marked
        insufficient; its ``gamma`` is set to ``NaN`` rather than estimated
        from a handful of pairs.

    Returns
    -------
    AutocovarianceEstimate
        ``status`` is ``"OK"`` when the window is usable, otherwise
        ``"INSUFFICIENT_TRAIN_OBSERVATIONS"`` or
        ``"INSUFFICIENT_PAIRS_AT_LAG"``.
    """
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")

    empty = np.full(max_lag + 1, np.nan)
    if train_end_position < train_start_position:
        return AutocovarianceEstimate(
            mu=float("nan"),
            gamma=empty,
            n_pairs=np.zeros(max_lag + 1, dtype=int),
            n_train_observed=0,
            train_start_position=train_start_position,
            train_end_position=train_end_position,
            max_lag=max_lag,
            status="INSUFFICIENT_TRAIN_OBSERVATIONS",
            notes=("empty training window",),
        )

    window = series.values[train_start_position : train_end_position + 1]
    observed = ~np.isnan(window)
    n_train_observed = int(observed.sum())

    if n_train_observed < min_train_observations:
        return AutocovarianceEstimate(
            mu=float("nan"),
            gamma=empty,
            n_pairs=np.zeros(max_lag + 1, dtype=int),
            n_train_observed=n_train_observed,
            train_start_position=train_start_position,
            train_end_position=train_end_position,
            max_lag=max_lag,
            status="INSUFFICIENT_TRAIN_OBSERVATIONS",
            notes=(f"{n_train_observed} < {min_train_observations}",),
        )

    mu = float(np.nanmean(window))
    deviations = window - mu

    gamma = np.full(max_lag + 1, np.nan)
    n_pairs = np.zeros(max_lag + 1, dtype=int)
    for k in range(max_lag + 1):
        if k == 0:
            left = deviations
            right = deviations
        else:
            left = deviations[k:]
            right = deviations[:-k]
        # Calendar-aligned by construction: `left` and `right` are the same
        # array offset by exactly k calendar days, because the index has no
        # gaps. A pair contributes only when both days are observed.
        valid = np.isfinite(left) & np.isfinite(right)
        count = int(valid.sum())
        n_pairs[k] = count
        if count > 0:
            gamma[k] = float(np.sum(left[valid] * right[valid]) / count)

    insufficient = [int(k) for k in range(max_lag + 1) if n_pairs[k] < min_pairs_per_lag]
    if insufficient:
        gamma[np.asarray(insufficient, dtype=int)] = np.nan
        return AutocovarianceEstimate(
            mu=mu,
            gamma=gamma,
            n_pairs=n_pairs,
            n_train_observed=n_train_observed,
            train_start_position=train_start_position,
            train_end_position=train_end_position,
            max_lag=max_lag,
            status="INSUFFICIENT_PAIRS_AT_LAG",
            notes=tuple(f"lag {k}: {n_pairs[k]} < {min_pairs_per_lag}" for k in insufficient[:5]),
        )

    return AutocovarianceEstimate(
        mu=mu,
        gamma=gamma,
        n_pairs=n_pairs,
        n_train_observed=n_train_observed,
        train_start_position=train_start_position,
        train_end_position=train_end_position,
        max_lag=max_lag,
        status="OK",
    )


def estimate_autocovariances_compressed_time(
    series: DailySeries,
    *,
    train_end_position: int,
    max_lag: int,
    train_start_position: int = 0,
) -> AutocovarianceEstimate:
    """Invalid comparison estimator: drop gaps first, then lag — **sensitivity only**.

    This is the historical compacted-calendar calculation. It is retained so
    that ``missingness_sensitivity.csv`` can quantify the difference against
    the calendar-preserving primary estimator. It must never be used as a
    primary reference; its ``estimator`` field marks it accordingly.
    """
    truncated = DailySeries(
        station=series.station,
        index=series.index[train_start_position : train_end_position + 1],
        values=series.values[train_start_position : train_end_position + 1],
        source_path=series.source_path,
    )
    compressed = compress_time_dropping_gaps(truncated)
    n_train_observed = int(compressed.size)

    gamma = np.full(max_lag + 1, np.nan)
    n_pairs = np.zeros(max_lag + 1, dtype=int)
    if n_train_observed == 0:
        status = "INSUFFICIENT_TRAIN_OBSERVATIONS"
        mu = float("nan")
    else:
        status = "OK"
        mu = float(np.mean(compressed))
        deviations = compressed - mu
        for k in range(max_lag + 1):
            if k >= n_train_observed:
                continue
            left = deviations[k:] if k else deviations
            right = deviations[:-k] if k else deviations
            n_pairs[k] = int(left.size)
            gamma[k] = float(np.sum(left * right) / left.size)

    return AutocovarianceEstimate(
        mu=mu,
        gamma=gamma,
        n_pairs=n_pairs,
        n_train_observed=n_train_observed,
        train_start_position=train_start_position,
        train_end_position=train_end_position,
        max_lag=max_lag,
        estimator="COMPRESSED_TIME_INVALID_AS_PRIMARY",
        status=status,
        notes=("gaps dropped before lagging; lag k means k observed steps, not k days",),
    )


def pair_counts_to_frame(estimate: AutocovarianceEstimate, **keys: object) -> pd.DataFrame:
    """Long-format valid-pair counts by lag, for the diagnostics artefacts."""
    frame = pd.DataFrame(
        {
            "lag": np.arange(estimate.max_lag + 1, dtype=int),
            "n_pairs": estimate.n_pairs,
            "gamma": estimate.gamma,
        }
    )
    for name, value in keys.items():
        frame.insert(0, name, value)
    return frame
