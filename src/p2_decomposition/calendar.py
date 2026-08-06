"""Complete-calendar handling for the P2 paired decomposition.

Contract: ``P2_PAIRED_DECOMPOSITION_CONTRACT.md`` section 4.

The single non-negotiable rule implemented here is that a daily series is
reindexed to the *complete* daily calendar and missing observations remain
``NaN``. Dropping missing days before lag construction compresses calendar
time and silently changes what a "lag of k days" means; that operation is
available in this module only as :func:`compress_time_dropping_gaps`, which is
explicitly labelled as a sensitivity-only estimator and is never used by the
primary path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DailySeries",
    "compress_time_dropping_gaps",
    "load_daily_series",
    "reindex_to_complete_calendar",
]


@dataclass(frozen=True)
class DailySeries:
    """A daily series on a gap-free calendar index, with ``NaN`` preserved.

    Attributes
    ----------
    station:
        Canonical station name.
    index:
        Complete daily :class:`~pandas.DatetimeIndex`; every calendar day
        between the first and last observed date is present exactly once.
    values:
        ``float64`` array aligned with ``index``. Missing days are ``NaN``.
    source_path:
        Path the series was read from, for provenance stamping.
    """

    station: str
    index: pd.DatetimeIndex
    values: np.ndarray
    source_path: str

    def __post_init__(self) -> None:
        if len(self.index) != len(self.values):
            raise ValueError(
                f"{self.station}: index length {len(self.index)} does not match "
                f"values length {len(self.values)}"
            )
        self.assert_complete_calendar()

    # -- invariants --------------------------------------------------------
    def assert_complete_calendar(self) -> None:
        """Raise unless the index is a strictly daily, gap-free, unique index."""
        if not isinstance(self.index, pd.DatetimeIndex):
            raise TypeError(f"{self.station}: index must be a DatetimeIndex")
        if self.index.has_duplicates:
            raise ValueError(f"{self.station}: calendar index contains duplicates")
        if not self.index.is_monotonic_increasing:
            raise ValueError(f"{self.station}: calendar index is not sorted")
        if len(self.index) > 1:
            deltas = np.diff(self.index.values).astype("timedelta64[D]").astype(int)
            if not np.all(deltas == 1):
                raise ValueError(
                    f"{self.station}: calendar index has gaps; "
                    f"{int((deltas != 1).sum())} non-daily step(s) found"
                )

    # -- accessors ---------------------------------------------------------
    @property
    def observed(self) -> np.ndarray:
        """Boolean mask of days with an observed value."""
        return ~np.isnan(self.values)

    @property
    def n_observed(self) -> int:
        return int(self.observed.sum())

    @property
    def n_missing(self) -> int:
        return int(len(self.values) - self.observed.sum())

    def position_of(self, date: pd.Timestamp) -> int:
        """Return the integer position of ``date``, or ``-1`` if outside the span."""
        try:
            return int(self.index.get_loc(pd.Timestamp(date)))
        except KeyError:
            return -1

    def lag_vector(self, origin_position: int, p: int) -> np.ndarray | None:
        """Return ``[y_t, y_{t-1}, ..., y_{t-p+1}]`` or ``None`` if unavailable.

        The vector is ordered newest-first, matching the ordering fixed by
        ``P2_PROJECT_CANON.md`` section 5.3. ``None`` is returned whenever any
        required calendar day falls outside the series or is missing; the
        caller must treat that as "forecast not available at this origin"
        rather than imputing anything.
        """
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        start = origin_position - p + 1
        if start < 0 or origin_position >= len(self.values):
            return None
        window = self.values[start : origin_position + 1][::-1]
        if np.isnan(window).any():
            return None
        return np.asarray(window, dtype=float)

    def lag_availability(self, p: int) -> np.ndarray:
        """Boolean mask over positions: is the full ``p``-lag vector observed?"""
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}")
        observed = self.observed.astype(np.int64)
        rolled = pd.Series(observed).rolling(p, min_periods=p).sum().to_numpy()
        return np.nan_to_num(rolled, nan=0.0) == p


def reindex_to_complete_calendar(
    frame: pd.DataFrame,
    *,
    station: str,
    date_column: str = "date",
    value_column: str = "pm10",
    source_path: str = "<in-memory>",
    freq: str = "D",
) -> DailySeries:
    """Reindex ``frame`` onto the complete calendar between its first and last date.

    Duplicate dates are a hard error: silently collapsing them would change the
    empirical autocovariance without any record of it.
    """
    if date_column not in frame.columns:
        raise KeyError(f"{station}: missing date column {date_column!r}")
    if value_column not in frame.columns:
        raise KeyError(f"{station}: missing value column {value_column!r}")

    dates = pd.to_datetime(frame[date_column], errors="raise")
    if dates.duplicated().any():
        duplicated = dates[dates.duplicated()].unique()[:5]
        raise ValueError(f"{station}: duplicate dates in source, e.g. {list(duplicated)}")

    series = pd.Series(
        pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(dates),
        name=value_column,
    ).sort_index()

    complete = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    reindexed = series.reindex(complete)
    return DailySeries(
        station=station,
        index=complete,
        values=reindexed.to_numpy(dtype=float),
        source_path=str(source_path),
    )


def load_daily_series(
    path: str | Path,
    *,
    station: str,
    date_column: str = "date",
    value_column: str = "pm10",
    freq: str = "D",
) -> DailySeries:
    """Read a daily CSV and return it on the complete calendar."""
    path = Path(path)
    frame = pd.read_csv(path, usecols=lambda c: c in {date_column, value_column})
    return reindex_to_complete_calendar(
        frame,
        station=station,
        date_column=date_column,
        value_column=value_column,
        source_path=str(path),
        freq=freq,
    )


def compress_time_dropping_gaps(series: DailySeries) -> np.ndarray:
    """Return the observed values with gaps removed — **sensitivity use only**.

    This reproduces the invalid historical estimator in which missing days are
    dropped before lag construction, so that "lag k" silently means "k observed
    steps" instead of "k calendar days". The canon marks it invalid as a
    primary estimator; it exists here only so the two treatments can be
    compared explicitly in ``missingness_sensitivity.csv``.
    """
    return series.values[series.observed]
