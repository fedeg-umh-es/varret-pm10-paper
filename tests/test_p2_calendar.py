"""Calendar, missingness and train-only estimation tests for P2.

Covers the required tests ``test_complete_calendar_preserved``,
``test_missing_dates_not_compressed``, ``test_train_only_autocovariance`` and
``test_temporal_order``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.p2_decomposition.autocovariance import (
    estimate_autocovariances,
    estimate_autocovariances_compressed_time,
    training_end_position,
)
from src.p2_decomposition.calendar import (
    DailySeries,
    compress_time_dropping_gaps,
    load_daily_series,
    reindex_to_complete_calendar,
)


@pytest.fixture()
def gapped_frame() -> pd.DataFrame:
    """Five calendar days with 2020-01-03 and 2020-01-04 absent from the source."""
    return pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-06"],
            "pm10": [10.0, 12.0, 20.0, 22.0],
        }
    )


def test_complete_calendar_preserved(gapped_frame: pd.DataFrame) -> None:
    series = reindex_to_complete_calendar(gapped_frame, station="T", value_column="pm10")

    assert len(series.index) == 6
    assert list(series.index) == list(pd.date_range("2020-01-01", "2020-01-06", freq="D"))
    # The two absent source dates now exist on the calendar, carrying NaN.
    assert np.isnan(series.values[2])
    assert np.isnan(series.values[3])
    assert series.n_observed == 4
    assert series.n_missing == 2
    series.assert_complete_calendar()


def test_complete_calendar_rejects_gapped_index() -> None:
    index = pd.DatetimeIndex(["2020-01-01", "2020-01-03"])
    with pytest.raises(ValueError, match="gaps"):
        DailySeries(station="T", index=index, values=np.array([1.0, 2.0]), source_path="x")


def test_complete_calendar_rejects_duplicate_dates() -> None:
    frame = pd.DataFrame({"date": ["2020-01-01", "2020-01-01"], "pm10": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate dates"):
        reindex_to_complete_calendar(frame, station="T")


def test_missing_dates_not_compressed(gapped_frame: pd.DataFrame) -> None:
    """Dropping gaps changes what "lag k" means; the primary path must not do it."""
    series = reindex_to_complete_calendar(gapped_frame, station="T")

    compressed = compress_time_dropping_gaps(series)
    assert compressed.size == 4
    assert series.values.size == 6

    # On the calendar, position 1 (2020-01-02) and position 2 (2020-01-03) are one
    # day apart but the second is missing, so lag 1 has no pair there. Under
    # compression, 2020-01-02 and 2020-01-05 would be adjacent and would be
    # counted as a lag-1 pair even though they are three calendar days apart.
    aware = estimate_autocovariances(series, train_end_position=5, max_lag=1)
    naive = estimate_autocovariances_compressed_time(
        series, train_end_position=5, max_lag=1
    )
    assert aware.n_pairs[1] == 2  # (01-02, 01-01) and (01-06, 01-05)
    assert naive.n_pairs[1] == 3  # additionally the spurious (01-05, 01-02)
    assert aware.estimator == "CALENDAR_ALIGNED_TRAIN_ONLY"
    assert naive.estimator == "COMPRESSED_TIME_INVALID_AS_PRIMARY"
    assert aware.gamma[1] != pytest.approx(naive.gamma[1])


def test_lag_vector_refuses_to_impute_missing_days() -> None:
    frame = pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02", "2020-01-04"], "pm10": [1.0, 2.0, 4.0]}
    )
    series = reindex_to_complete_calendar(frame, station="T")

    # Position 3 is 2020-01-04; the 3-lag window reaches back over the missing
    # 2020-01-03, so the vector is unavailable rather than shortened or filled.
    assert series.lag_vector(3, 3) is None
    assert series.lag_vector(3, 1) is not None
    np.testing.assert_allclose(series.lag_vector(1, 2), [2.0, 1.0])


def test_train_only_autocovariance() -> None:
    """Estimates must be identical whether or not post-origin data exists."""
    rng = np.random.default_rng(11)
    values = rng.normal(size=400)
    index = pd.date_range("2020-01-01", periods=400, freq="D")
    full = DailySeries(station="T", index=index, values=values, source_path="x")
    truncated = DailySeries(
        station="T", index=index[:200], values=values[:200].copy(), source_path="x"
    )

    from_full = estimate_autocovariances(full, train_end_position=199, max_lag=10)
    from_truncated = estimate_autocovariances(truncated, train_end_position=199, max_lag=10)

    assert from_full.mu == pytest.approx(from_truncated.mu)
    np.testing.assert_allclose(from_full.gamma, from_truncated.gamma)
    np.testing.assert_array_equal(from_full.n_pairs, from_truncated.n_pairs)

    # Mutating the future must not move a single estimate.
    poisoned = values.copy()
    poisoned[200:] = 1e6
    poisoned_series = DailySeries(
        station="T", index=index, values=poisoned, source_path="x"
    )
    from_poisoned = estimate_autocovariances(
        poisoned_series, train_end_position=199, max_lag=10
    )
    np.testing.assert_allclose(from_full.gamma, from_poisoned.gamma)


def test_temporal_order() -> None:
    """max(train_date) < origin_date < target_date is enforced structurally."""
    assert training_end_position(100, 1) == 99
    assert training_end_position(100, 7) == 93
    with pytest.raises(ValueError, match="max\\(train_date\\) < origin_date"):
        training_end_position(100, 0)


def test_autocovariance_reports_pair_counts_and_insufficiency() -> None:
    index = pd.date_range("2020-01-01", periods=50, freq="D")
    values = np.arange(50, dtype=float)
    values[10:40] = np.nan
    series = DailySeries(station="T", index=index, values=values, source_path="x")

    estimate = estimate_autocovariances(
        series, train_end_position=49, max_lag=5, min_pairs_per_lag=100
    )
    assert estimate.status == "INSUFFICIENT_PAIRS_AT_LAG"
    assert not estimate.is_usable
    assert np.isnan(estimate.gamma).all()
    assert estimate.n_pairs[0] == 20


def test_load_daily_series_from_repository_input() -> None:
    """The real Elche artefact loads onto a gap-free calendar with NaN preserved."""
    series = load_daily_series(
        "data/raw/pm10_daily.csv", station="Elche", value_column="pm10"
    )
    series.assert_complete_calendar()
    assert series.n_missing > 0  # this station genuinely has an incomplete calendar
    assert series.n_observed + series.n_missing == len(series.index)
