import numpy as np
import pandas as pd
import pytest

from src.evaluation.p4_exceedance.bootstrap import block_bootstrap, UNJUSTIFIED_PLACEHOLDER


def _contiguous_daily(n=60, seed=0):
    rng = np.random.default_rng(seed)
    origin_dates = pd.date_range("2020-01-01", periods=n, freq="D")
    values = rng.normal(0, 1, size=n)
    return origin_dates, values


class TestScenario16ReproducibleSeed:
    def test_same_seed_gives_identical_result(self):
        origin_dates, values = _contiguous_daily()
        r1 = block_bootstrap(
            origin_dates, values, statistic_fn=np.mean, block_length=7,
            random_seed=42, n_bootstrap=200,
        )
        r2 = block_bootstrap(
            origin_dates, values, statistic_fn=np.mean, block_length=7,
            random_seed=42, n_bootstrap=200,
        )
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high

    def test_different_seed_can_give_different_result(self):
        origin_dates, values = _contiguous_daily()
        r1 = block_bootstrap(
            origin_dates, values, statistic_fn=np.mean, block_length=7,
            random_seed=1, n_bootstrap=200,
        )
        r2 = block_bootstrap(
            origin_dates, values, statistic_fn=np.mean, block_length=7,
            random_seed=2, n_bootstrap=200,
        )
        assert (r1.ci_low, r1.ci_high) != (r2.ci_low, r2.ci_high)

    def test_random_seed_recorded_on_result(self):
        origin_dates, values = _contiguous_daily()
        result = block_bootstrap(
            origin_dates, values, statistic_fn=np.mean, block_length=7,
            random_seed=7, n_bootstrap=50,
        )
        assert result.random_seed == 7


class TestScenario17ContiguousOrderedBlocks:
    def test_out_of_order_input_is_sorted_before_blocking(self):
        origin_dates, values = _contiguous_daily(n=20)
        shuffled_idx = np.random.default_rng(0).permutation(len(origin_dates))
        result = block_bootstrap(
            origin_dates[shuffled_idx], values[shuffled_idx],
            statistic_fn=np.mean, block_length=5, random_seed=3, n_bootstrap=50,
        )
        assert np.isfinite(result.statistic_estimate)

    def test_non_contiguous_gap_rejected_by_default(self):
        origin_dates = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-10", "2020-01-11"]
        )
        values = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="not temporally contiguous"):
            block_bootstrap(
                origin_dates, values, statistic_fn=np.mean, block_length=2,
                random_seed=1, n_bootstrap=10,
            )

    def test_contiguity_check_can_be_disabled_explicitly(self):
        origin_dates = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-10", "2020-01-11"]
        )
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = block_bootstrap(
            origin_dates, values, statistic_fn=np.mean, block_length=2,
            random_seed=1, n_bootstrap=10, require_contiguous=False,
        )
        assert np.isfinite(result.statistic_estimate)


class TestBlockLengthJustification:
    def test_default_block_length_is_14_and_flagged_unjustified(self):
        origin_dates, values = _contiguous_daily(n=40)
        result = block_bootstrap(
            origin_dates, values, statistic_fn=np.mean, random_seed=1, n_bootstrap=20,
        )
        assert result.block_length == 14
        assert result.block_length_justification == UNJUSTIFIED_PLACEHOLDER
        assert result.warning != ""

    def test_explicit_justification_clears_warning(self):
        origin_dates, values = _contiguous_daily(n=40)
        result = block_bootstrap(
            origin_dates, values, statistic_fn=np.mean, block_length=10,
            random_seed=1, n_bootstrap=20,
            block_length_justification="Justified by ACF decorrelation length of 10 days.",
        )
        assert result.block_length_justification.startswith("Justified")
        assert result.warning == ""

    def test_block_length_larger_than_series_rejected(self):
        origin_dates, values = _contiguous_daily(n=5)
        with pytest.raises(ValueError, match="Not enough observations"):
            block_bootstrap(
                origin_dates, values, statistic_fn=np.mean, block_length=14,
                random_seed=1, n_bootstrap=10,
            )
