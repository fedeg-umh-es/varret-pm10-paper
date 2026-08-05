import numpy as np
import pytest

from src.evaluation.p4_exceedance.threshold_sweep import (
    diagnostic_sweep,
    calibrated_threshold_result,
    regulatory_threshold_result,
)


def _synthetic_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.gamma(shape=2.0, scale=15.0, size=n)
    y_pred = y_true + rng.normal(0, 5, size=n)
    return y_true, y_pred


class TestScenario15ThresholdSweepPostHocLabel:
    def test_diagnostic_sweep_always_labeled_post_hoc(self):
        y_true, y_pred = _synthetic_series()
        result = diagnostic_sweep(y_true, y_pred, thresholds=[10, 20, 30, 40])
        assert result.threshold_mode == "POST_HOC_DIAGNOSTIC"
        assert result.usable_as_primary_estimate is False

    def test_diagnostic_sweep_records_selection_fields(self):
        y_true, y_pred = _synthetic_series()
        result = diagnostic_sweep(
            y_true, y_pred, thresholds=[10, 20, 30], selection_metric="csi",
            evaluation_period=("2020-01-01", "2020-06-30"),
        )
        fields = result.to_manifest_fields()
        assert fields["threshold_mode"] == "POST_HOC_DIAGNOSTIC"
        assert fields["threshold_source"] == "post_hoc_diagnostic_sweep"
        assert fields["evaluation_period"] == ["2020-01-01", "2020-06-30"]
        assert fields["calibration_period"] is None

    def test_unknown_selection_metric_rejected(self):
        y_true, y_pred = _synthetic_series()
        with pytest.raises(ValueError):
            diagnostic_sweep(y_true, y_pred, thresholds=[10], selection_metric="not_a_metric")


class TestRegulatoryThreshold:
    def test_fixed_threshold_is_usable_as_primary_estimate(self):
        y_true, y_pred = _synthetic_series()
        result = regulatory_threshold_result(y_true, y_pred, threshold=50.0)
        assert result.threshold_mode == "FIXED"
        assert result.usable_as_primary_estimate is True
        assert result.threshold == 50.0


class TestCalibratedThreshold:
    def test_calibration_before_evaluation_is_usable_as_primary(self):
        y_true, y_pred = _synthetic_series(seed=1)
        result = calibrated_threshold_result(
            y_true, y_pred, thresholds=[10, 20, 30],
            calibration_period=("2019-01-01", "2019-12-31"),
            evaluation_period=("2020-01-01", "2020-12-31"),
        )
        assert result.threshold_mode == "CALIBRATED"
        assert result.usable_as_primary_estimate is True

    def test_calibration_overlapping_evaluation_period_rejected(self):
        y_true, y_pred = _synthetic_series(seed=1)
        with pytest.raises(ValueError, match="cannot be labeled a valid calibration"):
            calibrated_threshold_result(
                y_true, y_pred, thresholds=[10, 20, 30],
                calibration_period=("2020-06-01", "2020-12-31"),
                evaluation_period=("2020-01-01", "2020-12-31"),
            )
