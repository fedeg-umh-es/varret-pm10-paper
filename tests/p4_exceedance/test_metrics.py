import numpy as np
import pytest

from src.evaluation.p4_exceedance.metrics import compute_full_event_metrics


class TestFullEventMetrics:
    def test_perfect_forecast_gives_ideal_scores(self):
        y_true = np.array([10.0, 60.0, 20.0, 80.0])
        y_pred = y_true.copy()
        result = compute_full_event_metrics(y_true, y_pred, threshold=50.0)
        assert result["hits"] == 2
        assert result["misses"] == 0
        assert result["false_alarms"] == 0
        assert result["hit_rate"] == pytest.approx(1.0)
        assert result["false_alarm_rate"] == pytest.approx(0.0)
        assert result["csi"] == pytest.approx(1.0)
        assert result["event_bias"] == pytest.approx(1.0)
        assert result["exceedance_intensity_error"] == pytest.approx(0.0)

    def test_all_misses_gives_zero_hit_rate(self):
        y_true = np.array([60.0, 80.0])
        y_pred = np.array([10.0, 20.0])
        result = compute_full_event_metrics(y_true, y_pred, threshold=50.0)
        assert result["hits"] == 0
        assert result["misses"] == 2
        assert result["hit_rate"] == pytest.approx(0.0)
        assert result["csi"] == pytest.approx(0.0)
        assert result["exceedance_intensity_error"] < 0  # under-forecast during events

    def test_no_true_events_gives_nan_event_bias(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([60.0, 5.0])
        result = compute_full_event_metrics(y_true, y_pred, threshold=50.0)
        assert result["n_events_true"] == 0
        assert np.isnan(result["event_bias"])
        assert np.isnan(result["exceedance_intensity_error"])

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            compute_full_event_metrics([1, 2], [1], threshold=1)
