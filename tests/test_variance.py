"""Conceptual tests for variance-retention diagnostics."""

import pandas as pd

from src.diagnostics.variance import build_variance_retention_summary


def _build_skill_df(skill: float) -> pd.DataFrame:
    return pd.DataFrame([{"dataset": "pm10_daily", "model": "test_model", "horizon": 1, "skill": skill}])


def test_vr_near_100_when_predicted_variability_matches_observed() -> None:
    predictions = pd.DataFrame(
        {
            "dataset": ["pm10_daily"] * 4,
            "model": ["test_model"] * 4,
            "horizon": [1] * 4,
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [1.0, 2.0, 3.0, 4.0],
        }
    )
    summary = build_variance_retention_summary(predictions, _build_skill_df(0.2))
    assert abs(summary.loc[0, "vr"] - 100.0) < 1e-9


def test_vr_near_zero_when_predictions_are_flat() -> None:
    predictions = pd.DataFrame(
        {
            "dataset": ["pm10_daily"] * 4,
            "model": ["test_model"] * 4,
            "horizon": [1] * 4,
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [2.0, 2.0, 2.0, 2.0],
        }
    )
    summary = build_variance_retention_summary(predictions, _build_skill_df(0.2))
    assert summary.loc[0, "vr"] == 0.0
    assert bool(summary.loc[0, "collapse_flag"])


def test_inflation_flag_when_predicted_variance_is_too_large() -> None:
    predictions = pd.DataFrame(
        {
            "dataset": ["pm10_daily"] * 4,
            "model": ["test_model"] * 4,
            "horizon": [1] * 4,
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [-2.0, 1.0, 5.0, 9.0],
        }
    )
    summary = build_variance_retention_summary(predictions, _build_skill_df(0.2))
    assert bool(summary.loc[0, "inflation_flag"])


def test_near_ideal_flag_when_skill_positive_and_vr_near_100() -> None:
    predictions = pd.DataFrame(
        {
            "dataset": ["pm10_daily"] * 4,
            "model": ["test_model"] * 4,
            "horizon": [1] * 4,
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [1.1, 1.9, 3.1, 3.9],
        }
    )
    summary = build_variance_retention_summary(predictions, _build_skill_df(0.25))
    assert bool(summary.loc[0, "near_ideal_flag"])


def test_skill_vp_capped_when_variance_inflates() -> None:
    """skill_vp must not exceed skill even when vr > 100."""
    predictions = pd.DataFrame(
        {
            "dataset": ["pm10_daily"] * 4,
            "model": ["test_model"] * 4,
            "horizon": [1] * 4,
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [-3.0, 1.0, 7.0, 11.0],
        }
    )
    skill = 0.3
    summary = build_variance_retention_summary(predictions, _build_skill_df(skill))
    assert summary.loc[0, "skill_vp"] <= skill + 1e-9
