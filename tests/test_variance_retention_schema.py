"""Schema regression tests for variance-retention diagnostics."""

import pandas as pd

from src.data.schema import VARIANCE_SUMMARY_COLUMNS
from src.diagnostics.variance import build_variance_retention_summary


def test_variance_retention_summary_output_columns_match_contract() -> None:
    """The diagnostic table must preserve the exact post-evaluation schema."""
    predictions_df = pd.DataFrame(
        {
            "dataset": ["e1_rr_daily"] * 4,
            "model": ["ridge_direct"] * 4,
            "horizon": [1] * 4,
            "y_true": [10.0, 20.0, 30.0, 40.0],
            "y_pred": [12.0, 18.0, 29.0, 41.0],
        }
    )
    skill_df = pd.DataFrame(
        {
            "dataset": ["e1_rr_daily"],
            "model": ["ridge_direct"],
            "horizon": [1],
            "skill": [0.25],
            "mae_skill": [0.15],
        }
    )

    summary = build_variance_retention_summary(predictions_df, skill_df)

    expected_columns = [
        "dataset",
        "model",
        "horizon",
        "n",
        "skill",
        "mae_skill",
        "alpha",
        "alpha_ci_low",
        "alpha_ci_high",
        "skill_vp",
        "collapse_flag",
        "inflation_flag",
        "near_ideal_flag",
        "low_sample_flag",
    ]

    assert VARIANCE_SUMMARY_COLUMNS == expected_columns
    assert list(summary.columns) == expected_columns
    assert int(summary.loc[0, "n"]) == 4
    assert float(summary.loc[0, "mae_skill"]) == 0.15
    assert pd.notna(summary.loc[0, "alpha_ci_low"])
    assert pd.notna(summary.loc[0, "alpha_ci_high"])
    assert bool(summary.loc[0, "low_sample_flag"]) is True


def test_variance_retention_summary_backfills_missing_mae_skill() -> None:
    """Older skill tables without MAE skill remain readable."""
    predictions_df = pd.DataFrame(
        {
            "dataset": ["e1_rr_daily"] * 4,
            "model": ["ridge_direct"] * 4,
            "horizon": [1] * 4,
            "y_true": [10.0, 20.0, 30.0, 40.0],
            "y_pred": [12.0, 18.0, 29.0, 41.0],
        }
    )
    skill_df = pd.DataFrame(
        {
            "dataset": ["e1_rr_daily"],
            "model": ["ridge_direct"],
            "horizon": [1],
            "skill": [0.25],
        }
    )

    summary = build_variance_retention_summary(predictions_df, skill_df)

    assert "mae_skill" in summary.columns
    assert pd.isna(summary.loc[0, "mae_skill"])
