from __future__ import annotations

import pandas as pd
import pytest

from src.diagnostics.hstar import compute_hstar
from src.diagnostics.hstar import build_horizons_summary, build_skill_table, build_trajectory_summary
from src.diagnostics.lrs import LeakageRiskComponents, leakage_risk_label, leakage_risk_score
from src.diagnostics.skill_vp import summarize_skill_vp
from src.diagnostics.variance import (
    build_variance_retention_table,
    detect_variance_collapse,
    variance_diagnostic_flags,
)


def test_compute_hstar_strict() -> None:
    assert compute_hstar([0.3, 0.2, -0.1, 0.1], criterion="strict") == 2


def test_compute_hstar_relax() -> None:
    assert compute_hstar([0.3, -0.2, 0.1], criterion="relax") == 3


def test_build_skill_table() -> None:
    metrics_df = pd.DataFrame(
        [
            {"dataset": "pm10_example", "horizon": 1, "model": "persistence", "rmse": 2.0},
            {"dataset": "pm10_example", "horizon": 1, "model": "seasonal_persistence_7", "rmse": 1.5},
            {"dataset": "pm10_example", "horizon": 1, "model": "ridge", "rmse": 1.0},
            {"dataset": "pm10_example", "horizon": 2, "model": "persistence", "rmse": 4.0},
            {"dataset": "pm10_example", "horizon": 2, "model": "seasonal_persistence_7", "rmse": 3.0},
            {"dataset": "pm10_example", "horizon": 2, "model": "ridge", "rmse": 5.0},
        ]
    )

    skill_df = build_skill_table(metrics_df, baseline_model="seasonal_persistence_7")

    assert list(skill_df.columns) == [
        "dataset",
        "horizon",
        "model",
        "rmse",
        "baseline_model",
        "rmse_baseline",
        "skill_vs_baseline",
    ]
    assert skill_df.loc[:, "baseline_model"].unique().tolist() == ["seasonal_persistence_7"]
    assert skill_df.loc[
        skill_df["model"] == "seasonal_persistence_7", "skill_vs_baseline"
    ].tolist() == [0.0, 0.0]
    assert skill_df.loc[
        skill_df["model"] == "persistence", "skill_vs_baseline"
    ].tolist() == pytest.approx([-1 / 3, -1 / 3])
    assert skill_df.loc[
        skill_df["model"] == "ridge", "skill_vs_baseline"
    ].tolist() == pytest.approx([1 / 3, -2 / 3])


def test_build_horizons_summary() -> None:
    skill_df = pd.DataFrame(
        [
            {"dataset": "pm10_example", "horizon": 1, "model": "persistence", "baseline_model": "persistence", "skill_vs_baseline": 0.0},
            {"dataset": "pm10_example", "horizon": 2, "model": "persistence", "baseline_model": "persistence", "skill_vs_baseline": 0.0},
            {"dataset": "pm10_example", "horizon": 1, "model": "ridge", "baseline_model": "persistence", "skill_vs_baseline": 0.3},
            {"dataset": "pm10_example", "horizon": 2, "model": "ridge", "baseline_model": "persistence", "skill_vs_baseline": 0.2},
            {"dataset": "pm10_example", "horizon": 3, "model": "ridge", "baseline_model": "persistence", "skill_vs_baseline": -0.1},
            {"dataset": "pm10_example", "horizon": 4, "model": "ridge", "baseline_model": "persistence", "skill_vs_baseline": 0.1},
        ]
    )

    horizons_df = build_horizons_summary(skill_df)

    persistence_row = horizons_df.loc[horizons_df["model"] == "persistence"].iloc[0]
    ridge_row = horizons_df.loc[horizons_df["model"] == "ridge"].iloc[0]

    assert (persistence_row["H"], persistence_row["H_star_relax"], persistence_row["H_star_strict"]) == (0, 0, 0)
    assert (ridge_row["H"], ridge_row["H_star_relax"], ridge_row["H_star_strict"]) == (4, 4, 2)


def test_build_trajectory_summary() -> None:
    skill_df = pd.DataFrame(
        [
            {"dataset": "pm10_example", "horizon": 1, "model": "persistence", "baseline_model": "persistence", "skill_vs_baseline": 0.0},
            {"dataset": "pm10_example", "horizon": 2, "model": "persistence", "baseline_model": "persistence", "skill_vs_baseline": 0.0},
            {"dataset": "pm10_example", "horizon": 1, "model": "ridge", "baseline_model": "persistence", "skill_vs_baseline": 0.2},
            {"dataset": "pm10_example", "horizon": 2, "model": "ridge", "baseline_model": "persistence", "skill_vs_baseline": 0.5},
            {"dataset": "pm10_example", "horizon": 3, "model": "ridge", "baseline_model": "persistence", "skill_vs_baseline": 0.1},
        ]
    )

    summary_df = build_trajectory_summary(skill_df)
    persistence_row = summary_df.loc[summary_df["model"] == "persistence"].iloc[0]
    ridge_row = summary_df.loc[summary_df["model"] == "ridge"].iloc[0]

    assert list(summary_df.columns) == [
        "dataset",
        "model",
        "max_skill",
        "argmax_skill",
        "skill_range",
        "skill_drop_last_vs_peak",
    ]
    assert (
        persistence_row["max_skill"],
        persistence_row["argmax_skill"],
        persistence_row["skill_range"],
        persistence_row["skill_drop_last_vs_peak"],
    ) == (0.0, 1, 0.0, 0.0)
    assert ridge_row["max_skill"] == 0.5
    assert ridge_row["argmax_skill"] == 2
    assert ridge_row["skill_range"] == 0.4
    assert ridge_row["skill_drop_last_vs_peak"] == 0.4


def test_summarize_skill_vp_labels_collapse() -> None:
    result = summarize_skill_vp(skill=0.2, alpha=0.3)
    assert result.skill_vp < result.skill
    assert "collapse" in result.interpretation


def test_variance_flags_detect_collapse() -> None:
    flags = variance_diagnostic_flags(alpha=0.4)
    assert detect_variance_collapse(0.4) is True
    assert flags.collapse_flag is True
    assert flags.inflation_flag is False


def test_build_variance_retention_table() -> None:
    fold_metrics_df = pd.DataFrame(
        [
            {"dataset": "pm10_example", "model": "ridge", "horizon": 1, "y_true": 0.0, "y_pred": 0.0},
            {"dataset": "pm10_example", "model": "ridge", "horizon": 1, "y_true": 2.0, "y_pred": 1.0},
            {"dataset": "pm10_example", "model": "persistence", "horizon": 1, "y_true": 0.0, "y_pred": 0.0},
            {"dataset": "pm10_example", "model": "persistence", "horizon": 1, "y_true": 2.0, "y_pred": 2.0},
        ]
    )
    skill_df = pd.DataFrame(
        [
            {"dataset": "pm10_example", "model": "persistence", "horizon": 1, "skill_vs_baseline": 0.0},
            {"dataset": "pm10_example", "model": "ridge", "horizon": 1, "skill_vs_baseline": 0.5},
        ]
    )

    variance_df = build_variance_retention_table(fold_metrics_df, skill_df)

    assert list(variance_df.columns) == [
        "dataset",
        "model",
        "horizon",
        "skill",
        "alpha",
        "skill_vp",
        "collapse_flag",
        "inflation_flag",
        "near_ideal_flag",
    ]
    ridge_row = variance_df.loc[variance_df["model"] == "ridge"].iloc[0]
    assert ridge_row["skill"] == 0.5
    assert ridge_row["alpha"] == 0.25
    assert ridge_row["skill_vp"] == 0.125
    assert bool(ridge_row["collapse_flag"]) is True


def test_lrs_score_and_label() -> None:
    components = LeakageRiskComponents(0.1, 0.2, 0.3, 0.4)
    score = leakage_risk_score(components)
    assert score == 0.25
    assert leakage_risk_label(score) == "moderate"
