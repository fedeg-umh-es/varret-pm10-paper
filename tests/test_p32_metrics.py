from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    ranking_from_metric_dict,
    relative_skill_vs_persistence,
    rmse,
    skill_vp,
    variance_ratio_alpha,
)


def test_rmse_is_zero_for_identical_series() -> None:
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_relative_skill_vs_persistence_positive_when_model_better() -> None:
    assert relative_skill_vs_persistence(4.0, 5.0) == pytest.approx(0.2)


def test_variance_ratio_alpha_detects_collapse() -> None:
    alpha = variance_ratio_alpha([0.0, 2.0, 4.0], [1.0, 2.0, 3.0])
    assert alpha < 1.0


def test_skill_vp_penalizes_mismatch_in_variance() -> None:
    assert skill_vp(0.4, 0.25) == pytest.approx(0.1)


def test_ranking_from_metric_dict_orders_lower_values_first() -> None:
    ranking = ranking_from_metric_dict({"xgb": 5.0, "sarima": 4.0, "persist": 6.0})
    assert [name for name, _ in ranking] == ["sarima", "xgb", "persist"]
