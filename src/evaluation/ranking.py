"""Ranking helpers for forecast comparison."""

from __future__ import annotations

from collections.abc import Mapping

from .metrics import ranking_from_metric_dict


def rank_models_by_rmse(metric_by_model: Mapping[str, float]) -> list[tuple[str, float]]:
    """Rank models from lower RMSE to higher RMSE."""
    return ranking_from_metric_dict(metric_by_model=metric_by_model, ascending=True)
