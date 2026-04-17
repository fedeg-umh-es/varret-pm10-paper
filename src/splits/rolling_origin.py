"""Leakage-free rolling-origin split generation for P33."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RollingOriginFold:
    """Traceable description of one rolling-origin fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    origin_date: pd.Timestamp
    horizon: int
    target_date: pd.Timestamp
    train_indices: list[int]
    test_index: int


def generate_rolling_origin_folds(
    df: pd.DataFrame,
    date_column: str = "date",
    min_train_size: int = 60,
    max_horizon: int = 7,
    step_size: int = 1,
) -> list[RollingOriginFold]:
    """Generate leakage-free rolling-origin folds up to the requested horizon."""
    if max_horizon < 1 or max_horizon > 7:
        raise ValueError("max_horizon must be between 1 and 7 for P33.")
    if min_train_size < 1:
        raise ValueError("min_train_size must be positive.")
    if step_size < 1:
        raise ValueError("step_size must be positive.")

    ordered = df.reset_index(drop=True).copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column])
    ordered = ordered.sort_values(date_column).reset_index(drop=True)

    folds: list[RollingOriginFold] = []
    fold_id = 0
    final_origin = len(ordered) - max_horizon

    for origin_idx in range(min_train_size - 1, final_origin, step_size):
        train_indices = list(range(origin_idx + 1))
        origin_date = ordered.loc[origin_idx, date_column]
        for horizon in range(1, max_horizon + 1):
            test_index = origin_idx + horizon
            if test_index >= len(ordered):
                break
            folds.append(
                RollingOriginFold(
                    fold=fold_id,
                    train_start=ordered.loc[train_indices[0], date_column],
                    train_end=ordered.loc[train_indices[-1], date_column],
                    origin_date=origin_date,
                    horizon=horizon,
                    target_date=ordered.loc[test_index, date_column],
                    train_indices=train_indices,
                    test_index=test_index,
                )
            )
            fold_id += 1
    return folds
