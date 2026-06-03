"""Leakage-free rolling-origin split generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class RollingOriginFold:
    """Single leakage-free rolling-origin fold."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    horizon: int
    step: int

    @property
    def train_indices(self) -> list[int]:
        return list(range(self.train_start, self.train_end + 1))

    @property
    def test_indices(self) -> list[int]:
        return list(range(self.test_start, self.test_end + 1))


def generate_rolling_origin_folds(
    n_obs: int,
    min_train_size: int,
    horizon: int,
    step: int = 1,
    train_start: int = 0,
    expanding: bool = True,
) -> list[RollingOriginFold]:
    """Generate leakage-free rolling-origin folds.

    Parameters
    ----------
    n_obs:
        Total number of observations in temporal order.
    min_train_size:
        Number of initial observations required in the training window.
    horizon:
        Forecast horizon in number of observations.
    step:
        Shift applied to the forecast origin between folds.
    train_start:
        Inclusive start index for the first train window.
    expanding:
        If True, training starts at `train_start` for all folds.
        If False, the training window length stays fixed at `min_train_size`.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    if min_train_size <= 0:
        raise ValueError("min_train_size must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")
    if train_start < 0:
        raise ValueError("train_start must be non-negative.")

    first_train_end = train_start + min_train_size - 1
    first_test_start = first_train_end + 1
    first_test_end = first_test_start + horizon - 1
    if first_test_end >= n_obs:
        raise ValueError("Not enough observations for one full rolling-origin fold.")

    folds: list[RollingOriginFold] = []
    origin = first_test_start
    while True:
        test_start = origin
        test_end = test_start + horizon - 1
        if test_end >= n_obs:
            break

        if expanding:
            current_train_start = train_start
        else:
            current_train_start = test_start - min_train_size
        current_train_end = test_start - 1

        if current_train_start < 0:
            raise ValueError("Computed train_start is negative; check inputs.")
        if current_train_end < current_train_start:
            raise ValueError("Computed training window is invalid.")
        if current_train_end >= test_start:
            raise ValueError("Leakage detected: training overlaps test.")

        folds.append(
            RollingOriginFold(
                train_start=current_train_start,
                train_end=current_train_end,
                test_start=test_start,
                test_end=test_end,
                horizon=horizon,
                step=step,
            )
        )
        origin += step

    return folds


def iter_rolling_origin_folds(
    n_obs: int,
    min_train_size: int,
    horizon: int,
    step: int = 1,
    train_start: int = 0,
    expanding: bool = True,
) -> Iterator[RollingOriginFold]:
    """Yield rolling-origin folds lazily."""
    yield from generate_rolling_origin_folds(
        n_obs=n_obs,
        min_train_size=min_train_size,
        horizon=horizon,
        step=step,
        train_start=train_start,
        expanding=expanding,
    )
