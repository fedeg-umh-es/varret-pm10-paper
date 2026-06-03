from __future__ import annotations

import pytest

from src.evaluation.rolling_origin import generate_rolling_origin_folds


def test_generate_rolling_origin_folds_expanding() -> None:
    folds = generate_rolling_origin_folds(n_obs=10, min_train_size=4, horizon=2, step=1)
    assert len(folds) == 5
    first = folds[0]
    assert first.train_start == 0
    assert first.train_end == 3
    assert first.test_start == 4
    assert first.test_end == 5


def test_generate_rolling_origin_folds_rolling_window() -> None:
    folds = generate_rolling_origin_folds(
        n_obs=10,
        min_train_size=4,
        horizon=2,
        step=2,
        expanding=False,
    )
    assert folds[1].train_indices == [2, 3, 4, 5]
    assert folds[1].test_indices == [6, 7]


def test_generate_rolling_origin_folds_raises_when_not_enough_data() -> None:
    with pytest.raises(ValueError):
        generate_rolling_origin_folds(n_obs=5, min_train_size=4, horizon=2)
