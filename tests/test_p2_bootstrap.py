"""Moving-block bootstrap tests.

Covers ``test_bootstrap_resamples_origin_vectors`` and
``test_bootstrap_blocks_shared_across_methods``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.p2_decomposition.bootstrap import (
    block_start_grid,
    moving_block_origin_indices,
    percentile_interval,
    replicate_means,
    run_moving_block_bootstrap,
)


def test_block_start_grid_bounds() -> None:
    np.testing.assert_array_equal(block_start_grid(10, 3), np.arange(8))
    with pytest.raises(ValueError, match="exceeds the number of origins"):
        block_start_grid(5, 6)
    with pytest.raises(ValueError, match="block_length must be >= 1"):
        block_start_grid(5, 0)


def test_bootstrap_resamples_origin_vectors() -> None:
    """Sampled positions arrive in contiguous runs of the block length."""
    rng = np.random.default_rng(1)
    n_origins, block_length = 100, 7
    idx = moving_block_origin_indices(n_origins, block_length, rng)

    assert idx.size == n_origins
    assert idx.min() >= 0 and idx.max() < n_origins
    # Every complete block is a run of consecutive origin positions.
    n_complete = n_origins // block_length
    for block in range(n_complete):
        chunk = idx[block * block_length : (block + 1) * block_length]
        np.testing.assert_array_equal(np.diff(chunk), np.ones(block_length - 1, dtype=int))


def test_bootstrap_blocks_shared_across_methods() -> None:
    """One index draw per replicate drives every method and horizon column."""
    n_origins, n_methods, n_horizons = 60, 4, 3
    rng = np.random.default_rng(2)
    # Column j of origin i is encoded as i so the resampled origin identity is
    # recoverable from any column.
    identity_matrix = np.tile(
        np.arange(n_origins, dtype=float)[:, None], (1, n_methods * n_horizons)
    )
    result = run_moving_block_bootstrap(
        identity_matrix,
        station="T",
        support_type="GLOBAL_COMMON",
        series_names=tuple(f"m{m}|h{h}" for m in range(n_methods) for h in range(n_horizons)),
        block_length=10,
        n_replicates=50,
        seed=99,
        chunk_size=10,
    )
    means = result.replicate_means
    # Identical across columns => identical resampled origins for every method.
    assert np.allclose(means, means[:, [0]])
    # And the draws actually vary between replicates.
    assert means[:, 0].std() > 0.0


def test_bootstrap_is_deterministic_under_the_seed() -> None:
    values = np.random.default_rng(3).normal(size=(80, 6))
    names = tuple(f"s{i}" for i in range(6))
    kwargs = dict(
        station="T",
        support_type="GLOBAL_COMMON",
        series_names=names,
        block_length=14,
        n_replicates=40,
        seed=20260806,
        chunk_size=13,
    )
    first = run_moving_block_bootstrap(values, **kwargs)
    second = run_moving_block_bootstrap(values, **kwargs)
    np.testing.assert_allclose(first.replicate_means, second.replicate_means)


def test_bootstrap_preserves_pairing_of_missing_entries() -> None:
    """A NaN entry contributes to no replicate mean and shifts no other column."""
    values = np.ones((30, 2))
    values[:, 1] = 2.0
    values[0, 1] = np.nan
    result = run_moving_block_bootstrap(
        values,
        station="T",
        support_type="GLOBAL_COMMON",
        series_names=("a", "b"),
        block_length=5,
        n_replicates=25,
        seed=4,
    )
    np.testing.assert_allclose(result.replicate_means[:, 0], 1.0)
    np.testing.assert_allclose(result.replicate_means[:, 1], 2.0)
    np.testing.assert_allclose(result.effective_counts, [30.0, 29.0])


def test_replicate_means_use_occurrence_counts() -> None:
    """The count-weighted product equals the explicit resampled mean."""
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    mask = np.ones_like(values)
    idx = np.array([0, 0, 2])
    counts = np.bincount(idx, minlength=3).astype(float)[None, :]
    means = replicate_means(counts, values, mask)
    np.testing.assert_allclose(means[0], values[idx].mean(axis=0))


def test_percentile_interval_covers_the_requested_level() -> None:
    samples = np.random.default_rng(8).normal(size=(20000, 1))
    lower, upper = percentile_interval(samples, 0.95)
    assert lower[0] == pytest.approx(-1.96, abs=0.1)
    assert upper[0] == pytest.approx(1.96, abs=0.1)


def test_bootstrap_rejects_unsupported_interval_method() -> None:
    with pytest.raises(ValueError, match="unsupported interval_method"):
        run_moving_block_bootstrap(
            np.ones((10, 1)),
            station="T",
            support_type="GLOBAL_COMMON",
            series_names=("a",),
            block_length=2,
            n_replicates=2,
            seed=1,
            interval_method="bca",
        )
