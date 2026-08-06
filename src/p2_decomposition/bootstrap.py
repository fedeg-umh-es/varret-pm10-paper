"""Moving-block bootstrap over origin vectors.

Contract: ``P2_PAIRED_DECOMPOSITION_CONTRACT.md`` section 10.

The resampled unit is a **contiguous block of origin dates**. An origin carries,
jointly and intact, every available ``(method, horizon)`` loss it has. The same
sampled blocks are applied to every method and every horizon; nothing is
resampled independently by method, and nothing is pooled across stations.

Memory discipline
-----------------
Replicate loss matrices are never materialised. For each replicate the sampled
origin multiset is reduced to an occurrence-count vector, and replicate means
are obtained by two matrix products against the fixed origin-by-series value and
mask matrices. Replicates are processed in configured chunks, and only the
aggregated statistics survive.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BootstrapResult",
    "block_start_grid",
    "moving_block_origin_indices",
    "replicate_means",
    "run_moving_block_bootstrap",
]


def block_start_grid(n_origins: int, block_length: int) -> np.ndarray:
    """Valid start positions for a moving block of ``block_length`` origins."""
    if block_length < 1:
        raise ValueError(f"block_length must be >= 1, got {block_length}")
    if block_length > n_origins:
        raise ValueError(
            f"block_length {block_length} exceeds the number of origins {n_origins}"
        )
    return np.arange(n_origins - block_length + 1, dtype=np.int64)


def moving_block_origin_indices(
    n_origins: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw one moving-block resample of origin positions.

    ``ceil(n_origins / block_length)`` blocks are drawn with replacement from
    the grid of valid starts and concatenated, then truncated to exactly
    ``n_origins`` positions so every replicate has the same sample size.
    """
    n_starts = block_start_grid(n_origins, block_length).size
    n_blocks = int(np.ceil(n_origins / block_length))
    starts = rng.integers(0, n_starts, size=n_blocks, dtype=np.int64)
    offsets = np.arange(block_length, dtype=np.int64)
    return (starts[:, None] + offsets[None, :]).ravel()[:n_origins]


def replicate_means(
    counts: np.ndarray, values_filled: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Replicate means from origin occurrence counts.

    Parameters
    ----------
    counts:
        ``(n_replicates, n_origins)`` occurrence counts of each origin.
    values_filled:
        ``(n_origins, n_series)`` losses with unavailable entries set to zero.
    mask:
        ``(n_origins, n_series)`` 1.0 where the loss is available.

    Returns
    -------
    numpy.ndarray
        ``(n_replicates, n_series)`` means, ``NaN`` where a replicate contains
        no available case for that series.
    """
    sums = counts @ values_filled
    denominators = counts @ mask
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(denominators > 0, sums / denominators, np.nan)
    return means


@dataclass(frozen=True)
class BootstrapResult:
    """Aggregated bootstrap output for one station, support and block length."""

    station: str
    support_type: str
    block_length: int
    n_replicates: int
    n_origins: int
    seed: int
    confidence_level: float
    interval_method: str
    series_names: tuple[str, ...]
    #: ``(n_replicates, n_series)`` replicate means, retained only when the
    #: caller asks for them; otherwise an empty array.
    replicate_means: np.ndarray
    effective_counts: np.ndarray  # (n_series,) available cases in the point sample


def run_moving_block_bootstrap(
    values: np.ndarray,
    *,
    station: str,
    support_type: str,
    series_names: tuple[str, ...],
    block_length: int,
    n_replicates: int,
    seed: int,
    confidence_level: float = 0.95,
    interval_method: str = "percentile",
    chunk_size: int = 250,
    return_replicate_means: bool = True,
) -> BootstrapResult:
    """Run the moving-block bootstrap over origins.

    ``values`` is ``(n_origins, n_series)``; ``NaN`` marks a loss that does not
    exist for that origin (for instance a horizon whose target fell outside the
    evaluation period). The origin axis must be sorted by origin date so that a
    contiguous slice really is a contiguous stretch of time.
    """
    if interval_method != "percentile":
        raise ValueError(f"unsupported interval_method {interval_method!r}")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D (origins, series), got shape {values.shape}")
    n_origins, n_series = values.shape
    if n_series != len(series_names):
        raise ValueError(
            f"series_names has {len(series_names)} entries for {n_series} columns"
        )

    mask = np.isfinite(values).astype(float)
    values_filled = np.where(np.isfinite(values), values, 0.0)

    rng = np.random.default_rng(seed)
    collected: list[np.ndarray] = []
    remaining = n_replicates
    while remaining > 0:
        batch = int(min(chunk_size, remaining))
        counts = np.zeros((batch, n_origins), dtype=float)
        for row in range(batch):
            # One shared index draw per replicate: the same origins, hence the
            # same blocks, are applied to every method and horizon column.
            idx = moving_block_origin_indices(n_origins, block_length, rng)
            counts[row] = np.bincount(idx, minlength=n_origins)
        collected.append(replicate_means(counts, values_filled, mask))
        remaining -= batch

    means = np.concatenate(collected, axis=0) if collected else np.empty((0, n_series))
    return BootstrapResult(
        station=station,
        support_type=support_type,
        block_length=int(block_length),
        n_replicates=int(n_replicates),
        n_origins=int(n_origins),
        seed=int(seed),
        confidence_level=float(confidence_level),
        interval_method=interval_method,
        series_names=tuple(series_names),
        replicate_means=means if return_replicate_means else np.empty((0, n_series)),
        effective_counts=mask.sum(axis=0),
    )


def percentile_interval(
    samples: np.ndarray, confidence_level: float
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided percentile interval, ignoring ``NaN`` replicates.

    A column that is ``NaN`` in every replicate — for instance a suppressed
    fraction — yields ``NaN`` bounds rather than an error, so a suppressed
    quantity stays visibly suppressed instead of silently acquiring an interval.
    """
    alpha = (1.0 - confidence_level) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lower = np.nanpercentile(samples, 100.0 * alpha, axis=0)
        upper = np.nanpercentile(samples, 100.0 * (1.0 - alpha), axis=0)
    return lower, upper
