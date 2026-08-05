"""Block bootstrap over contiguous, time-ordered origin dates (Fase 8).

block_length defaults to 14 (days) purely as a provisional placeholder; it
is not justified by any ACF, episode-duration, or dependence analysis in
this codebase. The returned result carries that caveat explicitly and the
caller must not present the resulting interval as scientifically valid
until block_length is justified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

DEFAULT_BLOCK_LENGTH = 14
DEFAULT_BLOCK_LENGTH_UNIT = "days"
UNJUSTIFIED_PLACEHOLDER = "PROVISIONAL_DEFAULT_NOT_JUSTIFIED_BY_ACF_OR_EPISODE_DURATION"
UNVALIDATED_WARNING = (
    "Confidence interval is NOT scientifically validated: block_length has "
    "not been justified by autocorrelation (ACF), episode-duration, or "
    "dependence-structure analysis."
)

_UNIT_TO_TIMEDELTA = {
    "days": pd.Timedelta(days=1),
    "hours": pd.Timedelta(hours=1),
}


@dataclass
class BootstrapResult:
    statistic_estimate: float
    ci_low: float
    ci_high: float
    alpha: float
    block_length: int
    block_length_unit: str
    block_length_justification: str
    random_seed: int
    n_bootstrap: int
    warning: str

    def to_manifest_fields(self) -> dict:
        return {
            "block_length": self.block_length,
            "block_length_unit": self.block_length_unit,
            "block_length_justification": self.block_length_justification,
            "random_seed": self.random_seed,
        }


def _check_contiguous(origin_dates: pd.Series, unit: str) -> None:
    if unit not in _UNIT_TO_TIMEDELTA:
        raise ValueError(f"Unsupported block_length_unit '{unit}'; expected one of {list(_UNIT_TO_TIMEDELTA)}.")
    step = _UNIT_TO_TIMEDELTA[unit]
    diffs = origin_dates.diff().dropna()
    if not (diffs == step).all():
        n_gaps = int((diffs != step).sum())
        raise ValueError(
            f"origin_date series is not temporally contiguous at a {unit[:-1]} "
            f"resolution: {n_gaps} irregular gap(s) found. Block bootstrap "
            "requires a contiguous, gap-free, sorted origin_date index; "
            "resample or split into contiguous segments first."
        )


def block_bootstrap(
    origin_dates,
    values,
    *,
    statistic_fn: Callable[[np.ndarray], float],
    block_length: int = DEFAULT_BLOCK_LENGTH,
    block_length_unit: str = DEFAULT_BLOCK_LENGTH_UNIT,
    random_seed: int,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    block_length_justification: str | None = None,
    require_contiguous: bool = True,
) -> BootstrapResult:
    origin_dates = pd.Series(pd.to_datetime(pd.Series(origin_dates).reset_index(drop=True)))
    values = np.asarray(values, dtype=float)
    if len(origin_dates) != len(values):
        raise ValueError("origin_dates and values must have the same length.")
    if block_length < 1:
        raise ValueError("block_length must be >= 1.")

    order = np.argsort(origin_dates.values, kind="stable")
    origin_sorted = origin_dates.iloc[order].reset_index(drop=True)
    values_sorted = values[order]

    if require_contiguous:
        _check_contiguous(origin_sorted, block_length_unit)

    n = len(values_sorted)
    if n < block_length:
        raise ValueError(
            f"Not enough observations ({n}) for block_length={block_length}."
        )

    n_start_positions = n - block_length + 1
    n_blocks_needed = math.ceil(n / block_length)

    rng = np.random.default_rng(random_seed)

    boot_stats = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        starts = rng.integers(0, n_start_positions, size=n_blocks_needed)
        pieces = [values_sorted[s : s + block_length] for s in starts]
        resampled = np.concatenate(pieces)[:n]
        boot_stats[b] = statistic_fn(resampled)

    ci_low, ci_high = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    justification = block_length_justification or UNJUSTIFIED_PLACEHOLDER

    return BootstrapResult(
        statistic_estimate=float(statistic_fn(values_sorted)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        alpha=alpha,
        block_length=block_length,
        block_length_unit=block_length_unit,
        block_length_justification=justification,
        random_seed=random_seed,
        n_bootstrap=n_bootstrap,
        warning=UNVALIDATED_WARNING if justification == UNJUSTIFIED_PLACEHOLDER else "",
    )
