"""Phase-5 guard: distinguish variance ratio from standard-deviation ratio,
and pin the production alpha (src/diagnostics/variance.py) to the VARIANCE
ratio that Paper A reports.

Run: python3 audit/paper_a_alpha_var_sd/test_alpha_var_vs_sd.py
 or: python3 -m pytest audit/paper_a_alpha_var_sd/test_alpha_var_vs_sd.py
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

from src.diagnostics.variance import _compute_alpha


def _alpha_var(yt, yp):
    return np.var(np.asarray(yp, float), ddof=0) / np.var(np.asarray(yt, float), ddof=0)


def _alpha_sd(yt, yp):
    return np.std(np.asarray(yp, float), ddof=0) / np.std(np.asarray(yt, float), ddof=0)


def _grp(yt, yp):
    return pd.DataFrame({"y_true": yt, "y_pred": yp})


# ---- canonical separating example: std_ratio=2, variance_ratio=4 ----
def test_canonical_separation():
    yt = [0, 1, 2, 3]
    yp = [0, 2, 4, 6]
    assert math.isclose(_alpha_sd(yt, yp), 2.0, rel_tol=1e-12)
    assert math.isclose(_alpha_var(yt, yp), 4.0, rel_tol=1e-12)
    # production code must return the VARIANCE ratio (4), not the SD ratio (2)
    assert math.isclose(_compute_alpha(_grp(yt, yp)), 4.0, rel_tol=1e-12)


def test_identical_amplitude():
    yt = [1, 2, 3, 4]; yp = [1, 2, 3, 4]
    assert math.isclose(_compute_alpha(_grp(yt, yp)), 1.0, rel_tol=1e-12)


def test_reduced_amplitude_variance_below_sd():
    # smoothed forecast: half the SD -> a quarter of the variance
    yt = [-2, -1, 1, 2]; yp = [-1, -0.5, 0.5, 1]
    assert math.isclose(_alpha_sd(yt, yp), 0.5, rel_tol=1e-9)
    assert math.isclose(_compute_alpha(_grp(yt, yp)), 0.25, rel_tol=1e-9)
    # the variance ratio is strictly below the SD ratio for shrinking forecasts
    assert _compute_alpha(_grp(yt, yp)) < _alpha_sd(yt, yp)


def test_increased_amplitude():
    yt = [-1, -0.5, 0.5, 1]; yp = [-2, -1, 1, 2]
    assert math.isclose(_compute_alpha(_grp(yt, yp)), 4.0, rel_tol=1e-9)


def test_offset_without_dispersion_change():
    yt = [1, 2, 3, 4]; yp = [11, 12, 13, 14]  # +10 offset, same variance
    assert math.isclose(_compute_alpha(_grp(yt, yp)), 1.0, rel_tol=1e-12)


def test_constant_observation_returns_zero_guard():
    # observed variance 0 -> production returns 0.0 (documented guard)
    yt = [5, 5, 5, 5]; yp = [1, 2, 3, 4]
    assert _compute_alpha(_grp(yt, yp)) == 0.0


def test_constant_prediction_gives_zero_variance_ratio():
    yt = [1, 2, 3, 4]; yp = [7, 7, 7, 7]
    assert math.isclose(_compute_alpha(_grp(yt, yp)), 0.0, abs_tol=1e-12)


def test_small_sample():
    yt = [0.0, 4.0]; yp = [0.0, 2.0]  # sd_ratio 0.5 -> var_ratio 0.25
    assert math.isclose(_compute_alpha(_grp(yt, yp)), 0.25, rel_tol=1e-9)


def test_ddof_zero_convention():
    # ddof must be 0 (population). With ddof=1 the ratio is identical here,
    # but we assert the estimator matches population variance explicitly.
    yt = [0, 1, 2, 3, 4]; yp = [0, 2, 4, 6, 8]
    got = _compute_alpha(_grp(yt, yp))
    pop = np.var(yp, ddof=0) / np.var(yt, ddof=0)
    assert math.isclose(got, pop, rel_tol=1e-12)
    assert math.isclose(got, 4.0, rel_tol=1e-12)


def test_naming_variance_not_sd():
    # A shrinking forecast whose SD ratio rounds to ~0.55 must NOT be reported
    # as alpha~0.55; the variance ratio (~0.30) is what alpha means.
    rng = np.random.default_rng(0)
    yt = rng.normal(size=2000)
    yp = 0.55 * yt  # exact SD ratio 0.55
    a = _compute_alpha(_grp(yt, yp))
    assert math.isclose(a, 0.55 ** 2, rel_tol=1e-6)  # 0.3025, the variance ratio
    assert abs(a - 0.55) > 0.2  # decisively not the SD ratio


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
    print("ALL PASS")
