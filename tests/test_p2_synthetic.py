"""Synthetic validation tests.

Covers ``test_synthetic_white_noise``, ``test_synthetic_ar1``,
``test_synthetic_arp`` and ``test_synthetic_missing_calendar``.

These scenarios validate implementation behaviour on processes whose memory is
known by construction. They establish no empirical PM10 claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.p2_decomposition.autocovariance import (
    estimate_autocovariances,
    estimate_autocovariances_compressed_time,
)
from src.p2_decomposition.synthetic import (
    evaluate_reference_ladder,
    make_ar1,
    make_ar_q,
    make_white_noise,
    punch_missing_days,
    run_all_scenarios,
    to_daily_series,
)

HORIZONS = (1, 3, 7)
P = 14


def test_synthetic_white_noise() -> None:
    """On white noise no reference reproducibly beats persistence."""
    rng = np.random.default_rng(20260806)
    series = to_daily_series(make_white_noise(6000, rng), "wn")
    rows = evaluate_reference_ladder(series, scenario="wn", horizons=HORIZONS, p=P)

    for row in rows:
        # Persistence on white noise costs 2 * sigma^2; AR references shrink to
        # the mean and cost about sigma^2. What must not happen is extra
        # structure appearing between AR(1) and AR(p).
        assert abs(row.delta_mem) / row.l_persistence < 0.05
        assert row.identity_passed


def test_synthetic_ar1() -> None:
    """On an AR(1) process a rich AR(p) adds little over AR(1)."""
    rng = np.random.default_rng(20260807)
    series = to_daily_series(make_ar1(8000, 0.6, rng), "ar1")
    rows = evaluate_reference_ladder(series, scenario="ar1", horizons=HORIZONS, p=P)

    for row in rows:
        assert abs(row.delta_mem) / row.l_persistence < 0.02
        assert row.identity_passed
    # AR(1) itself must beat persistence at h = 1 on a genuinely AR(1) series.
    assert rows[0].delta_ar1 > 0.0


def test_synthetic_arp() -> None:
    """On an AR(q>1) process AR(p) captures memory AR(1) cannot."""
    rng = np.random.default_rng(20260808)
    series = to_daily_series(make_ar_q(8000, [0.5, -0.3, 0.25], rng), "arq")
    rows = evaluate_reference_ladder(series, scenario="arq", horizons=HORIZONS, p=P)

    h1 = rows[0]
    assert h1.delta_mem > 0.0
    assert h1.delta_mem / h1.l_persistence > 0.01
    assert all(row.identity_passed for row in rows)


def test_synthetic_missing_calendar() -> None:
    """Calendar-aware estimation recovers phi; compressed time does not."""
    phi, n = 0.6, 8000
    complete = make_ar1(n, phi, np.random.default_rng(20260809))
    gapped = to_daily_series(
        punch_missing_days(complete, 0.25, np.random.default_rng(20260810)), "gap"
    )

    aware = estimate_autocovariances(gapped, train_end_position=n - 1, max_lag=5)
    compressed = estimate_autocovariances_compressed_time(
        gapped, train_end_position=n - 1, max_lag=5
    )
    rho_aware = aware.gamma[1] / aware.gamma[0]
    rho_compressed = compressed.gamma[1] / compressed.gamma[0]

    assert rho_aware == pytest.approx(phi, abs=0.03)
    # Compressing time turns some lag-1 "pairs" into pairs that are two or more
    # calendar days apart, biasing the estimate downward.
    assert rho_compressed < rho_aware - 0.02
    assert abs(rho_aware - phi) < abs(rho_compressed - phi)


def test_synthetic_missing_calendar_end_to_end_losses() -> None:
    """The two treatments produce different AR forecasts on the same support."""
    phi, n = 0.6, 6000
    complete = make_ar1(n, phi, np.random.default_rng(11))
    gapped = to_daily_series(
        punch_missing_days(complete, 0.25, np.random.default_rng(12)), "gap"
    )
    aware = evaluate_reference_ladder(gapped, scenario="aware", horizons=(1,), p=7)
    compressed = evaluate_reference_ladder(
        gapped, scenario="compressed", horizons=(1,), p=7, compressed_time=True
    )
    assert aware[0].n_cases == compressed[0].n_cases  # same paired support
    assert aware[0].l_arp != pytest.approx(compressed[0].l_arp)


def test_synthetic_identity_holds_everywhere() -> None:
    summary = run_all_scenarios(
        {
            "seed": 20260806,
            "n_observations": 2000,
            "ar1_phi": 0.6,
            "ar_q_coefficients": [0.5, -0.3, 0.25],
            "nonlinear_threshold_coefficients": [0.7, -0.4],
            "missing_fraction": 0.2,
            "white_noise_tolerance": 0.05,
        }
    )
    identity = summary["scenarios"]["identity"]
    assert identity["all_passed"]
    assert identity["max_abs_residual"] < 1e-9
    assert summary["scenarios"]["bootstrap_pairing"]["deterministic_under_seed"]
    assert summary["scenarios"]["bootstrap_pairing"]["sample_size_preserved"]
    # The caveat travels with the artefact so it cannot be read as an empirical claim.
    assert "not evidence of nonlinearity" in summary["caveat"]


def test_synthetic_nonlinear_residual_is_not_labelled_as_proof() -> None:
    summary = run_all_scenarios(
        {"seed": 42, "n_observations": 2000, "missing_fraction": 0.1}
    )
    nonlinear = summary["scenarios"]["nonlinear"]
    assert "does NOT prove nonlinearity" in nonlinear["expectation"]
