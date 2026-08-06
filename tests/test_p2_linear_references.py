"""Direct AR(1)/AR(p) projection tests.

Covers ``test_ar1_is_p1_direct_projection``, ``test_arp_direct_projection_formula``,
``test_yule_walker_diagnostics_recorded`` and ``test_no_silent_regularisation``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.p2_decomposition.autocovariance import estimate_autocovariances
from src.p2_decomposition.calendar import DailySeries
from src.p2_decomposition.diagnostics import DiagnosticsCollector, diagnostics_record
from src.p2_decomposition.linear_references import (
    NumericsPolicy,
    ar1_direct_coefficient,
    direct_projection_coefficients,
    direct_projection_forecast,
    evaluate_gamma_matrix,
    persistence_forecast,
    toeplitz_from_gamma,
)


def ar1_gamma(max_lag: int, phi: float = 0.6, sigma2: float = 1.0) -> np.ndarray:
    """Population autocovariances of a stationary AR(1)."""
    gamma0 = sigma2 / (1.0 - phi**2)
    return np.array([gamma0 * phi**k for k in range(max_lag + 1)])


@pytest.fixture()
def policy() -> NumericsPolicy:
    return NumericsPolicy()


def test_toeplitz_structure() -> None:
    gamma = np.array([3.0, 2.0, 1.0])
    matrix = toeplitz_from_gamma(gamma, 3)
    np.testing.assert_allclose(
        matrix, [[3.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 3.0]]
    )
    assert np.allclose(matrix, matrix.T)


def test_ar1_is_p1_direct_projection(policy: NumericsPolicy) -> None:
    """AR(1) is the p = 1 case of the same solver, not a separate formula."""
    gamma = ar1_gamma(12, phi=0.55)
    for horizon in range(1, 8):
        solution = direct_projection_coefficients(gamma, 1, horizon, policy=policy)
        assert solution.is_valid
        assert solution.beta.shape == (1,)
        assert solution.beta[0] == pytest.approx(ar1_direct_coefficient(gamma, horizon))
        assert solution.beta[0] == pytest.approx(gamma[horizon] / gamma[0])


def test_ar1_does_not_use_rho1_to_the_h() -> None:
    """gamma(h)/gamma(0) must not be silently replaced by rho(1) ** h.

    On a population AR(1) the two coincide; on a series that is not AR(1) they
    do not, and the canon requires the projection coefficient.
    """
    gamma = np.array([4.0, 2.0, 1.9, 0.2, 0.05])  # deliberately not AR(1)-shaped
    policy = NumericsPolicy()
    rho1 = gamma[1] / gamma[0]
    for horizon in (2, 3, 4):
        solution = direct_projection_coefficients(gamma, 1, horizon, policy=policy)
        assert solution.beta[0] == pytest.approx(gamma[horizon] / gamma[0])
        assert solution.beta[0] != pytest.approx(rho1**horizon, abs=1e-6)


def test_arp_direct_projection_formula(policy: NumericsPolicy) -> None:
    """beta_h solves Gamma_p beta_h = c_h with c_h = [gamma(h) .. gamma(h+p-1)]."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=(60, 30))
    covariance = base.T @ base / 60.0
    gamma = np.array([np.mean(np.diag(covariance, k)) for k in range(30)])
    gamma[0] += 5.0  # ensure a comfortably positive definite Toeplitz matrix

    for p in (2, 7, 14):
        for horizon in (1, 3, 7):
            solution = direct_projection_coefficients(gamma, p, horizon, policy=policy)
            assert solution.is_valid
            matrix = toeplitz_from_gamma(gamma, p)
            c_h = gamma[horizon : horizon + p]
            np.testing.assert_allclose(matrix @ solution.beta, c_h, rtol=1e-9, atol=1e-9)
            assert solution.beta.shape == (p,)


def test_direct_projection_is_horizon_specific_not_iterated(policy: NumericsPolicy) -> None:
    """Each horizon gets its own solve; coefficients are not powers of a one-step fit."""
    gamma = ar1_gamma(20, phi=0.6)
    betas = {
        horizon: direct_projection_coefficients(gamma, 7, horizon, policy=policy).beta
        for horizon in (1, 2, 3)
    }
    assert not np.allclose(betas[1], betas[2])
    assert not np.allclose(betas[2], betas[3])


def test_direct_projection_forecast_formula() -> None:
    mu, beta = 10.0, np.array([0.5, 0.25])
    x = np.array([14.0, 6.0])
    expected = mu + 0.5 * (14.0 - mu) + 0.25 * (6.0 - mu)
    assert direct_projection_forecast(mu, beta, x) == pytest.approx(expected)


def test_persistence_forecast_is_the_origin_value() -> None:
    assert persistence_forecast(37.0) == 37.0


def test_ar1_recovers_the_true_ar1_coefficient_from_data() -> None:
    """End-to-end: estimate gamma from a simulated AR(1), recover phi ** h."""
    rng = np.random.default_rng(7)
    phi, n = 0.6, 20000
    values = np.empty(n)
    values[0] = rng.normal()
    for t in range(1, n):
        values[t] = phi * values[t - 1] + rng.normal()
    series = DailySeries(
        station="sim",
        index=pd.date_range("2000-01-01", periods=n, freq="D"),
        values=values,
        source_path="<sim>",
    )
    estimate = estimate_autocovariances(series, train_end_position=n - 1, max_lag=5)
    solution = direct_projection_coefficients(
        estimate.gamma, 1, 2, policy=NumericsPolicy()
    )
    assert solution.beta[0] == pytest.approx(phi**2, abs=0.03)


def test_yule_walker_diagnostics_recorded(policy: NumericsPolicy) -> None:
    """Every contract-required diagnostic field is populated and collected."""
    gamma = ar1_gamma(20, phi=0.6)
    n_pairs = np.full(21, 900)
    diagnostics = evaluate_gamma_matrix(
        gamma, 7, policy=policy, n_pairs=n_pairs, max_lag_required=13
    )

    for field in (
        "min_eigenvalue",
        "max_eigenvalue",
        "condition_number",
        "rank",
        "solver_status",
        "regularisation_type",
        "regularisation_value",
        "pair_count_min",
    ):
        assert hasattr(diagnostics, field)
    assert diagnostics.solver_status == "VALID"
    assert diagnostics.rank == 7
    assert diagnostics.min_eigenvalue > 0.0
    assert diagnostics.condition_number == pytest.approx(
        diagnostics.max_eigenvalue / diagnostics.min_eigenvalue
    )
    assert diagnostics.pair_count_min == 900
    assert diagnostics.regularisation_type == "NONE"
    assert diagnostics.regularisation_value == 0.0

    collector = DiagnosticsCollector()
    collector.add_matrix(
        diagnostics_record(
            diagnostics,
            station="T",
            fold_or_window_id="2020-01-01",
            origin_date=pd.Timestamp("2020-01-01"),
            horizons_solved=[1, 2, 3],
            horizons_refused=[],
        )
    )
    frame = collector.matrix_frame()
    assert len(frame) == 1
    for column in (
        "min_eigenvalue",
        "max_eigenvalue",
        "condition_number",
        "rank",
        "solver_status",
        "regularisation_type",
        "regularisation_value",
        "pair_count_min",
    ):
        assert column in frame.columns


def test_no_silent_regularisation() -> None:
    """A non positive definite Gamma_p fails closed; nothing is repaired."""
    # gamma(1) > gamma(0) cannot come from a valid autocovariance function.
    gamma = np.array([1.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01])
    policy = NumericsPolicy()
    diagnostics = evaluate_gamma_matrix(gamma, 2, policy=policy, max_lag_required=3)

    assert diagnostics.solver_status == "INVALID_NOT_POSITIVE_DEFINITE"
    assert diagnostics.min_eigenvalue < 0.0
    assert diagnostics.regularisation_type == "NONE"
    assert diagnostics.regularisation_value == 0.0

    solution = direct_projection_coefficients(gamma, 2, 1, policy=policy)
    assert solution.beta is None
    assert not solution.is_valid
    assert solution.solver_status == "INVALID_NOT_POSITIVE_DEFINITE"


def test_regularisation_policy_other_than_none_is_refused() -> None:
    with pytest.raises(ValueError, match="regularisation_policy='NONE'"):
        NumericsPolicy(regularisation_policy="RIDGE")


def test_non_finite_autocovariance_fails_closed() -> None:
    gamma = np.array([1.0, 0.5, np.nan, 0.1])
    policy = NumericsPolicy()
    diagnostics = evaluate_gamma_matrix(gamma, 2, policy=policy, max_lag_required=3)
    assert diagnostics.solver_status == "INVALID_NON_FINITE_AUTOCOVARIANCE"
    solution = direct_projection_coefficients(gamma, 2, 1, policy=policy)
    assert solution.beta is None


def test_ill_conditioned_matrix_is_refused() -> None:
    policy = NumericsPolicy(max_condition_number=10.0)
    gamma = ar1_gamma(20, phi=0.98)
    diagnostics = evaluate_gamma_matrix(gamma, 14, policy=policy, max_lag_required=20)
    assert diagnostics.solver_status == "INVALID_ILL_CONDITIONED"
    assert direct_projection_coefficients(gamma, 14, 1, policy=policy).beta is None
