"""Decomposition, identity and fraction-suppression tests.

Covers ``test_mse_identity``, ``test_negative_components_retained`` and
``test_fraction_suppression_rules``, plus the gate's refusal to declare
``P2_PAPER_GO``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.p2_decomposition.decomposition import (
    check_identity,
    compute_components,
    compute_components_arrays,
    linear_fraction,
    linear_fraction_arrays,
    squared_errors,
)
from src.p2_decomposition.gate import (
    GATE_CONDITIONS,
    GateCondition,
    GateEvaluationError,
    evaluate_gate,
    overall_status,
)

ATOL, RTOL = 1e-12, 1e-10


def test_squared_errors_are_mse_scale() -> None:
    errors = squared_errors(np.array([10.0, 20.0]), np.array([8.0, 23.0]))
    np.testing.assert_allclose(errors, [4.0, 9.0])


def test_mse_identity() -> None:
    """Delta_total == Delta_AR1 + Delta_mem + Delta_res across many random cells."""
    rng = np.random.default_rng(5)
    for _ in range(500):
        l_p, l_ar1, l_arp, l_m = rng.uniform(0.1, 500.0, size=4)
        components = compute_components(l_p, l_ar1, l_arp, l_m)
        identity = check_identity(components, atol=ATOL, rtol=RTOL)
        assert identity.passed
        assert abs(identity.residual) <= identity.tolerance


def test_mse_identity_vectorised_matches_scalar() -> None:
    rng = np.random.default_rng(6)
    l_p, l_ar1, l_arp, l_m = (rng.uniform(1.0, 100.0, size=64) for _ in range(4))
    arrays = compute_components_arrays(l_p, l_ar1, l_arp, l_m)
    np.testing.assert_allclose(
        arrays["delta_total"],
        arrays["delta_ar1"] + arrays["delta_mem"] + arrays["delta_res"],
        atol=1e-12,
    )


def test_mse_identity_detects_a_broken_component() -> None:
    components = compute_components(10.0, 9.0, 8.0, 7.0)
    broken = components.__class__(**{**components.as_dict(), "delta_mem": 0.5})
    assert not check_identity(broken, atol=ATOL, rtol=RTOL).passed


def test_negative_components_retained() -> None:
    """A reference that is worse than its predecessor produces a negative component."""
    # AR(p) is worse than AR(1) here, and the model is worse than persistence.
    components = compute_components(
        l_persistence=100.0, l_ar1=90.0, l_arp=95.0, l_model=110.0
    )
    assert components.delta_ar1 == pytest.approx(10.0)
    assert components.delta_mem == pytest.approx(-5.0)
    assert components.delta_res == pytest.approx(-15.0)
    assert components.delta_total == pytest.approx(-10.0)
    # Nothing is clipped at zero.
    assert components.delta_mem < 0.0
    assert components.delta_total < 0.0
    assert check_identity(components, atol=ATOL, rtol=RTOL).passed


def test_fraction_suppression_rules() -> None:
    kwargs = {"abs_threshold": 1e-8, "rel_threshold": 0.01}

    defined = linear_fraction(100.0, 80.0, 60.0, **kwargs)
    assert defined.status == "DEFINED_STABLE"
    assert defined.value == pytest.approx((100.0 - 80.0) / (100.0 - 60.0))

    # Denominator inside the declared near-zero band (1% of L_P = 1.0).
    near_zero = linear_fraction(100.0, 95.0, 99.5, **kwargs)
    assert near_zero.status == "SUPPRESSED_DENOMINATOR_NEAR_ZERO"
    assert np.isnan(near_zero.value)

    # Model worse than persistence: Delta_total < 0.
    nonpositive = linear_fraction(100.0, 95.0, 130.0, **kwargs)
    assert nonpositive.status == "SUPPRESSED_TOTAL_NONPOSITIVE"
    assert np.isnan(nonpositive.value)

    crossing = linear_fraction(100.0, 80.0, 60.0, interval_crosses_zero=True, **kwargs)
    assert crossing.status == "SUPPRESSED_INTERVAL_CROSSES_ZERO"

    unstable = linear_fraction(100.0, 80.0, 60.0, unstable=True, **kwargs)
    assert unstable.status == "DEFINED_UNSTABLE"


def test_fraction_is_not_clipped_to_the_unit_interval() -> None:
    # AR(p) beats the model outright: pi_linear exceeds 1 and must stay there.
    result = linear_fraction(100.0, 50.0, 80.0, abs_threshold=1e-8, rel_threshold=0.01)
    assert result.status == "DEFINED_STABLE"
    assert result.value == pytest.approx(50.0 / 20.0)
    assert result.value > 1.0

    # AR(p) worse than persistence: pi_linear is negative and must stay negative.
    negative = linear_fraction(100.0, 120.0, 60.0, abs_threshold=1e-8, rel_threshold=0.01)
    assert negative.value < 0.0


def test_fraction_arrays_match_scalar_suppression() -> None:
    l_p = np.array([100.0, 100.0, 100.0])
    l_arp = np.array([80.0, 95.0, 95.0])
    l_m = np.array([60.0, 99.5, 130.0])
    values = linear_fraction_arrays(l_p, l_arp, l_m, abs_threshold=1e-8, rel_threshold=0.01)
    assert values[0] == pytest.approx(0.5)
    assert np.isnan(values[1])
    assert np.isnan(values[2])


def _mechanical_conditions() -> list[GateCondition]:
    return [
        GateCondition(name, "PASS", [f"evidence/{name}.csv"], "ok")
        for name in GATE_CONDITIONS
        if name != "NON_TRIVIAL_INTERPRETATION_FOUND"
    ] + [
        GateCondition(
            "NON_TRIVIAL_INTERPRETATION_FOUND",
            "PENDING_SCIENTIFIC_REVIEW",
            ["reports/P2_PAIRED_DECOMPOSITION_REPORT.md"],
            "human judgement",
        )
    ]


def test_gate_never_declares_paper_go() -> None:
    payload = evaluate_gate(_mechanical_conditions())
    assert payload["_summary"]["p2_paper_status"] == "NO-GO PENDING SCIENTIFIC REVIEW"
    assert "P2_PAPER_GO" not in str(payload["_summary"]["p2_paper_status"])
    assert payload["NON_TRIVIAL_INTERPRETATION_FOUND"]["status"] == "PENDING_SCIENTIFIC_REVIEW"


def test_gate_refuses_to_settle_the_human_condition() -> None:
    with pytest.raises(GateEvaluationError):
        GateCondition("NON_TRIVIAL_INTERPRETATION_FOUND", "PASS", [], "")


def test_gate_requires_every_condition() -> None:
    incomplete = _mechanical_conditions()[:-2]
    with pytest.raises(ValueError, match="gate is incomplete"):
        evaluate_gate(incomplete)


def test_gate_reports_failures_and_blocks() -> None:
    conditions = _mechanical_conditions()
    conditions[0] = GateCondition(GATE_CONDITIONS[0], "FAIL", [], "broken")
    assert overall_status(conditions) == "NO-GO — MECHANICAL CONDITION FAILED"
    conditions[0] = GateCondition(GATE_CONDITIONS[0], "BLOCKED", [], "no inputs")
    assert overall_status(conditions) == "NO-GO — MECHANICAL CONDITION BLOCKED"
