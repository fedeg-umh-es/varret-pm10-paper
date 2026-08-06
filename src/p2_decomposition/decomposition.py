"""Paired losses, additive decomposition and the secondary normalised fraction.

Contract: ``P2_PAIRED_DECOMPOSITION_CONTRACT.md`` sections 8 and 11.

Components, all in MSE (never RMSE)::

    Delta_AR1   = L_P   - L_AR1
    Delta_mem   = L_AR1 - L_ARp
    Delta_res   = L_ARp - L_M
    Delta_total = L_P   - L_M

with the required identity ``Delta_total == Delta_AR1 + Delta_mem + Delta_res``
verified numerically against configured tolerances. Negative components are
retained; truncating them would hide exactly the cases the decomposition exists
to expose.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

__all__ = [
    "COMPONENT_NAMES",
    "FRACTION_STATUSES",
    "Components",
    "FractionResult",
    "IdentityCheck",
    "check_identity",
    "compute_components",
    "compute_components_arrays",
    "linear_fraction",
    "linear_fraction_arrays",
    "squared_errors",
]

COMPONENT_NAMES: tuple[str, ...] = ("delta_ar1", "delta_mem", "delta_res", "delta_total")

FRACTION_STATUSES: tuple[str, ...] = (
    "DEFINED_STABLE",
    "DEFINED_UNSTABLE",
    "SUPPRESSED_TOTAL_NONPOSITIVE",
    "SUPPRESSED_DENOMINATOR_NEAR_ZERO",
    "SUPPRESSED_INTERVAL_CROSSES_ZERO",
)


def squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Row-level squared error ``(y_true - y_pred) ** 2``, propagating ``NaN``."""
    return (np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2


@dataclass(frozen=True)
class Components:
    """The four paired MSE differences for one cell."""

    l_persistence: float
    l_ar1: float
    l_arp: float
    l_model: float
    delta_ar1: float
    delta_mem: float
    delta_res: float
    delta_total: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class IdentityCheck:
    """Outcome of the additive identity verification."""

    residual: float
    tolerance: float
    passed: bool
    atol: float
    rtol: float


def compute_components(
    l_persistence: float, l_ar1: float, l_arp: float, l_model: float
) -> Components:
    """Compute the four components for one ``(station, horizon, model, p)`` cell."""
    return Components(
        l_persistence=float(l_persistence),
        l_ar1=float(l_ar1),
        l_arp=float(l_arp),
        l_model=float(l_model),
        delta_ar1=float(l_persistence - l_ar1),
        delta_mem=float(l_ar1 - l_arp),
        delta_res=float(l_arp - l_model),
        delta_total=float(l_persistence - l_model),
    )


def compute_components_arrays(
    l_persistence: np.ndarray, l_ar1: np.ndarray, l_arp: np.ndarray, l_model: np.ndarray
) -> dict[str, np.ndarray]:
    """Vectorised component computation, used inside bootstrap replicates."""
    return {
        "delta_ar1": l_persistence - l_ar1,
        "delta_mem": l_ar1 - l_arp,
        "delta_res": l_arp - l_model,
        "delta_total": l_persistence - l_model,
    }


def check_identity(components: Components, *, atol: float, rtol: float) -> IdentityCheck:
    """Verify ``Delta_total == Delta_AR1 + Delta_mem + Delta_res``."""
    reconstructed = components.delta_ar1 + components.delta_mem + components.delta_res
    residual = float(components.delta_total - reconstructed)
    tolerance = float(atol + rtol * abs(components.delta_total))
    return IdentityCheck(
        residual=residual,
        tolerance=tolerance,
        passed=bool(abs(residual) <= tolerance),
        atol=float(atol),
        rtol=float(rtol),
    )


@dataclass(frozen=True)
class FractionResult:
    """Secondary normalised fraction with its suppression status."""

    value: float
    status: str
    denominator: float
    denominator_threshold: float


def linear_fraction(
    l_persistence: float,
    l_arp: float,
    l_model: float,
    *,
    abs_threshold: float,
    rel_threshold: float,
    interval_crosses_zero: bool | None = None,
    unstable: bool = False,
) -> FractionResult:
    """``pi_linear = (L_P - L_ARp) / (L_P - L_M)``, suppressed where undefined.

    Suppression order is fixed and evaluated before any value is computed, so
    the status never depends on how favourable the resulting number looks.
    The value is not clipped to ``[0, 1]``.
    """
    denominator = float(l_persistence - l_model)
    threshold = float(max(abs_threshold, rel_threshold * abs(l_persistence)))

    if abs(denominator) <= threshold:
        return FractionResult(float("nan"), "SUPPRESSED_DENOMINATOR_NEAR_ZERO", denominator, threshold)
    if denominator <= 0.0:
        return FractionResult(float("nan"), "SUPPRESSED_TOTAL_NONPOSITIVE", denominator, threshold)
    if interval_crosses_zero:
        return FractionResult(
            float("nan"), "SUPPRESSED_INTERVAL_CROSSES_ZERO", denominator, threshold
        )

    value = float((l_persistence - l_arp) / denominator)
    status = "DEFINED_UNSTABLE" if unstable else "DEFINED_STABLE"
    return FractionResult(value, status, denominator, threshold)


def linear_fraction_arrays(
    l_persistence: np.ndarray,
    l_arp: np.ndarray,
    l_model: np.ndarray,
    *,
    abs_threshold: float,
    rel_threshold: float,
) -> np.ndarray:
    """Vectorised fraction for bootstrap replicates; ``NaN`` where suppressed."""
    denominator = l_persistence - l_model
    threshold = np.maximum(abs_threshold, rel_threshold * np.abs(l_persistence))
    defined = (np.abs(denominator) > threshold) & (denominator > 0.0)
    out = np.full(denominator.shape, np.nan, dtype=float)
    np.divide(
        l_persistence - l_arp,
        denominator,
        out=out,
        where=defined,
    )
    return out
