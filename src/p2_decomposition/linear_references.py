"""Persistence, direct AR(1) and direct AR(p) linear references.

Contract: ``P2_PAIRED_DECOMPOSITION_CONTRACT.md`` section 5, canon section 5.

The canonical reference is a **direct**, horizon-specific linear projection.
For each horizon ``h`` and lag order ``p`` a separate coefficient vector is
solved::

    Gamma_p[i, j] = gamma_hat(|i - j|)
    c_h           = [gamma_hat(h), gamma_hat(h+1), ..., gamma_hat(h+p-1)]^T
    beta_h        = solve(Gamma_p, c_h)
    yhat(t, h)    = mu_hat + beta_h^T (x_t - mu_hat * 1)

with ``x_t = [y_t, y_{t-1}, ..., y_{t-p+1}]^T``.

AR(1) is *not* a separate formula: it is this projection at ``p = 1``, which
reduces exactly to ``mu_hat + gamma_hat(h)/gamma_hat(0) * (y_t - mu_hat)``.
``rho_hat(1) ** h`` is never substituted.

Fail-closed policy
------------------
``Gamma_p`` is never regularised, pseudo-inverted, clipped or repaired. When
the system is not numerically valid the projection is refused and the cell is
recorded as invalid, with the diagnostics that justify the refusal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "NumericsPolicy",
    "ProjectionDiagnostics",
    "ProjectionSolution",
    "ar1_direct_coefficient",
    "direct_projection_coefficients",
    "direct_projection_forecast",
    "evaluate_gamma_matrix",
    "persistence_forecast",
    "toeplitz_from_gamma",
]


@dataclass(frozen=True)
class NumericsPolicy:
    """Declared numerical validity policy. ``regularisation`` is fixed at NONE."""

    max_condition_number: float = 1.0e10
    min_eigenvalue_strictly_positive: bool = True
    regularisation_policy: str = "NONE"

    def __post_init__(self) -> None:
        if self.regularisation_policy != "NONE":
            raise ValueError(
                "Only regularisation_policy='NONE' is authorised for this run. "
                "Introducing a stabilisation policy requires an explicit, "
                "deterministic, pre-registered decision plus sensitivity analysis "
                "(see P2_PAIRED_DECOMPOSITION_CONTRACT.md section 6)."
            )


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Numerical diagnostics for one ``Gamma_p``.

    Every field required by the contract is present. ``regularisation_type``
    and ``regularisation_value`` exist so that the absence of regularisation is
    an explicit, auditable record rather than an omission.
    """

    p: int
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    rank: int
    solver_status: str
    regularisation_type: str = "NONE"
    regularisation_value: float = 0.0
    pair_count_min: int = -1
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return self.solver_status == "VALID"


@dataclass(frozen=True)
class ProjectionSolution:
    """Coefficients for one ``(p, h)`` pair, or the reason there are none."""

    p: int
    horizon: int
    beta: np.ndarray | None
    solver_status: str
    diagnostics: ProjectionDiagnostics

    @property
    def is_valid(self) -> bool:
        return self.beta is not None and self.solver_status == "VALID"


def toeplitz_from_gamma(gamma: np.ndarray, p: int) -> np.ndarray:
    """Build the ``p x p`` symmetric Toeplitz matrix ``Gamma_p[i, j] = gamma(|i-j|)``."""
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if gamma.size < p:
        raise ValueError(f"gamma has {gamma.size} lags, need at least {p}")
    offsets = np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    return np.asarray(gamma, dtype=float)[offsets]


def evaluate_gamma_matrix(
    gamma: np.ndarray,
    p: int,
    *,
    policy: NumericsPolicy,
    pair_count_min: int = -1,
    max_lag_required: int | None = None,
    n_pairs: np.ndarray | None = None,
) -> ProjectionDiagnostics:
    """Compute eigen-diagnostics for ``Gamma_p`` and decide whether to solve.

    The matrix is declared ``VALID`` only when every required autocovariance is
    finite, the smallest eigenvalue is strictly positive (when the policy
    demands it), the matrix has full rank, and the condition number is within
    the declared ceiling. Anything else fails closed.
    """
    required = p if max_lag_required is None else max_lag_required + 1
    notes: list[str] = []

    if gamma.size < required or not np.all(np.isfinite(gamma[:required])):
        missing = [
            int(k)
            for k in range(min(required, gamma.size))
            if not np.isfinite(gamma[k])
        ]
        if gamma.size < required:
            notes.append(f"gamma covers {gamma.size} lags, {required} required")
        if missing:
            notes.append(f"non-finite gamma at lags {missing[:5]}")
        return ProjectionDiagnostics(
            p=p,
            min_eigenvalue=float("nan"),
            max_eigenvalue=float("nan"),
            condition_number=float("nan"),
            rank=-1,
            solver_status="INVALID_NON_FINITE_AUTOCOVARIANCE",
            pair_count_min=pair_count_min,
            notes=tuple(notes),
        )

    matrix = toeplitz_from_gamma(gamma, p)
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eig = float(eigenvalues[0])
    max_eig = float(eigenvalues[-1])
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(abs(max_eig) / abs(min_eig)) if min_eig != 0.0 else float("inf")

    status = "VALID"
    if policy.min_eigenvalue_strictly_positive and min_eig <= 0.0:
        # Pairwise deletion under missingness does not guarantee a positive
        # definite Toeplitz matrix. The contract forbids repairing it.
        status = "INVALID_NOT_POSITIVE_DEFINITE"
        notes.append(f"min eigenvalue {min_eig:.6g} <= 0")
    elif rank < p:
        status = "INVALID_RANK_DEFICIENT"
        notes.append(f"rank {rank} < p {p}")
    elif not np.isfinite(condition) or condition > policy.max_condition_number:
        status = "INVALID_ILL_CONDITIONED"
        notes.append(f"condition number {condition:.6g} > {policy.max_condition_number:.6g}")

    if n_pairs is not None and pair_count_min < 0:
        pair_count_min = int(np.min(n_pairs[:required]))

    return ProjectionDiagnostics(
        p=p,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        condition_number=condition,
        rank=rank,
        solver_status=status,
        pair_count_min=pair_count_min,
        notes=tuple(notes),
    )


def direct_projection_coefficients(
    gamma: np.ndarray,
    p: int,
    horizon: int,
    *,
    policy: NumericsPolicy,
    diagnostics: ProjectionDiagnostics | None = None,
    pair_count_min: int = -1,
) -> ProjectionSolution:
    """Solve ``Gamma_p beta_h = c_h`` for one horizon, or refuse.

    A separate system is solved for every horizon; no one-step solution is
    iterated.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    max_lag_required = horizon + p - 1
    if diagnostics is None:
        diagnostics = evaluate_gamma_matrix(
            gamma,
            p,
            policy=policy,
            pair_count_min=pair_count_min,
            max_lag_required=max_lag_required,
        )

    if gamma.size <= max_lag_required or not np.all(np.isfinite(gamma[: max_lag_required + 1])):
        return ProjectionSolution(
            p=p,
            horizon=horizon,
            beta=None,
            solver_status="INVALID_NON_FINITE_AUTOCOVARIANCE",
            diagnostics=diagnostics,
        )
    if not diagnostics.is_valid:
        return ProjectionSolution(
            p=p,
            horizon=horizon,
            beta=None,
            solver_status=diagnostics.solver_status,
            diagnostics=diagnostics,
        )

    matrix = toeplitz_from_gamma(gamma, p)
    c_h = np.asarray(gamma[horizon : horizon + p], dtype=float)
    try:
        beta = np.linalg.solve(matrix, c_h)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by diagnostics
        return ProjectionSolution(
            p=p,
            horizon=horizon,
            beta=None,
            solver_status=f"INVALID_SOLVER_ERROR:{type(exc).__name__}",
            diagnostics=diagnostics,
        )
    if not np.all(np.isfinite(beta)):
        return ProjectionSolution(
            p=p,
            horizon=horizon,
            beta=None,
            solver_status="INVALID_NON_FINITE_SOLUTION",
            diagnostics=diagnostics,
        )
    return ProjectionSolution(
        p=p, horizon=horizon, beta=beta, solver_status="VALID", diagnostics=diagnostics
    )


def ar1_direct_coefficient(gamma: np.ndarray, horizon: int) -> float:
    """``gamma_hat(h) / gamma_hat(0)`` — the closed form of the ``p = 1`` projection.

    Provided only so tests can assert that the general solver at ``p = 1``
    reproduces it exactly. The pipeline always goes through
    :func:`direct_projection_coefficients`.
    """
    if gamma[0] == 0.0 or not np.isfinite(gamma[0]):
        return float("nan")
    return float(gamma[horizon] / gamma[0])


def direct_projection_forecast(mu: float, beta: np.ndarray, lag_vector: np.ndarray) -> float:
    """``mu_hat + beta_h^T (x_t - mu_hat * 1)``."""
    if beta.shape != lag_vector.shape:
        raise ValueError(f"beta shape {beta.shape} != lag vector shape {lag_vector.shape}")
    return float(mu + float(np.dot(beta, lag_vector - mu)))


def persistence_forecast(y_origin: float) -> float:
    """``yhat_P(t, h) = y_t`` for every horizon."""
    return float(y_origin)
