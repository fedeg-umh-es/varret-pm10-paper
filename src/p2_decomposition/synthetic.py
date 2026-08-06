"""Deterministic seeded synthetic validation of the decomposition machinery.

Contract: ``P2_PAIRED_DECOMPOSITION_CONTRACT.md`` section 12.

These simulations validate *implementation behaviour* on processes whose memory
structure is known by construction. They are entirely separate from the PM10
data and they establish no empirical PM10 claim. In particular, scenario 4
producing a positive residual component is a property of the simulated
generator, not evidence of nonlinearity in any real series.

All scenarios use a single train/evaluate split rather than per-origin
re-estimation: the object under test is the algebra and the estimator, and a
fixed split keeps the suite fast enough to run on every commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .autocovariance import (
    estimate_autocovariances,
    estimate_autocovariances_compressed_time,
)
from .calendar import DailySeries
from .decomposition import check_identity, compute_components
from .linear_references import (
    NumericsPolicy,
    direct_projection_coefficients,
    direct_projection_forecast,
)

__all__ = [
    "SyntheticEvaluation",
    "make_ar1",
    "make_ar_q",
    "make_nonlinear_threshold",
    "make_white_noise",
    "punch_missing_days",
    "run_all_scenarios",
    "to_daily_series",
    "evaluate_reference_ladder",
]

_START = pd.Timestamp("2000-01-01")


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def make_white_noise(n: int, rng: np.random.Generator, sigma: float = 1.0) -> np.ndarray:
    return rng.normal(0.0, sigma, size=n)


def make_ar1(n: int, phi: float, rng: np.random.Generator, sigma: float = 1.0) -> np.ndarray:
    innovations = rng.normal(0.0, sigma, size=n)
    out = np.empty(n)
    out[0] = innovations[0] / np.sqrt(max(1.0 - phi**2, 1e-12))
    for t in range(1, n):
        out[t] = phi * out[t - 1] + innovations[t]
    return out


def make_ar_q(
    n: int, coefficients: list[float], rng: np.random.Generator, sigma: float = 1.0
) -> np.ndarray:
    q = len(coefficients)
    coefficients = np.asarray(coefficients, dtype=float)
    innovations = rng.normal(0.0, sigma, size=n + q)
    out = np.zeros(n + q)
    for t in range(q, n + q):
        history = out[t - q : t][::-1]  # [y_{t-1}, ..., y_{t-q}]
        out[t] = float(np.dot(coefficients, history)) + innovations[t]
    return out[q:]


def make_nonlinear_threshold(
    n: int, coefficients: list[float], rng: np.random.Generator, sigma: float = 1.0
) -> np.ndarray:
    """Self-exciting threshold autoregression: sign of ``y_{t-1}`` flips the slope."""
    low, high = float(coefficients[0]), float(coefficients[1])
    innovations = rng.normal(0.0, sigma, size=n)
    out = np.zeros(n)
    for t in range(1, n):
        phi = low if out[t - 1] <= 0.0 else high
        out[t] = phi * out[t - 1] + innovations[t]
    return out


def punch_missing_days(
    values: np.ndarray, missing_fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Set a random subset of days to ``NaN``, preserving the calendar length."""
    out = values.astype(float).copy()
    n_missing = int(round(missing_fraction * out.size))
    if n_missing:
        positions = rng.choice(out.size, size=n_missing, replace=False)
        out[positions] = np.nan
    return out


def to_daily_series(values: np.ndarray, station: str) -> DailySeries:
    index = pd.date_range(_START, periods=len(values), freq="D")
    return DailySeries(
        station=station, index=index, values=np.asarray(values, dtype=float),
        source_path="<synthetic>",
    )


# ---------------------------------------------------------------------------
# Ladder evaluation on a synthetic series
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticEvaluation:
    """Losses and components for one synthetic scenario at one horizon."""

    scenario: str
    horizon: int
    p: int
    n_cases: int
    l_persistence: float
    l_ar1: float
    l_arp: float
    l_model: float
    delta_ar1: float
    delta_mem: float
    delta_res: float
    delta_total: float
    identity_residual: float
    identity_passed: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def evaluate_reference_ladder(
    series: DailySeries,
    *,
    scenario: str,
    horizons: tuple[int, ...],
    p: int,
    train_fraction: float = 0.5,
    policy: NumericsPolicy | None = None,
    model_values: np.ndarray | None = None,
    identity_atol: float = 1e-12,
    identity_rtol: float = 1e-10,
    compressed_time: bool = False,
    min_pairs_per_lag: int = 1,
) -> list[SyntheticEvaluation]:
    """Compute ``L_P``, ``L_AR1``, ``L_ARp`` and ``L_M`` on a paired support.

    ``model_values`` supplies an empirical-model forecast aligned with the
    series positions (``NaN`` where unavailable); when omitted, ``L_M`` is set
    equal to ``L_ARp`` so that ``Delta_res`` is exactly zero and the identity is
    still exercised.
    """
    policy = policy or NumericsPolicy()
    n = len(series.values)
    train_end = int(train_fraction * n) - 1
    max_lag = max(horizons) + p - 1

    estimator = (
        estimate_autocovariances_compressed_time
        if compressed_time
        else estimate_autocovariances
    )
    kwargs: dict[str, object] = {"train_end_position": train_end, "max_lag": max_lag}
    if not compressed_time:
        kwargs["min_pairs_per_lag"] = min_pairs_per_lag
    estimate = estimator(series, **kwargs)  # type: ignore[arg-type]

    results: list[SyntheticEvaluation] = []
    for horizon in horizons:
        solution_p = direct_projection_coefficients(
            estimate.gamma, p, horizon, policy=policy
        )
        solution_1 = direct_projection_coefficients(
            estimate.gamma, 1, horizon, policy=policy
        )
        if not (solution_p.is_valid and solution_1.is_valid):
            results.append(
                SyntheticEvaluation(
                    scenario=scenario,
                    horizon=horizon,
                    p=p,
                    n_cases=0,
                    l_persistence=float("nan"),
                    l_ar1=float("nan"),
                    l_arp=float("nan"),
                    l_model=float("nan"),
                    delta_ar1=float("nan"),
                    delta_mem=float("nan"),
                    delta_res=float("nan"),
                    delta_total=float("nan"),
                    identity_residual=float("nan"),
                    identity_passed=False,
                    notes=(
                        f"p={p}: {solution_p.solver_status}",
                        f"p=1: {solution_1.solver_status}",
                    ),
                )
            )
            continue

        persistence, ar1, arp, truth, model = [], [], [], [], []
        for origin in range(train_end + 1, n - horizon):
            target = origin + horizon
            y_target = series.values[target]
            if not np.isfinite(y_target):
                continue
            x_p = series.lag_vector(origin, p)
            x_1 = series.lag_vector(origin, 1)
            if x_p is None or x_1 is None:
                continue
            if model_values is not None and not np.isfinite(model_values[target]):
                continue
            truth.append(y_target)
            persistence.append(series.values[origin])
            ar1.append(direct_projection_forecast(estimate.mu, solution_1.beta, x_1))
            arp.append(direct_projection_forecast(estimate.mu, solution_p.beta, x_p))
            model.append(
                float(model_values[target]) if model_values is not None else np.nan
            )

        truth_a = np.asarray(truth, dtype=float)
        l_p = float(np.mean((truth_a - np.asarray(persistence)) ** 2))
        l_1 = float(np.mean((truth_a - np.asarray(ar1)) ** 2))
        l_p_order = float(np.mean((truth_a - np.asarray(arp)) ** 2))
        l_m = (
            float(np.mean((truth_a - np.asarray(model)) ** 2))
            if model_values is not None
            else l_p_order
        )

        components = compute_components(l_p, l_1, l_p_order, l_m)
        identity = check_identity(components, atol=identity_atol, rtol=identity_rtol)
        results.append(
            SyntheticEvaluation(
                scenario=scenario,
                horizon=horizon,
                p=p,
                n_cases=int(truth_a.size),
                l_persistence=l_p,
                l_ar1=l_1,
                l_arp=l_p_order,
                l_model=l_m,
                delta_ar1=components.delta_ar1,
                delta_mem=components.delta_mem,
                delta_res=components.delta_res,
                delta_total=components.delta_total,
                identity_residual=identity.residual,
                identity_passed=identity.passed,
            )
        )
    return results


def _fit_threshold_model_train_only(
    series: DailySeries, train_end: int, horizon: int
) -> np.ndarray:
    """Two-regime least-squares predictor fitted on training positions only."""
    y = series.values
    n = len(y)
    predictions = np.full(n, np.nan)
    origins = np.arange(1, n - horizon)
    usable = np.isfinite(y[origins]) & np.isfinite(y[origins + horizon])
    train = usable & (origins + horizon <= train_end)

    slopes: dict[bool, float] = {}
    for regime in (False, True):
        selector = train & ((y[origins] > 0.0) == regime)
        if selector.sum() < 10:
            slopes[regime] = 0.0
            continue
        x = y[origins[selector]]
        z = y[origins[selector] + horizon]
        slopes[regime] = float(np.dot(x, z) / np.dot(x, x))

    evaluate = usable & (origins > train_end)
    for origin in origins[evaluate]:
        predictions[origin + horizon] = slopes[y[origin] > 0.0] * y[origin]
    return predictions


# ---------------------------------------------------------------------------
# Scenario suite
# ---------------------------------------------------------------------------
def run_all_scenarios(config: dict) -> dict[str, object]:
    """Run every mandated synthetic scenario and return a JSON-ready summary."""
    seed = int(config.get("seed", 20260806))
    n = int(config.get("n_observations", 4000))
    phi = float(config.get("ar1_phi", 0.6))
    ar_q = list(config.get("ar_q_coefficients", [0.5, -0.3, 0.25]))
    nonlinear = list(config.get("nonlinear_threshold_coefficients", [0.7, -0.4]))
    missing_fraction = float(config.get("missing_fraction", 0.2))
    tolerance = float(config.get("white_noise_tolerance", 0.05))
    horizons = (1, 3, 7)
    p = 14

    summary: dict[str, object] = {
        "seed": seed,
        "n_observations": n,
        "horizons": list(horizons),
        "p": p,
        "caveat": (
            "Synthetic scenarios validate implementation behaviour only. A residual "
            "component in the nonlinear scenario is a property of the simulated "
            "generator and is not evidence of nonlinearity in any real PM10 series."
        ),
        "scenarios": {},
    }

    # 1. White noise: no systematic persistence-relative gain.
    rng = np.random.default_rng(seed)
    wn = evaluate_reference_ladder(
        to_daily_series(make_white_noise(n, rng), "synthetic_white_noise"),
        scenario="white_noise",
        horizons=horizons,
        p=p,
    )
    summary["scenarios"]["white_noise"] = {
        "rows": [row.__dict__ for row in wn],
        "max_relative_delta_ar1": max(abs(r.delta_ar1) / r.l_persistence for r in wn),
        "max_relative_delta_mem": max(abs(r.delta_mem) / r.l_persistence for r in wn),
        "tolerance": tolerance,
        "expectation": "|Delta_AR1| and |Delta_mem| are small relative to L_P",
    }

    # 2. AR(1): a sufficiently rich AR(p) adds little over AR(1).
    rng = np.random.default_rng(seed + 1)
    ar1 = evaluate_reference_ladder(
        to_daily_series(make_ar1(n, phi, rng), "synthetic_ar1"),
        scenario="ar1",
        horizons=horizons,
        p=p,
    )
    summary["scenarios"]["ar1"] = {
        "rows": [row.__dict__ for row in ar1],
        "max_relative_delta_mem": max(abs(r.delta_mem) / r.l_persistence for r in ar1),
        "phi": phi,
        "expectation": "AR(1) and AR(p) are close; Delta_mem is near zero",
    }

    # 3. AR(q), q > 1: AR(p) captures memory AR(1) cannot.
    rng = np.random.default_rng(seed + 2)
    arq = evaluate_reference_ladder(
        to_daily_series(make_ar_q(n, ar_q, rng), "synthetic_arq"),
        scenario="ar_q",
        horizons=horizons,
        p=p,
    )
    summary["scenarios"]["ar_q"] = {
        "rows": [row.__dict__ for row in arq],
        "relative_delta_mem_h1": arq[0].delta_mem / arq[0].l_persistence,
        "coefficients": ar_q,
        "expectation": "Delta_mem is materially positive at short horizons",
    }

    # 4. Nonlinear autoregression: residual gain may appear.
    rng = np.random.default_rng(seed + 3)
    nl_values = make_nonlinear_threshold(n, nonlinear, rng)
    nl_series = to_daily_series(nl_values, "synthetic_nonlinear")
    train_end = int(0.5 * n) - 1
    nl_rows: list[SyntheticEvaluation] = []
    for horizon in horizons:
        model_values = _fit_threshold_model_train_only(nl_series, train_end, horizon)
        nl_rows.extend(
            evaluate_reference_ladder(
                nl_series,
                scenario="nonlinear",
                horizons=(horizon,),
                p=p,
                model_values=model_values,
            )
        )
    summary["scenarios"]["nonlinear"] = {
        "rows": [row.__dict__ for row in nl_rows],
        "delta_res_h1": nl_rows[0].delta_res,
        "expectation": "residual gain may appear; this does NOT prove nonlinearity in real data",
    }

    # 5. Incomplete calendar: calendar-aware vs compressed-time estimation.
    rng = np.random.default_rng(seed + 4)
    complete = make_ar1(n, phi, rng)
    gapped = punch_missing_days(complete, missing_fraction, np.random.default_rng(seed + 5))
    gapped_series = to_daily_series(gapped, "synthetic_gapped")
    aware = evaluate_reference_ladder(
        gapped_series, scenario="missing_calendar_aware", horizons=horizons, p=p
    )
    compressed = evaluate_reference_ladder(
        gapped_series,
        scenario="missing_calendar_compressed",
        horizons=horizons,
        p=p,
        compressed_time=True,
    )
    train_end_gap = int(0.5 * n) - 1
    aware_estimate = estimate_autocovariances(
        gapped_series, train_end_position=train_end_gap, max_lag=max(horizons) + p - 1
    )
    compressed_estimate = estimate_autocovariances_compressed_time(
        gapped_series, train_end_position=train_end_gap, max_lag=max(horizons) + p - 1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_aware = aware_estimate.gamma / aware_estimate.gamma[0]
        rho_compressed = compressed_estimate.gamma / compressed_estimate.gamma[0]
    summary["scenarios"]["missing_calendar"] = {
        "missing_fraction": missing_fraction,
        "calendar_aware_rows": [row.__dict__ for row in aware],
        "compressed_time_rows": [row.__dict__ for row in compressed],
        "rho1_calendar_aware": float(rho_aware[1]),
        "rho1_compressed_time": float(rho_compressed[1]),
        "rho1_true_process": phi,
        "abs_error_calendar_aware": float(abs(rho_aware[1] - phi)),
        "abs_error_compressed_time": float(abs(rho_compressed[1] - phi)),
        "expectation": (
            "the calendar-aware lag-1 autocorrelation is closer to the true phi than "
            "the compressed-time estimate, which measures k observed steps rather "
            "than k calendar days"
        ),
    }

    # 6. Identity: exact algebraic recovery in every scenario evaluated above.
    all_rows = wn + ar1 + arq + nl_rows + aware + compressed
    summary["scenarios"]["identity"] = {
        "n_rows_checked": len(all_rows),
        "max_abs_residual": float(max(abs(row.identity_residual) for row in all_rows)),
        "all_passed": bool(all(row.identity_passed for row in all_rows)),
    }

    # 7. Bootstrap pairing: identical sampled origins across methods.
    from .bootstrap import moving_block_origin_indices

    generator = np.random.default_rng(seed + 6)
    n_origins, block_length = 200, 14
    draws = [moving_block_origin_indices(n_origins, block_length, generator) for _ in range(5)]
    replay = np.random.default_rng(seed + 6)
    replayed = [moving_block_origin_indices(n_origins, block_length, replay) for _ in range(5)]
    summary["scenarios"]["bootstrap_pairing"] = {
        "n_origins": n_origins,
        "block_length": block_length,
        "deterministic_under_seed": bool(
            all(np.array_equal(a, b) for a, b in zip(draws, replayed))
        ),
        "sample_size_preserved": bool(all(draw.size == n_origins for draw in draws)),
        "expectation": (
            "one index draw per replicate is shared by every method and horizon "
            "column, so pairing survives resampling"
        ),
    }

    return summary


def scenarios_to_frame(summary: dict[str, object]) -> pd.DataFrame:
    """Flatten the scenario rows into a tabular view."""
    records: list[dict[str, object]] = []
    for name, payload in summary["scenarios"].items():  # type: ignore[index]
        if not isinstance(payload, dict):
            continue
        for key in ("rows", "calendar_aware_rows", "compressed_time_rows"):
            for row in payload.get(key, []):  # type: ignore[union-attr]
                records.append({"scenario_group": name, **row})
    return pd.DataFrame(records)
