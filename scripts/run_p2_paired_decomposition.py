#!/usr/bin/env python3
"""Execute the P2 paired finite-linear-memory skill decomposition.

Canon:    docs/canon/P2_PROJECT_CANON.md (v2.0)
Contract: docs/canon/P2_PAIRED_DECOMPOSITION_CONTRACT.md (v1.0)
Decision: 2026-08-06-p2-finite-linear-memory-skill-decomposition

The script is read-only with respect to every input. It writes only under
``outputs/p2_paired_decomposition/`` and ``inputs/P2_INPUT_PROVENANCE.json``.
It trains no models, opens no datasets beyond those declared in the config,
and never selects a model using the evaluated test loss.

Usage::

    python scripts/run_p2_paired_decomposition.py \
        --config config/p2_paired_decomposition.yaml
"""

from __future__ import annotations

import os

# Bound BLAS threading before numpy is imported: the target environment is an
# 8 GB machine with limited parallelism.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "4")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.p2_decomposition import DECISION_ID, CANON_VERSION  # noqa: E402
from src.p2_decomposition.autocovariance import (  # noqa: E402
    estimate_autocovariances,
    estimate_autocovariances_compressed_time,
    training_end_position,
)
from src.p2_decomposition.bootstrap import (  # noqa: E402
    percentile_interval,
    run_moving_block_bootstrap,
)
from src.p2_decomposition.calendar import load_daily_series  # noqa: E402
from src.p2_decomposition.decomposition import (  # noqa: E402
    check_identity,
    compute_components,
    compute_components_arrays,
    linear_fraction,
    linear_fraction_arrays,
)
from src.p2_decomposition.diagnostics import (  # noqa: E402
    DiagnosticsCollector,
    diagnostics_record,
)
from src.p2_decomposition.gate import GateCondition, evaluate_gate  # noqa: E402
from src.p2_decomposition.linear_references import (  # noqa: E402
    NumericsPolicy,
    direct_projection_coefficients,
    direct_projection_forecast,
    evaluate_gamma_matrix,
)
from src.p2_decomposition.pairing import (  # noqa: E402
    PAIRED_KEY,
    build_common_support,
    load_model_predictions,
    select_evaluated_models,
)
from src.p2_decomposition.provenance import (  # noqa: E402
    ArtefactProvenance,
    RunStamp,
    git_commit,
    sha256_file,
    stamp_frame,
    utc_now,
    write_json,
)
from src.p2_decomposition.synthetic import run_all_scenarios  # noqa: E402

LOGGER = logging.getLogger("p2")

PRODUCER_REPOSITORY = "fedeg-umh-es/varret-pm10-paper"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_nanmean(samples: np.ndarray) -> float:
    """Mean over non-``NaN`` replicates; ``NaN`` when every replicate is ``NaN``."""
    finite = np.isfinite(samples)
    if not finite.any():
        return float("nan")
    return float(samples[finite].mean())


def _display_path(path: Path) -> str:
    """Repository-relative path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ar_method_name(p: int) -> str:
    """Canonical method label for a lag order (``1 -> ar1``, ``7 -> ar7``…)."""
    return f"ar{p}"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_reference_predictions(
    series,
    *,
    station: str,
    origins: pd.DatetimeIndex,
    horizons: list[int],
    orders: list[int],
    policy: NumericsPolicy,
    estimation: dict,
    collector: DiagnosticsCollector | None,
    compressed_time: bool = False,
) -> pd.DataFrame:
    """Compute direct AR(p) reference forecasts for every origin and horizon.

    For each origin the training window ends strictly before the origin date;
    ``mu_hat`` and ``gamma_hat`` come from that window only. One Toeplitz system
    is evaluated per lag order and one projection is solved per horizon, so
    every horizon has its own direct coefficients.
    """
    max_lag = max(horizons) + max(orders) - 1
    offset = int(estimation["train_end_offset_days"])
    min_pairs = int(estimation["min_pairs_per_lag"])
    min_train = int(estimation["min_train_observations"])
    window_mode = str(estimation.get("window", "expanding"))
    rolling_days = estimation.get("rolling_window_days")

    records: list[dict[str, object]] = []
    for origin in origins:
        origin_position = series.position_of(origin)
        if origin_position < 0:
            continue
        train_end = training_end_position(origin_position, offset)
        train_start = 0
        if window_mode == "rolling" and rolling_days:
            train_start = max(0, train_end - int(rolling_days) + 1)

        if compressed_time:
            estimate = estimate_autocovariances_compressed_time(
                series,
                train_end_position=train_end,
                max_lag=max_lag,
                train_start_position=train_start,
            )
        else:
            estimate = estimate_autocovariances(
                series,
                train_end_position=train_end,
                max_lag=max_lag,
                train_start_position=train_start,
                min_pairs_per_lag=min_pairs,
                min_train_observations=min_train,
            )

        fold_id = pd.Timestamp(origin).strftime("%Y-%m-%d")
        if collector is not None and not compressed_time:
            collector.add_pair_counts(
                station=station,
                origin_date=origin,
                n_pairs=estimate.n_pairs,
                gamma=estimate.gamma,
            )

        for p in orders:
            diagnostics = evaluate_gamma_matrix(
                estimate.gamma,
                p,
                policy=policy,
                max_lag_required=max(horizons) + p - 1,
                n_pairs=estimate.n_pairs,
            )
            lag_vector = series.lag_vector(origin_position, p)
            solved: list[int] = []
            refused: list[int] = []

            for horizon in horizons:
                target_position = origin_position + horizon
                if target_position >= len(series.values):
                    continue
                solution = direct_projection_coefficients(
                    estimate.gamma,
                    p,
                    horizon,
                    policy=policy,
                    diagnostics=diagnostics,
                    pair_count_min=diagnostics.pair_count_min,
                )
                if collector is not None and not compressed_time:
                    collector.add_solver_status(
                        station=station,
                        fold_or_window_id=fold_id,
                        origin_date=origin,
                        p=p,
                        horizon=horizon,
                        solver_status=solution.solver_status,
                    )

                available = solution.is_valid and lag_vector is not None
                if available:
                    solved.append(horizon)
                    y_pred = direct_projection_forecast(
                        estimate.mu, solution.beta, lag_vector
                    )
                else:
                    refused.append(horizon)
                    y_pred = np.nan

                records.append(
                    {
                        "station": station,
                        "model": ar_method_name(p),
                        "fold_or_window_id": fold_id,
                        "origin_date": pd.Timestamp(origin),
                        "target_date": series.index[target_position],
                        "horizon": horizon,
                        "y_true": series.values[target_position],
                        "y_pred": y_pred,
                        "solver_status": solution.solver_status,
                        "lag_vector_available": lag_vector is not None,
                        "autocovariance_status": estimate.status,
                        "mu_hat": estimate.mu,
                        "n_train_observed": estimate.n_train_observed,
                    }
                )

            if collector is not None and not compressed_time:
                collector.add_matrix(
                    diagnostics_record(
                        diagnostics,
                        station=station,
                        fold_or_window_id=fold_id,
                        origin_date=origin,
                        horizons_solved=solved,
                        horizons_refused=refused,
                    )
                )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame["forecast_available_at_origin"] = frame["y_pred"].notna()
    frame["squared_error"] = np.where(
        frame["y_true"].notna() & frame["y_pred"].notna(),
        (frame["y_true"] - frame["y_pred"]) ** 2,
        np.nan,
    )
    frame["source_artifact"] = series.source_path
    frame["dataset_id"] = station
    return frame


def losses_from_support(long_frame: pd.DataFrame, support) -> pd.DataFrame:
    """Restrict the long loss frame to a support and return the paired rows."""
    key = list(PAIRED_KEY)
    restricted = long_frame.merge(support.keys[key], on=key, how="inner")
    return restricted[restricted["model"].isin(support.methods)].copy()


def mean_losses(paired: pd.DataFrame) -> pd.DataFrame:
    """Mean squared error by ``(horizon, model)`` on a fixed support."""
    return (
        paired.groupby(["horizon", "model"], observed=True)["squared_error"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "mse", "size": "n_cases"})
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/p2_paired_decomposition.yaml", type=Path
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )
    logging.Formatter.converter = __import__("time").gmtime

    config_path = (REPO_ROOT / args.config).resolve()
    config = load_config(config_path)
    config_sha = sha256_file(config_path)

    scope = config["scope"]
    horizons = list(scope["horizons"])
    p_orders = list(scope["p_orders"])
    ar_orders = [int(scope["ar1_order"])] + p_orders
    declared_models = list(scope["models"])
    baseline = str(scope["baseline_method"])

    policy = NumericsPolicy(
        max_condition_number=float(config["numerics"]["max_condition_number"]),
        min_eigenvalue_strictly_positive=bool(
            config["numerics"]["min_eigenvalue_strictly_positive"]
        ),
        regularisation_policy=str(config["numerics"]["regularisation_policy"]),
    )
    identity_atol = float(config["identity_check"]["atol"])
    identity_rtol = float(config["identity_check"]["rtol"])
    y_true_tolerance = float(config["pairing"]["y_true_tolerance"])
    fractions_cfg = config["fractions"]
    bootstrap_cfg = config["bootstrap"]

    out_dir = REPO_ROOT / config["outputs"]["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = REPO_ROOT / config["outputs"]["provenance_manifest"]

    code_commit = git_commit(REPO_ROOT)

    # -- 1. Input provenance ------------------------------------------------
    LOGGER.info("auditing input provenance")
    artefacts: list[ArtefactProvenance] = []
    for station_cfg in config["stations"]:
        for role, rel_path, model_scope, schema in (
            (
                "daily_pm10_series",
                station_cfg["series_path"],
                "n/a",
                "date,pm10 (daily, incomplete calendar preserved)",
            ),
            (
                "row_level_predictions",
                station_cfg["predictions_path"],
                "persistence,ridge_direct,hgb_direct,sarima,seasonal_naive,stl_ridge_direct",
                "dataset,model,fold,origin_date,horizon,date,y_true,y_pred",
            ),
        ):
            absolute = REPO_ROOT / rel_path
            if not absolute.exists():
                artefacts.append(
                    ArtefactProvenance(
                        logical_role=role,
                        station_scope=station_cfg["name"],
                        model_scope=model_scope,
                        source_path=rel_path,
                        copied_path=None,
                        producer_repository=PRODUCER_REPOSITORY,
                        producer_commit=code_commit,
                        sha256="",
                        size_bytes=0,
                        schema_summary=schema,
                        provenance_status="MISSING",
                        allowed_use="none",
                    )
                )
                continue
            artefacts.append(
                ArtefactProvenance(
                    logical_role=role,
                    station_scope=station_cfg["name"],
                    model_scope=model_scope,
                    source_path=rel_path,
                    copied_path=None,
                    producer_repository=PRODUCER_REPOSITORY,
                    producer_commit=code_commit,
                    sha256=sha256_file(absolute),
                    size_bytes=absolute.stat().st_size,
                    schema_summary=schema,
                    provenance_status="VERIFIED_LOCAL_P2",
                    allowed_use="read-only input to the P2 paired decomposition",
                )
            )

    manifest_payload = {
        "decision_id": DECISION_ID,
        "canon_version": CANON_VERSION,
        "generated_at_utc": utc_now(),
        "execution_repository": PRODUCER_REPOSITORY,
        "execution_commit": code_commit,
        "notes": [
            "The row-level prediction artefacts are native to the execution "
            "repository, so they are VERIFIED_LOCAL_P2 rather than "
            "VERIFIED_EXTERNAL_IMMUTABLE_INPUT; see "
            "docs/canon/PM10_RESEARCH_DECISION_LOG.md.",
            "No artefact was copied from P4. P4 was not reachable, read, executed "
            "or modified.",
            "No model was trained and no new dataset was opened.",
        ],
        "artefacts": [artefact.as_dict() for artefact in artefacts],
    }
    manifest_sha = write_json(manifest_path, manifest_payload)
    LOGGER.info("wrote %s (sha256=%s)", manifest_path, manifest_sha[:12])

    if any(a.provenance_status == "MISSING" for a in artefacts):
        LOGGER.error("required inputs missing; refusing to fabricate an empirical run")
        return 2

    stamp = RunStamp(
        code_commit=code_commit,
        config_sha256=config_sha,
        input_manifest_sha256=manifest_sha,
        generated_at_utc=utc_now(),
        decision_id=DECISION_ID,
        canon_version=CANON_VERSION,
    )

    # -- 2. Per-station computation ----------------------------------------
    collector = DiagnosticsCollector()
    paired_rows: list[pd.DataFrame] = []
    decomposition_rows: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    fraction_rows: list[dict[str, object]] = []
    p_sensitivity_rows: list[dict[str, object]] = []
    missingness_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    station_periods: dict[str, str] = {}

    support_definitions = config["pairing"]["support_types"]
    global_methods = [baseline] + [ar_method_name(p) for p in ar_orders] + [
        "ridge_direct",
        "hgb_direct",
    ]

    for station_cfg in config["stations"]:
        station = station_cfg["name"]
        LOGGER.info("station %s: loading series and predictions", station)
        series = load_daily_series(
            REPO_ROOT / station_cfg["series_path"],
            station=station,
            date_column=station_cfg["date_column"],
            value_column=station_cfg["value_column"],
            freq=config["calendar"]["freq"],
        )
        station_periods[station] = (
            f"{series.index[0].date()}..{series.index[-1].date()}"
        )
        LOGGER.info(
            "station %s: calendar %s, observed %d, missing %d",
            station,
            station_periods[station],
            series.n_observed,
            series.n_missing,
        )

        predictions_path = REPO_ROOT / station_cfg["predictions_path"]
        model_frame = load_model_predictions(
            predictions_path,
            station=station,
            dataset_id=station_cfg["dataset_id"],
            horizons=horizons,
            source_sha256=sha256_file(predictions_path),
            producer_repository=PRODUCER_REPOSITORY,
            producer_commit=code_commit,
        )

        available_models = set(model_frame["model"].unique())
        evaluated_models = select_evaluated_models(
            declared_models,
            available_models=available_models,
            allow_test_loss_selection=bool(
                config["selection"]["allow_test_loss_selection"]
            ),
        )
        LOGGER.info("station %s: evaluated models %s", station, evaluated_models)

        origins = pd.DatetimeIndex(sorted(model_frame["origin_date"].unique()))
        LOGGER.info("station %s: %d origins, building AR references", station, len(origins))
        reference_frame = build_reference_predictions(
            series,
            station=station,
            origins=origins,
            horizons=horizons,
            orders=ar_orders,
            policy=policy,
            estimation=config["estimation"],
            collector=collector,
        )

        shared_columns = [
            "station",
            "model",
            "fold_or_window_id",
            "origin_date",
            "target_date",
            "horizon",
            "y_true",
            "y_pred",
            "forecast_available_at_origin",
            "squared_error",
        ]
        long_frame = pd.concat(
            [model_frame[shared_columns], reference_frame[shared_columns]],
            ignore_index=True,
        )

        # -- supports -------------------------------------------------------
        supports = {}
        supports["GLOBAL_COMMON"] = build_common_support(
            long_frame,
            station=station,
            support_type="GLOBAL_COMMON",
            methods=global_methods,
            y_true_tolerance=y_true_tolerance,
        )
        if "sarima" in available_models:
            supports["GLOBAL_COMMON_WITH_SARIMA"] = build_common_support(
                long_frame,
                station=station,
                support_type="GLOBAL_COMMON_WITH_SARIMA",
                methods=global_methods + ["sarima"],
                y_true_tolerance=y_true_tolerance,
            )
        # Order-specific supports deliberately exclude sarima: they exist to
        # isolate the effect of the lag order on the sample, so the model set
        # must match the principal support's, otherwise two things move at once.
        order_specific_models = [
            m for m in evaluated_models if m in {"ridge_direct", "hgb_direct"}
        ]
        for p in p_orders:
            supports[f"ORDER_SPECIFIC_p{p}"] = build_common_support(
                long_frame,
                station=station,
                support_type=f"ORDER_SPECIFIC_p{p}",
                methods=[baseline, ar_method_name(1), ar_method_name(p)]
                + order_specific_models,
                y_true_tolerance=y_true_tolerance,
            )

        for name, support in supports.items():
            by_horizon = support.counts_by_horizon()
            for horizon in horizons:
                support_rows.append(
                    {
                        "station": station,
                        "support_type": name,
                        "horizon": horizon,
                        "methods": ";".join(support.methods),
                        "n_cases": int(by_horizon.get(horizon, 0)),
                        "period": station_periods[station],
                        "resolution": scope["resolution"],
                    }
                )
            LOGGER.info(
                "station %s: support %s -> %d cases", station, name, support.n_cases
            )

        # -- principal decomposition ---------------------------------------
        for support_name, support in supports.items():
            if support.n_cases == 0:
                continue
            paired = losses_from_support(long_frame, support)
            losses = mean_losses(paired)
            wide = losses.pivot(index="horizon", columns="model", values="mse")
            counts = losses.pivot(index="horizon", columns="model", values="n_cases")

            if support_name == "GLOBAL_COMMON":
                stamped = paired.copy()
                stamped["support_type"] = support_name
                paired_rows.append(stamped)

            support_models = [
                m for m in support.methods if m in set(declared_models)
            ]
            orders_here = (
                p_orders
                if not support_name.startswith("ORDER_SPECIFIC")
                else [int(support_name.split("_p")[-1])]
            )

            for horizon in horizons:
                if horizon not in wide.index:
                    continue
                l_p = float(wide.loc[horizon, baseline])
                l_ar1 = float(wide.loc[horizon, ar_method_name(1)])
                n_cases = int(counts.loc[horizon, baseline])
                for p in orders_here:
                    ar_p = ar_method_name(p)
                    if ar_p not in wide.columns:
                        continue
                    l_arp = float(wide.loc[horizon, ar_p])
                    for model in support_models:
                        if model not in wide.columns:
                            continue
                        l_m = float(wide.loc[horizon, model])
                        components = compute_components(l_p, l_ar1, l_arp, l_m)
                        identity = check_identity(
                            components, atol=identity_atol, rtol=identity_rtol
                        )
                        row = {
                            "station": station,
                            "period": station_periods[station],
                            "resolution": scope["resolution"],
                            "support_type": support_name,
                            "horizon": horizon,
                            "p": p,
                            "model": model,
                            "n_cases": n_cases,
                            **components.as_dict(),
                        }
                        decomposition_rows.append(row)
                        identity_rows.append(
                            {
                                "station": station,
                                "support_type": support_name,
                                "horizon": horizon,
                                "p": p,
                                "model": model,
                                "n_cases": n_cases,
                                "delta_total": components.delta_total,
                                "sum_of_components": components.delta_ar1
                                + components.delta_mem
                                + components.delta_res,
                                "residual": identity.residual,
                                "tolerance": identity.tolerance,
                                "atol": identity.atol,
                                "rtol": identity.rtol,
                                "passed": identity.passed,
                            }
                        )
                        fraction = linear_fraction(
                            l_p,
                            l_arp,
                            l_m,
                            abs_threshold=float(
                                fractions_cfg["denominator_abs_threshold"]
                            ),
                            rel_threshold=float(
                                fractions_cfg["denominator_rel_threshold"]
                            ),
                        )
                        fraction_rows.append(
                            {
                                "station": station,
                                "support_type": support_name,
                                "horizon": horizon,
                                "p": p,
                                "model": model,
                                "n_cases": n_cases,
                                "pi_linear": fraction.value,
                                "status": fraction.status,
                                "denominator": fraction.denominator,
                                "denominator_threshold": fraction.denominator_threshold,
                                "rel_threshold": float(
                                    fractions_cfg["denominator_rel_threshold"]
                                ),
                            }
                        )
                        if support_name.startswith("ORDER_SPECIFIC") or (
                            support_name == "GLOBAL_COMMON"
                        ):
                            p_sensitivity_rows.append(
                                {
                                    "station": station,
                                    "support_type": support_name,
                                    "horizon": horizon,
                                    "p": p,
                                    "model": model,
                                    "n_cases": n_cases,
                                    "delta_ar1": components.delta_ar1,
                                    "delta_mem": components.delta_mem,
                                    "delta_res": components.delta_res,
                                    "delta_total": components.delta_total,
                                }
                            )

        # -- fraction threshold sensitivity ---------------------------------
        support = supports["GLOBAL_COMMON"]
        if support.n_cases:
            paired = losses_from_support(long_frame, support)
            wide = mean_losses(paired).pivot(
                index="horizon", columns="model", values="mse"
            )
            for rel in fractions_cfg["denominator_rel_threshold_sensitivity"]:
                for horizon in wide.index:
                    l_p = float(wide.loc[horizon, baseline])
                    for p in p_orders:
                        l_arp = float(wide.loc[horizon, ar_method_name(p)])
                        for model in evaluated_models:
                            if model not in wide.columns:
                                continue
                            alt = linear_fraction(
                                l_p,
                                l_arp,
                                float(wide.loc[horizon, model]),
                                abs_threshold=float(
                                    fractions_cfg["denominator_abs_threshold"]
                                ),
                                rel_threshold=float(rel),
                            )
                            fraction_rows.append(
                                {
                                    "station": station,
                                    "support_type": "GLOBAL_COMMON_THRESHOLD_SENSITIVITY",
                                    "horizon": int(horizon),
                                    "p": p,
                                    "model": model,
                                    "n_cases": int(support.counts_by_horizon().get(horizon, 0)),
                                    "pi_linear": alt.value,
                                    "status": alt.status,
                                    "denominator": alt.denominator,
                                    "denominator_threshold": alt.denominator_threshold,
                                    "rel_threshold": float(rel),
                                }
                            )

        # -- missingness sensitivity ---------------------------------------
        if config["calendar"]["run_compressed_time_sensitivity"]:
            LOGGER.info("station %s: compressed-time missingness sensitivity", station)
            compressed_frame = build_reference_predictions(
                series,
                station=station,
                origins=origins,
                horizons=horizons,
                orders=ar_orders,
                policy=policy,
                estimation=config["estimation"],
                collector=None,
                compressed_time=True,
            )
            support = supports["GLOBAL_COMMON"]
            if support.n_cases and not compressed_frame.empty:
                key = list(PAIRED_KEY)
                compressed_paired = compressed_frame.merge(
                    support.keys[key], on=key, how="inner"
                )
                aware_paired = losses_from_support(long_frame, support)
                aware_mse = mean_losses(aware_paired).pivot(
                    index="horizon", columns="model", values="mse"
                )
                compressed_mse = (
                    compressed_paired.groupby(["horizon", "model"], observed=True)[
                        "squared_error"
                    ]
                    .mean()
                    .unstack()
                )
                for horizon in horizons:
                    if horizon not in aware_mse.index:
                        continue
                    for p in ar_orders:
                        method = ar_method_name(p)
                        aware_value = float(aware_mse.loc[horizon, method])
                        compressed_value = (
                            float(compressed_mse.loc[horizon, method])
                            if horizon in compressed_mse.index
                            and method in compressed_mse.columns
                            else float("nan")
                        )
                        missingness_rows.append(
                            {
                                "station": station,
                                "period": station_periods[station],
                                "support_type": "GLOBAL_COMMON",
                                "horizon": horizon,
                                "p": p,
                                "method": method,
                                "n_cases": int(
                                    support.counts_by_horizon().get(horizon, 0)
                                ),
                                "mse_calendar_aware": aware_value,
                                "mse_compressed_time": compressed_value,
                                "difference": compressed_value - aware_value,
                                "relative_difference": (
                                    (compressed_value - aware_value) / aware_value
                                    if aware_value
                                    else float("nan")
                                ),
                                "primary_estimator": "CALENDAR_ALIGNED_TRAIN_ONLY",
                                "note": (
                                    "compressed-time estimation drops gaps before "
                                    "lagging and is invalid as a primary estimator"
                                ),
                            }
                        )

        # -- bootstrap ------------------------------------------------------
        if bootstrap_cfg["enabled"]:
            for support_name in ("GLOBAL_COMMON", "GLOBAL_COMMON_WITH_SARIMA"):
                support = supports.get(support_name)
                if support is None or support.n_cases == 0:
                    continue
                paired = losses_from_support(long_frame, support)
                origin_index = pd.DatetimeIndex(
                    sorted(paired["origin_date"].unique())
                )
                methods_here = list(support.methods)
                columns = [(m, h) for m in methods_here for h in horizons]
                column_names = tuple(f"{m}|h{h}" for m, h in columns)
                lookup = {name: i for i, name in enumerate(column_names)}
                position = {date: i for i, date in enumerate(origin_index)}

                matrix = np.full((len(origin_index), len(columns)), np.nan)
                rows = paired[["origin_date", "model", "horizon", "squared_error"]]
                for origin_date, model, horizon, error in rows.itertuples(index=False):
                    matrix[
                        position[origin_date], lookup[f"{model}|h{horizon}"]
                    ] = error

                for block_length in bootstrap_cfg["block_lengths_origins"]:
                    if block_length > len(origin_index):
                        continue
                    result = run_moving_block_bootstrap(
                        matrix,
                        station=station,
                        support_type=support_name,
                        series_names=column_names,
                        block_length=int(block_length),
                        n_replicates=int(bootstrap_cfg["n_replicates"]),
                        seed=int(bootstrap_cfg["seed"]),
                        confidence_level=float(bootstrap_cfg["confidence_level"]),
                        interval_method=str(bootstrap_cfg["interval_method"]),
                        chunk_size=int(bootstrap_cfg["replicate_chunk_size"]),
                    )
                    means = result.replicate_means

                    def column(method: str, horizon: int) -> np.ndarray:
                        return means[:, lookup[f"{method}|h{horizon}"]]

                    for horizon in horizons:
                        l_p = column(baseline, horizon)
                        l_ar1 = column(ar_method_name(1), horizon)
                        for p in p_orders:
                            l_arp = column(ar_method_name(p), horizon)
                            for model in [
                                m for m in methods_here if m in set(declared_models)
                            ]:
                                l_m = column(model, horizon)
                                comps = compute_components_arrays(
                                    l_p, l_ar1, l_arp, l_m
                                )
                                comps["pi_linear"] = linear_fraction_arrays(
                                    l_p,
                                    l_arp,
                                    l_m,
                                    abs_threshold=float(
                                        fractions_cfg["denominator_abs_threshold"]
                                    ),
                                    rel_threshold=float(
                                        fractions_cfg["denominator_rel_threshold"]
                                    ),
                                )
                                identity_residual = np.nanmax(
                                    np.abs(
                                        comps["delta_total"]
                                        - (
                                            comps["delta_ar1"]
                                            + comps["delta_mem"]
                                            + comps["delta_res"]
                                        )
                                    )
                                )
                                for quantity, samples in comps.items():
                                    lower, upper = percentile_interval(
                                        samples[:, None],
                                        float(bootstrap_cfg["confidence_level"]),
                                    )
                                    bootstrap_rows.append(
                                        {
                                            "station": station,
                                            "support_type": support_name,
                                            "horizon": horizon,
                                            "p": p,
                                            "model": model,
                                            "quantity": quantity,
                                            "block_length": int(block_length),
                                            "n_replicates": result.n_replicates,
                                            "seed": result.seed,
                                            "confidence_level": result.confidence_level,
                                            "interval_method": result.interval_method,
                                            "n_origins": result.n_origins,
                                            # An all-NaN column is a suppressed
                                            # fraction, not an error; it must
                                            # stay visibly undefined.
                                            "replicate_mean": _safe_nanmean(samples),
                                            "ci_lower": float(lower[0]),
                                            "ci_upper": float(upper[0]),
                                            "excludes_zero": bool(
                                                (lower[0] > 0.0) or (upper[0] < 0.0)
                                            ),
                                            "max_identity_residual_in_replicates": float(
                                                identity_residual
                                            ),
                                        }
                                    )

    # -- 3. Synthetic validation -------------------------------------------
    LOGGER.info("running synthetic validation")
    synthetic_summary = run_all_scenarios(config["synthetic"])
    synthetic_summary["run_stamp"] = stamp.as_dict()

    # -- 4. Write artefacts -------------------------------------------------
    LOGGER.info("writing artefacts to %s", out_dir)
    written: dict[str, dict[str, object]] = {}

    def write_table(frame: pd.DataFrame, filename: str, description: str) -> None:
        path = out_dir / filename
        stamped = stamp_frame(frame, stamp)
        if filename.endswith(".parquet"):
            stamped.to_parquet(path, index=False)
        else:
            stamped.to_csv(path, index=False)
        written[filename] = {
            "path": _display_path(path),
            "sha256": sha256_file(path),
            "rows": int(len(stamped)),
            "columns": int(stamped.shape[1]),
            "description": description,
        }

    paired_all = (
        pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    )
    write_table(
        paired_all,
        "losses_paired.parquet",
        "Row-level paired squared errors on the principal GLOBAL_COMMON support.",
    )
    write_table(
        pd.DataFrame(decomposition_rows),
        "decomposition_by_station_horizon_model_p.csv",
        "Primary additive decomposition by station, horizon, model and lag order.",
    )
    write_table(
        pd.DataFrame(support_rows),
        "support_counts.csv",
        "Paired case counts by station, support type and horizon.",
    )
    write_table(
        collector.matrix_frame(),
        "yule_walker_diagnostics.csv",
        "Gamma_p eigen-diagnostics, rank, conditioning and solver status by station, "
        "fold and lag order (horizon-invariant by construction).",
    )
    write_table(
        collector.solver_frame(),
        "yule_walker_solver_status.parquet",
        "Per-horizon solver status for every fitted direct projection.",
    )
    write_table(
        collector.pair_count_frame(),
        "autocovariance_pair_counts.parquet",
        "Calendar-aligned valid-pair count and gamma estimate by station, fold and lag.",
    )
    write_table(
        collector.summary(),
        "yule_walker_diagnostics_summary.csv",
        "Counts of valid and refused Gamma_p systems by station and lag order.",
    )
    write_table(
        pd.DataFrame(identity_rows),
        "mse_identity_checks.csv",
        "Numerical verification of Delta_total = Delta_AR1 + Delta_mem + Delta_res.",
    )
    write_table(
        pd.DataFrame(bootstrap_rows),
        "bootstrap_intervals.csv",
        "Moving-block bootstrap percentile intervals for every component and fraction.",
    )
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    if not bootstrap_frame.empty:
        sensitivity = bootstrap_frame.pivot_table(
            index=["station", "support_type", "horizon", "p", "model", "quantity"],
            columns="block_length",
            values=["ci_lower", "ci_upper", "excludes_zero"],
            aggfunc="first",
        )
        sensitivity.columns = [f"{a}_block{b}" for a, b in sensitivity.columns]
        sensitivity = sensitivity.reset_index()
        exclusion_columns = [
            c for c in sensitivity.columns if c.startswith("excludes_zero_block")
        ]
        sensitivity["sign_conclusion_stable_across_blocks"] = (
            sensitivity[exclusion_columns].nunique(axis=1) == 1
        )
    else:  # pragma: no cover
        sensitivity = pd.DataFrame()
    write_table(
        sensitivity,
        "bootstrap_block_sensitivity.csv",
        "Block-length sensitivity of every bootstrap interval; no length is primary.",
    )
    write_table(
        pd.DataFrame(p_sensitivity_rows),
        "p_sensitivity.csv",
        "Components under the fixed global support and under order-specific support.",
    )
    write_table(
        pd.DataFrame(fraction_rows),
        "normalised_fractions_secondary.csv",
        "Secondary pi_linear fractions with suppression status and threshold sensitivity.",
    )
    write_table(
        pd.DataFrame(missingness_rows),
        "missingness_sensitivity.csv",
        "Calendar-aware versus compressed-time autocovariance estimation, same support.",
    )

    synthetic_path = out_dir / "synthetic_validation_summary.json"
    synthetic_sha = write_json(synthetic_path, synthetic_summary)
    written["synthetic_validation_summary.json"] = {
        "path": str(synthetic_path.relative_to(REPO_ROOT)),
        "sha256": synthetic_sha,
        "rows": len(synthetic_summary["scenarios"]),
        "columns": 0,
        "description": "Deterministic seeded synthetic validation of the machinery.",
    }

    # -- 5. Mechanical gate -------------------------------------------------
    LOGGER.info("evaluating the mechanical gate")
    identity_frame = pd.DataFrame(identity_rows)
    decomposition_frame = pd.DataFrame(decomposition_rows)
    support_frame = pd.DataFrame(support_rows)

    global_support = support_frame[support_frame["support_type"] == "GLOBAL_COMMON"]
    stations_with_results = sorted(
        decomposition_frame[decomposition_frame["support_type"] == "GLOBAL_COMMON"][
            "station"
        ].unique()
    )
    identity_ok = bool(identity_frame["passed"].all()) if len(identity_frame) else False
    diagnostics_frame = collector.matrix_frame()
    invalid_matrices = (
        int((diagnostics_frame["solver_status"] != "VALID").sum())
        if len(diagnostics_frame)
        else 0
    )

    missingness_frame = pd.DataFrame(missingness_rows)
    synthetic_ok = bool(synthetic_summary["scenarios"]["identity"]["all_passed"])

    conditions = [
        GateCondition(
            "PAIRED_SUPPORT_VALID",
            "PASS" if bool((global_support["n_cases"] > 0).all()) else "FAIL",
            ["outputs/p2_paired_decomposition/support_counts.csv"],
            "One-to-one keys, identical y_true and target dates, and identical "
            "availability were asserted for every support before any loss was "
            "aggregated; the principal support is fixed across p.",
        ),
        GateCondition(
            "TRAIN_ONLY_VALID",
            "PASS",
            [
                "outputs/p2_paired_decomposition/yule_walker_diagnostics.csv",
                "outputs/p2_paired_decomposition/autocovariance_pair_counts.parquet",
            ],
            "mu_hat and gamma_hat for every origin are estimated on observations "
            f"ending {config['estimation']['train_end_offset_days']} day(s) before "
            "the origin date; the lag vector at the origin is information available "
            "at the origin, exactly like persistence.",
        ),
        GateCondition(
            "NO_ORACLE_SELECTION",
            "PASS",
            ["outputs/p2_paired_decomposition/decomposition_by_station_horizon_model_p.csv"],
            "Models come from a declared fixed list; select_evaluated_models raises "
            "if evaluated losses are passed in. No best-model envelope row exists.",
        ),
        GateCondition(
            "MSE_IDENTITY_VERIFIED",
            "PASS" if identity_ok else "FAIL",
            ["outputs/p2_paired_decomposition/mse_identity_checks.csv"],
            f"{int(identity_frame['passed'].sum()) if len(identity_frame) else 0}/"
            f"{len(identity_frame)} cells within atol={identity_atol}, rtol={identity_rtol}.",
        ),
        GateCondition(
            "MISSINGNESS_TESTS_PASS",
            "PASS" if len(missingness_frame) else "FAIL",
            ["outputs/p2_paired_decomposition/missingness_sensitivity.csv"],
            "The complete daily calendar is preserved, gaps are never dropped before "
            "lag construction, and the compressed-time estimator is reported only as "
            "a labelled sensitivity.",
        ),
        GateCondition(
            "P_SENSITIVITY_COMPLETED",
            "PASS" if len(pd.DataFrame(p_sensitivity_rows)) else "FAIL",
            ["outputs/p2_paired_decomposition/p_sensitivity.csv"],
            f"p in {p_orders} evaluated on the fixed global support and on "
            "order-specific supports, with case counts on both.",
        ),
        GateCondition(
            "BLOCK_BOOTSTRAP_COMPLETED",
            "PASS" if len(bootstrap_frame) else "FAIL",
            [
                "outputs/p2_paired_decomposition/bootstrap_intervals.csv",
                "outputs/p2_paired_decomposition/bootstrap_block_sensitivity.csv",
            ],
            f"Block lengths {bootstrap_cfg['block_lengths_origins']} over origin "
            f"vectors, {bootstrap_cfg['n_replicates']} replicates, seed "
            f"{bootstrap_cfg['seed']}. No length is marked primary.",
        ),
        GateCondition(
            "SYNTHETIC_VALIDATION_COMPLETED",
            "PASS" if synthetic_ok else "FAIL",
            ["outputs/p2_paired_decomposition/synthetic_validation_summary.json"],
            "White noise, AR(1), AR(q), nonlinear, incomplete-calendar, identity and "
            "bootstrap-pairing scenarios all executed under a fixed seed.",
        ),
        GateCondition(
            "RESULT_REPLICATES_ACROSS_MORE_THAN_ONE_STATION",
            "PASS" if len(stations_with_results) > 1 else "FAIL",
            ["outputs/p2_paired_decomposition/decomposition_by_station_horizon_model_p.csv"],
            f"A complete, identity-verified decomposition with bootstrap intervals was "
            f"produced on the principal support for {len(stations_with_results)} "
            f"stations: {stations_with_results}. This condition records mechanical "
            "availability across stations only; whether the substantive pattern "
            "replicates is part of NON_TRIVIAL_INTERPRETATION_FOUND.",
        ),
        GateCondition(
            "NON_TRIVIAL_INTERPRETATION_FOUND",
            "PENDING_SCIENTIFIC_REVIEW",
            ["reports/P2_PAIRED_DECOMPOSITION_REPORT.md"],
            "Scientific judgement reserved to the human reviewer. This pipeline "
            "cannot declare P2_PAPER_GO.",
        ),
    ]
    gate_payload = evaluate_gate(conditions)
    gate_payload["run_stamp"] = stamp.as_dict()
    gate_payload["invalid_gamma_matrices"] = invalid_matrices
    gate_path = out_dir / "paper_go_gate.json"
    gate_sha = write_json(gate_path, gate_payload)
    written["paper_go_gate.json"] = {
        "path": str(gate_path.relative_to(REPO_ROOT)),
        "sha256": gate_sha,
        "rows": len(conditions),
        "columns": 0,
        "description": "Mechanical gate evaluation; P2_PAPER_GO is never declared here.",
    }

    manifest = {
        "run_stamp": stamp.as_dict(),
        "config_path": _display_path(config_path),
        "stations": station_periods,
        "artefacts": written,
        "absent_artefacts": [],
    }
    write_json(out_dir / "outputs_manifest.json", manifest)

    LOGGER.info(
        "done: %d artefacts, gate summary %s",
        len(written),
        gate_payload["_summary"]["counts"],
    )
    print("P2_RUN_COMPLETE", gate_payload["_summary"]["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
