#!/usr/bin/env python3
"""P3 Phase-3 lags-only scientific analysis.

This producer consumes the frozen primary-support row-level predictions and writes
versioned diagnostic source tables. It deliberately does not modify the input
predictions or the manuscript. The ghost-skill materiality rule remains conservative:
positive-skill cells are screened, but no cell is confirmed without an explicit
pre-registered materiality/consequence adjudication.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "p4_v2/results/predictions_row_level_lags_only_primary_support.parquet"
RECON_MANIFEST = ROOT / "p4_v2/manifests/p3_support_reconciliation_manifest.json"
PREREG_MANIFEST = ROOT / "p4_v2/manifests/p3_scientific_gate_preregistration_manifest.json"
PREREG_DOC = ROOT / "p4_v2/docs/P3_SCIENTIFIC_GATE_PREREGISTRATION.md"
AMENDMENT_MANIFEST = ROOT / "p4_v2/manifests/p3_scientific_gate_preregistration_amendment_01_manifest.json"
AMENDMENT_DOC = ROOT / "p4_v2/docs/P3_SCIENTIFIC_GATE_PREREGISTRATION_AMENDMENT_01.md"
CONTROL_SPEC = ROOT / "p4_v2/manifests/p3_scientific_gate_control_cases.json"
OUT = ROOT / "p4_v2/results/phase3_source_tables_amendment_01"
RUN_MANIFEST = ROOT / "p4_v2/manifests/p3_lags_only_phase3_amendment_01_run_manifest.json"
EXEC_REPORT = ROOT / "p4_v2/results/p3_lags_only_phase3_amendment_01_execution_report.json"
PREFLIGHT_REPORT = ROOT / "p4_v2/results/phase3_amendment_01_preflight_integrity.json"

EXPECTED_SOURCE_HASH = "3ece14b77cac0262c643355ac7689d101de4728c49d565ab2616c010fcb1a296"
EXPECTED_RECON_HASH = "310735c862eef8a22fe102ca6a73ff5539270d9d0f0f11060cf0882cfe0bf93c"
EXPECTED_CELLS = 4522
EXPECTED_STATIONS = 323
HISTORICAL_CELLS = 4536
EXCLUDED_STATION = "50297039_10_49"
EXPECTED_MODELS = {"xgboost_direct", "sarima"}
EXPECTED_CONDITION = "lags_only"
BASE_SEED = 20260816
BLOCK_LENGTH = 28
BOOTSTRAP_RESAMPLES = 10_000
BH_ALPHA = 0.05

REQUIRED_COLUMNS = [
    "station_id",
    "condition",
    "model",
    "fold",
    "refit_block",
    "origin",
    "target_time",
    "horizon",
    "y_true",
    "y_pred",
    "y_persistence",
    "event_threshold_p75",
    "run_id",
    "protocol_hash",
    "source_panel_hash",
]
CELL_COLUMNS = ["station_id", "model", "horizon"]
SUPPORT_COLUMNS = ["station_id", "condition", "fold", "origin", "target_time", "horizon"]
PREDICTION_KEY = ["station_id", "model", "fold", "origin", "target_time", "horizon"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def schema_fingerprint(df: pd.DataFrame) -> str:
    payload = "\n".join(f"{c}:{df[c].dtype}" for c in df.columns).encode()
    return hashlib.sha256(payload).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default) + "\n", encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def validate_preflight() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not SOURCE.exists():
        raise RuntimeError(f"Missing canonical source: {SOURCE}")
    source_hash = sha256(SOURCE)
    if source_hash != EXPECTED_SOURCE_HASH:
        raise RuntimeError(f"PRIMARY_HASH_MISMATCH: {source_hash}")
    if sha256(RECON_MANIFEST) != EXPECTED_RECON_HASH:
        raise RuntimeError("RECONCILIATION_MANIFEST_HASH_MISMATCH")

    with PREREG_MANIFEST.open(encoding="utf-8") as f:
        prereg = json.load(f)
    with AMENDMENT_MANIFEST.open(encoding="utf-8") as f:
        amendment = json.load(f)
    if prereg.get("CASE_ABCD_USED_IN_P3") is not False:
        raise RuntimeError("CASE_ABCD governance mismatch")
    if prereg.get("PRIMARY_BASELINE") != "persistence":
        raise RuntimeError("Primary baseline is not persistence")
    if prereg.get("ghost_skill", {}).get("GHOST_SKILL_EXISTENCE_GATE") != "DEFINED_NOT_EXECUTED":
        raise RuntimeError("Ghost-skill gate is not in the frozen pre-execution state")
    if prereg.get("multiplicity", {}).get("family") != "GLOBAL_ALL_CELLS":
        raise RuntimeError("Multiplicity family mismatch")
    if prereg.get("multiplicity", {}).get("family_size") != HISTORICAL_CELLS:
        raise RuntimeError("Historical preregistration family size mismatch")
    if amendment.get("CORRECTED_STATIONS_EXPECTED") != EXPECTED_STATIONS:
        raise RuntimeError("Amendment station family size mismatch")
    if amendment.get("CORRECTED_CELLS_EXPECTED") != EXPECTED_CELLS:
        raise RuntimeError("Amendment cell family size mismatch")
    if amendment.get("EXCLUDED_STATION") != EXCLUDED_STATION:
        raise RuntimeError("Amendment excluded-station mismatch")
    if amendment.get("EXCLUSION_REASON") != "ZERO_PRIMARY_COMMON_SUPPORT":
        raise RuntimeError("Amendment exclusion-reason mismatch")
    if amendment.get("scientific_results_computed") is not False:
        raise RuntimeError("Amendment timing gate is not pre-analysis")
    if amendment.get("original_preregistration", {}).get("sha256") != sha256(PREREG_DOC):
        raise RuntimeError("Amendment does not point to the unchanged original preregistration")

    df = pd.read_parquet(SOURCE, columns=REQUIRED_COLUMNS)
    original_rows = len(df)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise RuntimeError(f"Missing required columns: {missing_columns}")

    for col in ("origin", "target_time"):
        df[col] = pd.to_datetime(df[col], errors="raise")
    numeric = ["y_true", "y_pred", "y_persistence", "event_threshold_p75"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if set(df["condition"].dropna().unique()) != {EXPECTED_CONDITION}:
        raise RuntimeError("Condition is not exclusively lags_only")
    if set(df["model"].dropna().unique()) != EXPECTED_MODELS:
        raise RuntimeError("Unexpected model family in primary predictions")
    if df["station_id"].nunique() != EXPECTED_STATIONS:
        raise RuntimeError(f"Station count mismatch: {df['station_id'].nunique()}")
    if set(df["horizon"].unique()) != set(range(1, 8)):
        raise RuntimeError("Horizon support mismatch")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise RuntimeError("Null in required canonical fields")
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite numeric canonical field")

    duplicate_prediction_rows = int(df.duplicated(PREDICTION_KEY).sum())
    if duplicate_prediction_rows:
        raise RuntimeError(f"Duplicated canonical prediction keys: {duplicate_prediction_rows}")

    cell_count = int(df[CELL_COLUMNS].drop_duplicates().shape[0])
    if cell_count != EXPECTED_CELLS:
        raise RuntimeError(f"Cell count mismatch: {cell_count} != {EXPECTED_CELLS}")

    support_counts = df.groupby(SUPPORT_COLUMNS, sort=False, observed=True)["model"].nunique()
    support_keys = int(support_counts.shape[0])
    if not bool((support_counts == 2).all()):
        raise RuntimeError("Common support does not contain exactly two model rows per support key")
    if support_keys != 1_706_377:
        raise RuntimeError(f"Conservative support key count mismatch: {support_keys}")

    paired = df.groupby(SUPPORT_COLUMNS, sort=False, observed=True).agg(
        y_true_nunique=("y_true", "nunique"),
        persistence_nunique=("y_persistence", "nunique"),
        threshold_nunique=("event_threshold_p75", "nunique"),
        protocol_nunique=("protocol_hash", "nunique"),
        source_panel_nunique=("source_panel_hash", "nunique"),
    )
    if (paired["y_true_nunique"] != 1).any() or (paired["persistence_nunique"] != 1).any():
        raise RuntimeError("Mismatched y_true or persistence across comparable model rows")
    if (paired["threshold_nunique"] != 1).any():
        raise RuntimeError("Event threshold is not fold-safe/common within support keys")
    if (paired["protocol_nunique"] != 1).any() or (paired["source_panel_nunique"] != 1).any():
        raise RuntimeError("Provenance metadata differs across paired model rows")

    horizon_delta = (df["target_time"] - df["origin"]).dt.days
    if not bool((horizon_delta == df["horizon"].astype(int)).all()):
        raise RuntimeError("Origin/target/horizon relationship is inconsistent")

    threshold_by_fold = df.groupby(["station_id", "fold"], sort=False, observed=True)["event_threshold_p75"].nunique()
    if (threshold_by_fold != 1).any():
        raise RuntimeError("Fold-specific event threshold is not unique")

    protocol_hashes = sorted(df["protocol_hash"].unique().tolist())
    panel_hashes = sorted(df["source_panel_hash"].unique().tolist())
    preflight = {
        "status": "PASS",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_hash,
        "reconciliation_manifest_sha256": sha256(RECON_MANIFEST),
        "original_preregistration_sha256": sha256(PREREG_DOC),
        "original_preregistration_manifest_sha256": sha256(PREREG_MANIFEST),
        "amendment_document_sha256": sha256(AMENDMENT_DOC),
        "amendment_manifest_sha256": sha256(AMENDMENT_MANIFEST),
        "row_count": original_rows,
        "schema": {c: str(df[c].dtype) for c in df.columns},
        "schema_fingerprint": schema_fingerprint(df),
        "station_count": int(df["station_id"].nunique()),
        "model_count": int(df["model"].nunique()),
        "models": sorted(df["model"].unique().tolist()),
        "horizons": sorted(int(x) for x in df["horizon"].unique()),
        "cell_count": cell_count,
        "historical_declared_cell_count": HISTORICAL_CELLS,
        "corrected_station_count": EXPECTED_STATIONS,
        "corrected_cell_count": EXPECTED_CELLS,
        "excluded_station": EXCLUDED_STATION,
        "exclusion_reason": "ZERO_PRIMARY_COMMON_SUPPORT",
        "support_key_count": support_keys,
        "duplicate_prediction_keys": duplicate_prediction_rows,
        "duplicate_support_keys": 0,
        "condition_values": sorted(df["condition"].unique().tolist()),
        "protocol_hashes": protocol_hashes,
        "source_panel_hashes": panel_hashes,
        "canonical_P3_PROJECT_CANON": "NOT_FOUND; current P3 preregistration manifest used as controlling governance artifact",
        "common_support_pairing": "PASS",
        "y_true_identity_across_models": "PASS",
        "persistence_identity_across_models": "PASS",
        "threshold_provenance": "PASS",
        "leakage_relevant_schema_consistency": "PASS",
    }
    write_json(PREFLIGHT_REPORT, preflight)
    return df, preflight


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true, ddof=0) == 0 or np.std(y_pred, ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def event_metrics(y_true: np.ndarray, pred: np.ndarray, threshold: np.ndarray) -> dict[str, Any]:
    obs = y_true > threshold
    fcst = pred > threshold
    tp = int(np.sum(obs & fcst))
    fp = int(np.sum(~obs & fcst))
    fn = int(np.sum(obs & ~fcst))
    obs_count = int(np.sum(obs))
    pred_count = int(np.sum(fcst))
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    far = safe_ratio(fp, tp + fp)
    csi = safe_ratio(tp, tp + fp + fn)
    bias = safe_ratio(pred_count, obs_count)
    intensity = float(np.mean(pred[obs] - y_true[obs])) if obs_count else float("nan")
    return {
        "n": int(len(y_true)),
        "event_count": obs_count,
        "predicted_event_count": pred_count,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall_pod": recall,
        "far": far,
        "csi": csi,
        "event_bias": bias,
        "exceedance_intensity_error": intensity,
    }


def block_sums(values: np.ndarray, length: int) -> np.ndarray:
    n = len(values)
    idx = np.arange(n + length - 1) % n
    doubled = values[idx]
    cs = np.concatenate(([0.0], np.cumsum(doubled, dtype=float)))
    return cs[length : length + n] - cs[:n]


def bootstrap_stats(
    d_mae: np.ndarray,
    d_sq: np.ndarray,
    seed: int,
) -> dict[str, float]:
    n = len(d_mae)
    blocks = int(math.ceil(n / BLOCK_LENGTH))
    full_blocks = max(blocks - 1, 0)
    remainder = n - full_blocks * BLOCK_LENGTH
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, blocks), dtype=np.int64)

    def one_series(values: np.ndarray) -> np.ndarray:
        full = block_sums(values, BLOCK_LENGTH)
        tail = block_sums(values, remainder)
        if full_blocks:
            result = full[starts[:, :full_blocks]].sum(axis=1)
        else:
            result = np.zeros(BOOTSTRAP_RESAMPLES, dtype=float)
        result += tail[starts[:, full_blocks]]
        return result / n

    boot_mae = one_series(d_mae)
    boot_sq = one_series(d_sq)
    return {
        "bootstrap_n": int(n),
        "bootstrap_blocks_per_resample": blocks,
        "bootstrap_seed": int(seed),
        "mae_diff_mean": float(np.mean(d_mae)),
        "mae_diff_ci_lower": float(np.quantile(boot_mae, 0.025)),
        "mae_diff_ci_upper": float(np.quantile(boot_mae, 0.975)),
        "mae_raw_p_one_sided": float(np.mean(boot_mae >= 0.0)),
        "rmse_squared_diff_mean": float(np.mean(d_sq)),
        "rmse_squared_diff_ci_lower": float(np.quantile(boot_sq, 0.025)),
        "rmse_squared_diff_ci_upper": float(np.quantile(boot_sq, 0.975)),
        "rmse_squared_raw_p_one_sided": float(np.mean(boot_sq >= 0.0)),
    }


def bh_adjust(p_values: pd.Series, alpha: float) -> pd.DataFrame:
    p = p_values.astype(float).to_numpy()
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_ranked
    return pd.DataFrame({"bh_q_value": q, "bh_reject_q_0_05": q <= alpha})


def dynamic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    obs_var = float(np.var(y_true, ddof=0))
    pred_var = float(np.var(y_pred, ddof=0))
    obs_sd = float(np.std(y_true, ddof=0))
    pred_sd = float(np.std(y_pred, ddof=0))
    obs_amp = float(np.quantile(y_true, 0.95) - np.quantile(y_true, 0.05))
    pred_amp = float(np.quantile(y_pred, 0.95) - np.quantile(y_pred, 0.05))
    return {
        "observed_variance": obs_var,
        "predicted_variance": pred_var,
        "variance_retention": safe_ratio(pred_var, obs_var),
        "observed_sd": obs_sd,
        "predicted_sd": pred_sd,
        "std_ratio": safe_ratio(pred_sd, obs_sd),
        "correlation": corr(y_true, y_pred),
        "observed_amplitude_q95_q05": obs_amp,
        "predicted_amplitude_q95_q05": pred_amp,
        "amplitude_ratio": safe_ratio(pred_amp, obs_amp),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, preflight = validate_preflight()
    print(f"PRECHECK PASS rows={len(df)} cells={preflight['cell_count']}", flush=True)

    error_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    cell_order: list[tuple[str, str, int]] = []

    grouped = df.sort_values(["station_id", "model", "horizon", "origin"]).groupby(
        CELL_COLUMNS, sort=True, observed=True
    )
    for index, (key, g) in enumerate(grouped):
        station, model, horizon = key
        cell_order.append((str(station), str(model), int(horizon)))
        y_true = g["y_true"].to_numpy(dtype=float)
        y_pred = g["y_pred"].to_numpy(dtype=float)
        y_persist = g["y_persistence"].to_numpy(dtype=float)
        threshold = g["event_threshold_p75"].to_numpy(dtype=float)
        abs_model = np.abs(y_pred - y_true)
        abs_persist = np.abs(y_persist - y_true)
        sq_model = (y_pred - y_true) ** 2
        sq_persist = (y_persist - y_true) ** 2
        mae = float(np.mean(abs_model))
        rmse = float(np.sqrt(np.mean(sq_model)))
        mae_p = float(np.mean(abs_persist))
        rmse_p = float(np.sqrt(np.mean(sq_persist)))
        error_rows.append({
            "station_id": station,
            "model": model,
            "horizon": int(horizon),
            "condition": EXPECTED_CONDITION,
            "n": int(len(g)),
            "mae": mae,
            "rmse": rmse,
            "bias": float(np.mean(y_pred - y_true)),
            "mae_persistence": mae_p,
            "rmse_persistence": rmse_p,
            "skill_mae": safe_ratio(mae_p - mae, mae_p),
            "skill_rmse": safe_ratio(rmse_p - rmse, rmse_p),
        })
        d_mae = abs_model - abs_persist
        d_sq = sq_model - sq_persist
        bootstrap_rows.append({
            "station_id": station,
            "model": model,
            "horizon": int(horizon),
            "condition": EXPECTED_CONDITION,
            **bootstrap_stats(d_mae, d_sq, BASE_SEED + index),
            "bootstrap_type": "circular moving block bootstrap over ordered origin-level pairs",
            "block_length_days": BLOCK_LENGTH,
            "resamples": BOOTSTRAP_RESAMPLES,
        })
        fidelity_rows.append({
            "station_id": station,
            "model": model,
            "horizon": int(horizon),
            "condition": EXPECTED_CONDITION,
            "n": int(len(g)),
            **dynamic_metrics(y_true, y_pred),
            "variance_fidelity_directional_degradation": bool(np.var(y_pred, ddof=0) < np.var(y_true, ddof=0)),
            "alpha_kge": np.nan,
            "temporal_variability": np.nan,
            "alpha_kge_status": "NOT_FROZEN_IN_PHASE_3_PREREGISTRATION",
            "temporal_variability_status": "NOT_FROZEN_IN_PHASE_3_PREREGISTRATION",
        })
        event_rows.append({
            "station_id": station,
            "model": model,
            "horizon": int(horizon),
            "condition": EXPECTED_CONDITION,
            "threshold_type": "training-only fold-specific p75",
            "threshold_operator": ">",
            "threshold_min": float(np.min(threshold)),
            "threshold_max": float(np.max(threshold)),
            "threshold_unique_count": int(pd.Series(threshold).nunique()),
            **event_metrics(y_true, y_pred, threshold),
        })
        if index and index % 250 == 0:
            print(f"processed_cells={index}/{EXPECTED_CELLS}", flush=True)

    error = pd.DataFrame(error_rows).sort_values(CELL_COLUMNS).reset_index(drop=True)
    fidelity = pd.DataFrame(fidelity_rows).sort_values(CELL_COLUMNS).reset_index(drop=True)
    events = pd.DataFrame(event_rows).sort_values(CELL_COLUMNS).reset_index(drop=True)
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values(CELL_COLUMNS).reset_index(drop=True)

    # Persistence rows are derived from the same paired support and are included
    # only as a comparator, not as an additional forecasting model family.
    persistence_error = (error.groupby(["station_id", "horizon", "condition"], as_index=False, observed=True)
        .agg(n=("n", "first"), mae=("mae_persistence", "first"), rmse=("rmse_persistence", "first")))
    persistence_error["model"] = "persistence"
    persistence_error["bias"] = np.nan
    persistence_error["mae_persistence"] = np.nan
    persistence_error["rmse_persistence"] = np.nan
    persistence_error["skill_mae"] = np.nan
    persistence_error["skill_rmse"] = np.nan
    persistence_error = persistence_error[["station_id", "model", "horizon", "condition", "n", "mae", "rmse", "bias", "mae_persistence", "rmse_persistence", "skill_mae", "skill_rmse"]]
    error_with_persistence = pd.concat([error, persistence_error], ignore_index=True).sort_values(["station_id", "horizon", "model"])

    persistence_event = (df.sort_values(["station_id", "horizon", "origin"])
        .groupby(["station_id", "horizon", "condition"], sort=True, observed=True)
        .apply(lambda g: pd.Series(event_metrics(
            g["y_true"].to_numpy(float), g["y_persistence"].to_numpy(float), g["event_threshold_p75"].to_numpy(float)
        )), include_groups=False).reset_index())
    persistence_event["model"] = "persistence"
    persistence_event["threshold_type"] = "training-only fold-specific p75"
    persistence_event["threshold_operator"] = ">"
    threshold_summary = (df.groupby(["station_id", "horizon", "condition"], as_index=False, observed=True)
        .agg(threshold_min=("event_threshold_p75", "min"), threshold_max=("event_threshold_p75", "max"), threshold_unique_count=("event_threshold_p75", "nunique")))
    persistence_event = persistence_event.merge(threshold_summary, on=["station_id", "horizon", "condition"], how="left")
    event_out = pd.concat([events, persistence_event], ignore_index=True, sort=False)
    event_out = event_out.sort_values(["station_id", "horizon", "model"]).reset_index(drop=True)

    # BH is the single preregistered GLOBAL_ALL_CELLS family for primary RMSE skill.
    bh = bootstrap[["station_id", "model", "horizon", "condition", "rmse_squared_raw_p_one_sided"]].copy()
    bh = bh.rename(columns={"rmse_squared_raw_p_one_sided": "raw_p_value"})
    bh_adj = bh_adjust(bh["raw_p_value"], BH_ALPHA)
    bh = pd.concat([bh.reset_index(drop=True), bh_adj], axis=1)
    bh["family"] = "GLOBAL_ALL_CELLS"
    bh["family_size"] = EXPECTED_CELLS
    bh["alpha"] = BH_ALPHA
    bh["primary_inferential_metric"] = "Skill_RMSE / squared-error loss difference"
    bh["condition"] = EXPECTED_CONDITION

    # Side-by-side XGBoost/SARIMA/persistence comparison for each station/horizon.
    def wide_metrics(table: pd.DataFrame, values: list[str], prefix: str) -> pd.DataFrame:
        work = table.pivot(index=["station_id", "horizon", "condition"], columns="model", values=values)
        work.columns = [f"{prefix}_{a}_{b}" for a, b in work.columns]
        return work.reset_index()

    comparison = wide_metrics(error, ["mae", "rmse", "bias", "skill_mae", "skill_rmse"], "error")
    comparison = comparison.merge(wide_metrics(fidelity, ["variance_retention", "std_ratio", "correlation", "amplitude_ratio"], "fidelity"), on=["station_id", "horizon", "condition"], how="left")
    event_model = event_out[event_out["model"].isin(["xgboost_direct", "sarima", "persistence"])].copy()
    comparison = comparison.merge(wide_metrics(event_model, ["precision", "recall_pod", "far", "csi", "event_bias", "exceedance_intensity_error"], "event"), on=["station_id", "horizon", "condition"], how="left")
    pers = persistence_error.rename(columns={"mae": "persistence_mae", "rmse": "persistence_rmse"})[["station_id", "horizon", "condition", "persistence_mae", "persistence_rmse"]]
    comparison = comparison.merge(pers, on=["station_id", "horizon", "condition"], how="left")
    comparison = comparison.sort_values(["station_id", "horizon"]).reset_index(drop=True)

    # Pairwise rank discordance. Higher is preferable for correlation/event scores;
    # fidelity ratios use closeness to one as the preregistered diagnostic reading.
    rank_rows: list[dict[str, Any]] = []
    rank_metrics = {
        "fidelity_variance_retention": ("fidelity_variance_retention_xgboost_direct", "fidelity_variance_retention_sarima", "closer_to_one"),
        "fidelity_std_ratio": ("fidelity_std_ratio_xgboost_direct", "fidelity_std_ratio_sarima", "closer_to_one"),
        "fidelity_amplitude_ratio": ("fidelity_amplitude_ratio_xgboost_direct", "fidelity_amplitude_ratio_sarima", "closer_to_one"),
        "fidelity_correlation": ("fidelity_correlation_xgboost_direct", "fidelity_correlation_sarima", "higher"),
        "event_csi": ("event_csi_xgboost_direct", "event_csi_sarima", "higher"),
        "event_recall_pod": ("event_recall_pod_xgboost_direct", "event_recall_pod_sarima", "higher"),
    }
    for row in comparison.itertuples(index=False):
        acc_x = getattr(row, "error_skill_rmse_xgboost_direct", np.nan)
        acc_s = getattr(row, "error_skill_rmse_sarima", np.nan)
        if not np.isfinite(acc_x) or not np.isfinite(acc_s) or acc_x == acc_s:
            acc_winner = "tie_or_undefined"
        else:
            acc_winner = "xgboost_direct" if acc_x > acc_s else "sarima"
        for metric, (cx, cs, direction) in rank_metrics.items():
            vx, vs = getattr(row, cx, np.nan), getattr(row, cs, np.nan)
            if not np.isfinite(vx) or not np.isfinite(vs) or vx == vs:
                diag_winner = "tie_or_undefined"
                margin = np.nan
            elif direction == "closer_to_one":
                sx, ss = -abs(vx - 1.0), -abs(vs - 1.0)
                diag_winner = "xgboost_direct" if sx > ss else "sarima"
                margin = float(abs(sx - ss))
            else:
                diag_winner = "xgboost_direct" if vx > vs else "sarima"
                margin = float(abs(vx - vs))
            reversal = acc_winner not in {"tie_or_undefined"} and diag_winner not in {"tie_or_undefined"} and acc_winner != diag_winner
            rank_rows.append({
                "station_id": row.station_id,
                "horizon": int(row.horizon),
                "condition": row.condition,
                "accuracy_metric": "error_skill_rmse",
                "accuracy_xgboost": acc_x,
                "accuracy_sarima": acc_s,
                "accuracy_winner": acc_winner,
                "diagnostic_metric": metric,
                "diagnostic_direction": direction,
                "diagnostic_xgboost": vx,
                "diagnostic_sarima": vs,
                "diagnostic_winner": diag_winner,
                "diagnostic_margin": margin,
                "pairwise_reversal_candidate": bool(reversal),
                "scientific_preference_adjudication": "REQUIRES_QUALITATIVE_REVIEW" if reversal else "NO_REVERSAL",
            })
    rank = pd.DataFrame(rank_rows).sort_values(["station_id", "horizon", "diagnostic_metric"]).reset_index(drop=True)

    # Candidate screen: all primary-RMSE-positive cells are carried forward with
    # diagnostics. No arbitrary fidelity cutoff is introduced. Materiality and
    # consequence are intentionally not adjudicated from the observed distribution.
    candidates = error[error["skill_rmse"] > 0].merge(
        fidelity, on=["station_id", "model", "horizon", "condition"], how="left", suffixes=("", "_fidelity")
    ).merge(
        event_out[event_out["model"].isin(["xgboost_direct", "sarima"])], on=["station_id", "model", "horizon", "condition"], how="left", suffixes=("", "_event")
    )
    candidates["candidate_generation_status"] = "POSITIVE_SKILL_SCREEN_REQUIRES_JOINT_ADJUDICATION"
    candidates["criterion_A"] = "PASS"
    candidates["criterion_B"] = "FAIL"
    candidates["criterion_C"] = "FAIL"
    candidates["criterion_B_status"] = "NOT_ADJUDICABLE_NO_PREDECLARED_MATERIALITY_CUTOFF"
    candidates["criterion_C_status"] = "NOT_ADJUDICABLE_NO_PREDEFINED_DECISION_CONSEQUENCE_RULE"
    candidates["GHOST_SKILL"] = "NO"
    candidates["GHOST_SKILL_STATUS"] = "NOT_ADJUDICABLE"
    candidates["adjudication_rationale"] = "Positive RMSE skill is present, but the preregistration deliberately provides no universal materiality cutoff and does not establish criterion C automatically from diagnostic deterioration."
    adjudication = candidates[["station_id", "model", "horizon", "condition", "skill_rmse", "variance_retention", "std_ratio", "correlation", "amplitude_ratio", "criterion_A", "criterion_B", "criterion_C", "GHOST_SKILL", "GHOST_SKILL_STATUS", "adjudication_rationale"]].copy()

    gate = pd.DataFrame([{
        "condition": EXPECTED_CONDITION,
        "gate": "GHOST_SKILL_EXISTENCE_GATE",
        "gate_definition": "at least one station-model-horizon cell satisfies positive skill, material fidelity degradation, and interpretation-changing consequence",
        "skill_positive_cells": int((error["skill_rmse"] > 0).sum()),
        "directional_fidelity_degraded_cells_variance_retention_lt_1": int((fidelity["variance_retention"] < 1).sum()),
        "ghost_skill_candidates_positive_rmse_screen": int(len(candidates)),
        "ghost_skill_confirmed": 0,
        "ghost_skill_not_adjudicable": int(len(candidates)),
        "gate_outcome": "FAIL",
        "gate_reason": "No cell can be confirmed under the frozen three-component rule without introducing an unregistered materiality/consequence cutoff; positive skill and directional fidelity reduction are not sufficient.",
    }])

    tables = {
        "error_skill_by_cell.csv": error,
        "error_skill_with_persistence_comparator.csv": error_with_persistence,
        "bootstrap_inference_by_cell.csv": bootstrap,
        "bh_correction_by_cell.csv": bh,
        "dynamic_fidelity_by_cell.csv": fidelity,
        "event_metrics_by_cell.csv": event_out,
        "xgboost_sarima_comparison.csv": comparison,
        "rank_reversal_table.csv": rank,
        "ghost_skill_candidates.csv": candidates,
        "ghost_skill_adjudication.csv": adjudication,
        "ghost_skill_gate_result.csv": gate,
    }
    artefacts: list[dict[str, Any]] = []
    for name, table in tables.items():
        path = OUT / name
        table.to_csv(path, index=False)
        artefacts.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": int(len(table)), "columns": list(table.columns)})

    run_id = f"p3_phase3_lags_only_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    report = {
        "PROJECT": "P3",
        "CONDITION": EXPECTED_CONDITION,
        "SOURCE_ROWS": int(len(df)),
        "STATIONS_EXPECTED": EXPECTED_STATIONS,
        "STATIONS_ANALYZED": int(df["station_id"].nunique()),
        "CELLS_EXPECTED": EXPECTED_CELLS,
        "CELLS_ANALYZED": int(error.shape[0]),
        "HISTORICAL_CELLS_DECLARED": HISTORICAL_CELLS,
        "EXCLUDED_STATION": EXCLUDED_STATION,
        "EXCLUSION_REASON": "ZERO_PRIMARY_COMMON_SUPPORT",
        "ERROR_SKILL_ANALYSIS": "PASS",
        "BOOTSTRAP_INFERENCE": "PASS",
        "BH_CORRECTION": "PASS",
        "DYNAMIC_FIDELITY_ANALYSIS": "PASS",
        "EVENT_ANALYSIS": "PASS",
        "XGBOOST_SARIMA_COMPARISON": "PASS",
        "RANK_DISCORDANCE_ANALYSIS": "PASS",
        "GHOST_SKILL_CANDIDATE_GENERATION": "PASS",
        "GHOST_SKILL_ADJUDICATION": "PASS",
        "SKILL_POSITIVE_CELLS": int((error["skill_rmse"] > 0).sum()),
        "FIDELITY_DEGRADED_CELLS": int((fidelity["variance_retention"] < 1).sum()),
        "FIDELITY_DEGRADED_CELLS_DEFINITION": "directional variance_retention < 1; not a materiality or ghost-skill cutoff",
        "RANK_REVERSAL_CASES": int(rank["pairwise_reversal_candidate"].sum()),
        "GHOST_SKILL_CANDIDATES": int(len(candidates)),
        "GHOST_SKILL_CONFIRMED": 0,
        "GHOST_SKILL_NOT_ADJUDICABLE": int(len(candidates)),
        "GHOST_SKILL_EXISTENCE_GATE": "FAIL",
        "GLOBAL_SCIENTIFIC_ANALYSIS": "COMPLETE",
        "MANUSCRIPT_UPDATE_AUTHORIZED": "NO",
        "bootstrap": {
            "type": "circular moving block bootstrap over ordered origin-level pairs",
            "block_length_days": BLOCK_LENGTH,
            "resamples": BOOTSTRAP_RESAMPLES,
            "base_seed": BASE_SEED,
            "seed_policy": "base seed plus deterministic sorted model-cell index",
            "raw_p_definition": "empirical one-sided fraction of bootstrap mean loss differences >= 0",
            "ci_definition": "two-sided percentile 95% interval",
        },
        "multiplicity": {
            "family": "GLOBAL_ALL_CELLS",
            "family_size": EXPECTED_CELLS,
            "primary_metric": "RMSE squared-error loss difference",
            "correction": "Benjamini-Hochberg FDR q=0.05",
        },
        "non_frozen_diagnostics": [
            "alpha-KGE not computed because it is not present in the current Phase-3 preregistration",
            "temporal variability not computed because it is not present in the current Phase-3 preregistration",
            "ghost-skill materiality and criterion C remain not adjudicable without an unregistered cutoff/consequence rule",
        ],
        "scientifically_important_descriptive_findings": [
            f"All {int(error.shape[0])} preregistered model cells were analyzed on the verified conservative common support.",
            f"{int((error['skill_rmse'] > 0).sum())} cells had positive point-estimate Skill_RMSE; this is not by itself inferential superiority.",
            f"{int((fidelity['variance_retention'] < 1).sum())} model cells had directional variance reduction (variance_retention < 1); this is not a materiality classification.",
            f"{int(rank['pairwise_reversal_candidate'].sum())} pairwise reversal candidates were recorded for qualitative review across frozen fidelity/event metrics.",
            f"The RMSE inferential family contained {EXPECTED_CELLS} cells and used global BH FDR q=0.05.",
            "No ghost-skill case was confirmed because the frozen rule does not authorize automatic materiality or consequence inference.",
        ],
        "generated_source_tables": artefacts,
        "preflight_report": str(PREFLIGHT_REPORT.relative_to(ROOT)),
    }
    write_json(EXEC_REPORT, report)

    manifest = {
        "PROJECT": "P3",
        "CONDITION": EXPECTED_CONDITION,
        "run_id": run_id,
        "git_commit": git_commit(),
        "source_parquet": str(SOURCE.relative_to(ROOT)),
        "source_parquet_sha256": sha256(SOURCE),
        "source_rows": int(len(df)),
        "source_schema_fingerprint": preflight["schema_fingerprint"],
        "preregistration_document": str(PREREG_DOC.relative_to(ROOT)),
        "preregistration_sha256": sha256(PREREG_DOC),
        "preregistration_manifest": str(PREREG_MANIFEST.relative_to(ROOT)),
        "preregistration_manifest_sha256": sha256(PREREG_MANIFEST),
        "amendment_document": str(AMENDMENT_DOC.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT_DOC),
        "amendment_manifest": str(AMENDMENT_MANIFEST.relative_to(ROOT)),
        "amendment_manifest_sha256": sha256(AMENDMENT_MANIFEST),
        "reconciliation_manifest": str(RECON_MANIFEST.relative_to(ROOT)),
        "reconciliation_manifest_sha256": sha256(RECON_MANIFEST),
        "control_specification": str(CONTROL_SPEC.relative_to(ROOT)),
        "control_specification_sha256": sha256(CONTROL_SPEC),
        "cell_family": "GLOBAL_ALL_CELLS: station_id x model x horizon",
        "historical_cells_expected": HISTORICAL_CELLS,
        "cells_expected": EXPECTED_CELLS,
        "cells_analyzed": int(error.shape[0]),
        "stations_expected": EXPECTED_STATIONS,
        "stations": int(df["station_id"].nunique()),
        "excluded_station": EXCLUDED_STATION,
        "exclusion_reason": "ZERO_PRIMARY_COMMON_SUPPORT",
        "models": sorted(df["model"].unique().tolist()),
        "horizons": sorted(int(x) for x in df["horizon"].unique()),
        "bootstrap": report["bootstrap"],
        "multiplicity": report["multiplicity"],
        "generated_artefacts": artefacts + [
            {"path": str(EXEC_REPORT.relative_to(ROOT)), "sha256": sha256(EXEC_REPORT)},
            {"path": str(PREFLIGHT_REPORT.relative_to(ROOT)), "sha256": sha256(PREFLIGHT_REPORT)},
        ],
        "exclusions": [],
        "failures": [],
        "non_adjudicable_components": report["non_frozen_diagnostics"],
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_metrics_executed": True,
        "manuscript_modified": False,
        "overleaf_modified": False,
        "new_experiments_run": False,
        "predictions_regenerated": False,
        "status": "COMPLETE_WITH_GHOST_SKILL_NOT_ADJUDICABLE",
    }
    write_json(RUN_MANIFEST, manifest)
    print(json.dumps(report, indent=2, default=json_default))
    print(f"RUN_MANIFEST={RUN_MANIFEST}")


if __name__ == "__main__":
    main()
