#!/usr/bin/env python3
"""
Project: P4 — Ghost Skill & Dynamic Fidelity
Role: auxiliary exceedance and rank-reversal diagnostic (CLI)
Evidence status: DEMO_SYNTHETIC for `demo`; REAL_DATA_UNVERIFIED for `probe`
(no producer-pipeline leakage/train-only audit exists yet; see
docs/p4_exceedance_module.md and audit_followup_report.md).

Usage
-----
    python scripts/p4_run_exceedance_diagnostic.py demo
    python scripts/p4_run_exceedance_diagnostic.py probe

`demo` runs on fully synthetic data (labeled DEMO_SYNTHETIC) and is safe to
run anywhere. `probe` only runs if outputs/metrics/predictions*.csv (this
repository's real row-level prediction tables) are present, and its output
stays REAL_DATA_UNVERIFIED — it must not be read as a scientific finding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.p4_exceedance.schema_adapter import adapt_schema, SchemaAdapterError
from src.evaluation.p4_exceedance.contract import (
    validate_contract,
    classify_evidence_status,
    ProducerEvidence,
)
from src.evaluation.p4_exceedance.integrity_checks import detect_duplicates, check_common_support
from src.evaluation.p4_exceedance.ranking_comparison import ranking_comparison
from src.evaluation.p4_exceedance.classification import classify_reversal_from_row
from src.evaluation.p4_exceedance.metrics import compute_full_event_metrics
from src.evaluation.p4_exceedance.manifest import build_manifest

OUTPUT_ROOT = ROOT / "outputs" / "p4_exceedance"

# Station identity for these files is not a literal column in the source
# CSVs; it is documented in this repository's own read-only audit
# (audit/trazabilidad_tres_estaciones.md, section 2-3), which maps
# `dataset` -> station for exactly these three files. Not inferred here.
REAL_STATION_FILES = {
    "outputs/metrics/predictions.csv": "Elche",
    "outputs/metrics/predictions_valencia_vivers.csv": "Valencia-Vivers",
    "outputs/metrics/predictions_zarra_emep.csv": "Zarra",
}
BASELINE_MODEL = "persistence"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Synthetic demo data (Fase 12.4)
# ---------------------------------------------------------------------------


def build_synthetic_dataset(seed: int = 0, n_folds: int = 60) -> pd.DataFrame:
    """Fully synthetic row-level data, DEMO_SYNTHETIC only.

    Two stations, engineered so that "north" agrees between the continuous
    (skill) and event (CSI) rankings, while "south" disagrees: model_a has
    better average accuracy (higher skill) but specifically under-predicts
    during true exceedance events (worse CSI) than model_b. This exercises
    the YES/NO classification paths without claiming anything about real
    PM10 forecasts.
    """
    rng = np.random.default_rng(seed)
    horizon = 1
    origins = pd.date_range("2021-01-01", periods=n_folds, freq="D")
    rows = []

    for station in ("north", "south"):
        y_true = rng.gamma(shape=2.0, scale=15.0, size=n_folds)
        threshold = np.percentile(y_true, 75)
        is_event = y_true > threshold

        y_pred_persistence = 0.5 * y_true + rng.normal(0, 10.0, size=n_folds)

        if station == "south":
            # model_a: near-perfect except it caps its own forecast just
            # under the threshold whenever the true event occurs (hides
            # exceedances -> CSI=0 there), keeping overall error small.
            # model_b: noisier everywhere, but tracks true exceedances
            # closely when they occur -> lower skill, higher CSI.
            y_pred_a = np.where(
                is_event, threshold - 1.0, y_true + rng.normal(0, 0.5, size=n_folds)
            )
            y_pred_b = np.where(
                is_event,
                y_true + rng.normal(0, 1.5, size=n_folds),
                0.55 * y_true + rng.normal(0, 18.0, size=n_folds),
            )
        else:
            # model_a strictly dominates model_b on both skill and CSI.
            y_pred_a = y_true + rng.normal(0, 0.5, size=n_folds)
            y_pred_b = 0.55 * y_true + rng.normal(0, 6.0, size=n_folds)

        for model, y_pred in (
            (BASELINE_MODEL, y_pred_persistence),
            ("model_a", y_pred_a),
            ("model_b", y_pred_b),
        ):
            for i, origin in enumerate(origins):
                rows.append(
                    {
                        "station": station,
                        "model": model,
                        "origin_date": origin.strftime("%Y-%m-%d"),
                        "date": (origin + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d"),
                        "horizon": horizon,
                        "fold_id": origin.strftime("%Y-%m-%d"),
                        "y_true": float(y_true[i]),
                        "y_pred": float(y_pred[i]),
                    }
                )
    return pd.DataFrame(rows)


def build_integrity_issue_dataset() -> pd.DataFrame:
    """Small dataset with a deliberate duplicate and a deliberate
    case-misalignment, purely to exercise/demonstrate S4 and S5.
    """
    origins = pd.date_range("2022-01-01", periods=5, freq="D")
    rows = []
    for i, origin in enumerate(origins):
        target = origin + pd.Timedelta(days=1)
        rows.append(
            {
                "station": "demo_station", "model": "model_x", "origin_date": origin.strftime("%Y-%m-%d"),
                "date": target.strftime("%Y-%m-%d"), "horizon": 1, "fold_id": origin.strftime("%Y-%m-%d"),
                "y_true": 10.0 + i, "y_pred": 9.0 + i,
            }
        )
        if i != 3:  # model_y is missing the fold_id for i == 3 -> misalignment
            rows.append(
                {
                    "station": "demo_station", "model": "model_y", "origin_date": origin.strftime("%Y-%m-%d"),
                    "date": target.strftime("%Y-%m-%d"), "horizon": 1, "fold_id": origin.strftime("%Y-%m-%d"),
                    "y_true": 10.0 + i, "y_pred": 8.5 + i,
                }
            )
    rows.append(dict(rows[0]))  # exact duplicate of model_x's first row
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shared pipeline
# ---------------------------------------------------------------------------


def compute_metrics_table(
    adapted: pd.DataFrame,
    *,
    threshold_percentile: float = 75.0,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate row-level predictions into one (station, horizon, model)
    row with a continuous metric (skill vs. persistence) and an event
    metric (CSI). The event threshold is a percentile of true observations
    within the same (station, horizon) evaluation slice, so it is reported
    as POST_HOC_DIAGNOSTIC, never as a calibrated threshold.
    """
    metric_rows = []
    n_events_total = 0
    thresholds_used = {}

    for (station, horizon), station_df in adapted.groupby(["station", "horizon"]):
        persistence_df = station_df[station_df["model"] == BASELINE_MODEL]
        if persistence_df.empty:
            continue
        threshold = float(np.percentile(persistence_df["y_true"], threshold_percentile))
        thresholds_used[(station, horizon)] = threshold

        for model, model_df in station_df.groupby("model"):
            if model == BASELINE_MODEL:
                continue
            merged = model_df.merge(
                persistence_df[["origin_date", "target_date", "y_true"]],
                on=["origin_date", "target_date"],
                suffixes=("", "_persist"),
                how="inner",
            )
            if merged.empty:
                continue
            rmse_model = float(np.sqrt(np.mean((merged["y_pred"] - merged["y_true"]) ** 2)))
            rmse_persist = float(np.sqrt(np.mean((persistence_df["y_pred"] - persistence_df["y_true"]) ** 2)))
            skill = 1 - (rmse_model / rmse_persist) if rmse_persist > 0 else np.nan

            event_metrics = compute_full_event_metrics(merged["y_true"], merged["y_pred"], threshold)
            n_events_total += event_metrics["n_events_true"]

            metric_rows.append(
                {
                    "station": station,
                    "horizon": horizon,
                    "model": model,
                    "n_cases": len(merged),
                    "threshold": threshold,
                    "skill": skill,
                    "csi": event_metrics["csi"],
                    "hit_rate": event_metrics["hit_rate"],
                    "false_alarm_rate": event_metrics["false_alarm_rate"],
                    "precision": event_metrics["precision"],
                    "event_bias": event_metrics["event_bias"],
                    "exceedance_intensity_error": event_metrics["exceedance_intensity_error"],
                }
            )

    table = pd.DataFrame(metric_rows)
    return table, {"n_events_total": n_events_total, "thresholds_used": thresholds_used}


def run_diagnostic(
    raw_df: pd.DataFrame,
    *,
    label: str,
    data_source: str,
    output_dir: Path,
    resolution: str | None = None,
    producer_evidence: ProducerEvidence | None = None,
    input_predictions_path: str | None = None,
    input_predictions_sha256: str | None = None,
    dataset: str | None = None,
    station: str | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    t0 = time.perf_counter()

    contract_report = validate_contract(raw_df)
    print(f"  [{label}] contract valid={contract_report.is_valid}, "
          f"missing_required={contract_report.missing_required}")

    t_adapt = time.perf_counter()
    adapted, schema_report = adapt_schema(raw_df, resolution=resolution)
    timings["schema_adapt_s"] = time.perf_counter() - t_adapt
    print(f"  [{label}] schema adapted: source={schema_report.target_date_source}, "
          f"resolution={schema_report.resolution}, n_rows={schema_report.n_rows}")

    t_dup = time.perf_counter()
    dup_report = detect_duplicates(adapted)
    timings["duplicate_check_s"] = time.perf_counter() - t_dup
    dup_report.duplicate_keys.to_csv(output_dir / "duplicate_report.csv", index=False)
    print(f"  [{label}] duplicates: n={dup_report.n_duplicates}, "
          f"affected_models={dup_report.affected_models}")

    t_align = time.perf_counter()
    group_cols = [c for c in ("station", "horizon") if c in adapted.columns]
    align_report = check_common_support(adapted, group_columns=group_cols)
    timings["alignment_check_s"] = time.perf_counter() - t_align
    align_report.misalignment_table.to_csv(output_dir / "case_alignment_table.csv", index=False)
    print(f"  [{label}] alignment: is_aligned={align_report.is_aligned}, "
          f"n_misaligned_groups={len(align_report.misaligned_groups)}")

    ranking_misaligned = set(align_report.misaligned_groups)

    if dup_report.has_duplicates:
        dup_groups = (
            adapted.loc[adapted.duplicated(subset=dup_report.key_columns, keep=False), ["station", "horizon"]]
            .drop_duplicates()
        )
        ranking_misaligned |= set(map(tuple, dup_groups.to_numpy()))
        print(f"  [{label}] WARNING: duplicates found; affected (station, horizon) groups "
              f"forced to NOT_EVALUABLE_NO_COMMON_CASES: {sorted(ranking_misaligned)}")

    t_metrics = time.perf_counter()
    metrics_table, metrics_meta = compute_metrics_table(adapted)
    timings["metrics_compute_s"] = time.perf_counter() - t_metrics
    metrics_table.to_csv(output_dir / "metrics_table.csv", index=False)

    t_rank = time.perf_counter()
    ranking_table = ranking_comparison(
        metrics_table,
        metric_continuous_col="skill",
        metric_event_col="csi",
        metric_continuous_name="skill_vs_persistence",
        metric_event_name="csi",
        misaligned_groups=ranking_misaligned,
    )
    if len(ranking_table) > 0:
        ranking_table["reversal_classification"] = ranking_table.apply(
            lambda row: classify_reversal_from_row(row.to_dict(), both_families_tested=True), axis=1
        )
    else:
        ranking_table["reversal_classification"] = pd.Series(dtype="object")
    timings["ranking_comparison_s"] = time.perf_counter() - t_rank
    ranking_table.to_csv(output_dir / "ranking_comparison_table.csv", index=False)

    evidence_status = classify_evidence_status(data_source, producer_evidence)

    manifest = build_manifest(
        input_predictions_path=input_predictions_path,
        input_predictions_sha256=input_predictions_sha256,
        input_schema=schema_report.to_dict(),
        schema_adapter={
            "target_date_source": schema_report.target_date_source,
            "origin_date_source": schema_report.origin_date_source,
            "notes": schema_report.notes,
        },
        producer_repository=(producer_evidence.producer_repository if producer_evidence else None),
        producer_commit=(producer_evidence.producer_commit if producer_evidence else None),
        dataset=dataset,
        station=station,
        period=None,
        resolution=schema_report.resolution,
        forecast_horizons=sorted(adapted["horizon"].unique().tolist()),
        rolling_origin_protocol=(producer_evidence.rolling_origin_protocol if producer_evidence else None),
        preprocessing_protocol=(producer_evidence.preprocessing_train_only if producer_evidence else None),
        event_threshold="per (station,horizon) P75 of persistence y_true (see metrics_table.csv)",
        threshold_source="p75_of_evaluation_period_diagnostic_only",
        threshold_mode="POST_HOC_DIAGNOSTIC",
        block_length=None,
        block_length_unit=None,
        block_length_justification=None,
        random_seed=None,
        models=sorted(adapted["model"].unique().tolist()),
        baselines=[BASELINE_MODEL],
        metrics_continuous=["skill_vs_persistence"],
        metrics_event=["csi", "hit_rate", "false_alarm_rate", "precision", "event_bias", "exceedance_intensity_error"],
        common_case_status="ALIGNED" if align_report.is_aligned else "MISALIGNED",
        duplicate_status="CLEAN" if not dup_report.has_duplicates else f"{dup_report.n_duplicates}_DUPLICATE_ROWS",
        execution_timestamp=_now_iso(),
        execution_label=f"{label}::{evidence_status}",
    )
    with open(output_dir / "evaluation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    timings["total_s"] = time.perf_counter() - t0

    summary = {
        "label": label,
        "evidence_status": evidence_status,
        "n_rows": len(adapted),
        "n_stations": adapted["station"].nunique(),
        "n_models": adapted["model"].nunique(),
        "n_horizons": adapted["horizon"].nunique(),
        "n_duplicates": dup_report.n_duplicates,
        "is_aligned": align_report.is_aligned,
        "n_misaligned_groups": len(align_report.misaligned_groups),
        "n_events_observed": metrics_meta["n_events_total"],
        "ranking_rows": len(ranking_table),
        "classification_counts": (
            ranking_table["reversal_classification"].value_counts().to_dict() if len(ranking_table) else {}
        ),
        "timings_s": timings,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_demo() -> list:
    summaries = []
    demo_df = build_synthetic_dataset()
    summaries.append(
        run_diagnostic(
            demo_df,
            label="demo_synthetic_main",
            data_source="synthetic",
            output_dir=OUTPUT_ROOT / "demo_synthetic",
            dataset="synthetic_p4_exceedance_demo",
            station="synthetic (north, south)",
        )
    )

    integrity_df = build_integrity_issue_dataset()
    summaries.append(
        run_diagnostic(
            integrity_df,
            label="demo_synthetic_integrity_issues",
            data_source="synthetic",
            output_dir=OUTPUT_ROOT / "demo_synthetic" / "integrity_issue_showcase",
            dataset="synthetic_s4_s5_showcase",
            station="demo_station",
        )
    )
    return summaries


def run_probe() -> list:
    summaries = []
    for rel_path, station in REAL_STATION_FILES.items():
        csv_path = ROOT / rel_path
        if not csv_path.exists():
            print(f"  Skipping probe for {rel_path}: not found in repository.")
            continue

        raw_df = pd.read_csv(csv_path)
        raw_df["station"] = station  # 'dataset' encodes station; see audit/trazabilidad_tres_estaciones.md

        producer_evidence = ProducerEvidence(
            producer_repository="fedeg-umh-es/varret-pm10-paper",
            producer_commit=_git_head_commit(),
            dataset=raw_df["dataset"].iloc[0],
            station=station,
            # Deliberately left PENDING_VERIFICATION: no leakage / train-only
            # preprocessing audit of the producer pipeline exists yet.
            rolling_origin_protocol=None,
            preprocessing_train_only=None,
            baseline_explicit=BASELINE_MODEL,
            period=None,
            fold=None,
        )

        summary = run_diagnostic(
            raw_df,
            label=f"real_probe_{station.lower().replace('-', '_')}",
            data_source="real",
            output_dir=OUTPUT_ROOT / "real_probe_elche_equivalent" / station.lower().replace("-", "_"),
            dataset=raw_df["dataset"].iloc[0],
            station=station,
            producer_evidence=producer_evidence,
            input_predictions_path=rel_path,
            input_predictions_sha256=_sha256(csv_path),
        )
        assert summary["evidence_status"] == "REAL_DATA_UNVERIFIED", (
            "Real-data probe must stay REAL_DATA_UNVERIFIED until the producer "
            "pipeline is audited; refusing to proceed with a different label."
        )
        summaries.append(summary)
    return summaries


def _git_head_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return None


def _environment_report() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: __import__(name).__version__
            for name in ("pandas", "numpy", "scipy")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["demo", "probe", "all"], nargs="?", default="all")
    args = parser.parse_args()

    print("=== P4 exceedance & rank-reversal diagnostic ===")
    print(json.dumps(_environment_report(), indent=2))

    all_summaries = []
    if args.mode in ("demo", "all"):
        print("\n--- DEMO_SYNTHETIC ---")
        all_summaries.extend(run_demo())
    if args.mode in ("probe", "all"):
        print("\n--- REAL_DATA_UNVERIFIED probe ---")
        all_summaries.extend(run_probe())

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("\n=== Summary ===")
    for s in all_summaries:
        print(json.dumps(s, indent=2, default=str))
    print(f"\nApprox. peak memory (RSS): {peak_rss_kb / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
