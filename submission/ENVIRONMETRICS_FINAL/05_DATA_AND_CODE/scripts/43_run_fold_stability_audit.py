#!/usr/bin/env python3
"""
43 — Per-Fold Dynamic Fidelity & Ghost Skill Stability Audit Runner
Evaluates metrics per fold ensuring intra-fold contiguous step differences for temporal_variability.
Exports fold stability source tables with SHA-256 metadata.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.exceedance_adapter import normalize_predictions_schema, check_case_alignment
from src.evaluation.fold_stability import compute_fold_level_metrics, summarize_sarima_fold_stability

INPUT_PARQUET = ROOT_DIR / "outputs" / "reproduction" / "predictions_rolling_origin.parquet"
OUTPUT_DIR = ROOT_DIR / "outputs" / "source_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCER_COMMIT = "4909e048e0b9f516031b9e217be0b806fa9dfb8b"
ANALYSIS_COMMIT = "4909e048e0b9f516031b9e217be0b806fa9dfb8b"
EVIDENCE_LABEL = "B_HIGH_SOURCE_PROVENANCE_PENDING"
STATION_STATUS = "MISSING_FROM_SOURCE"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def add_provenance_metadata(df: pd.DataFrame, input_file: Path, input_sha256: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "input_file", input_file.name)
    out.insert(1, "input_sha256", input_sha256)
    out.insert(2, "producer_commit", PRODUCER_COMMIT)
    out.insert(3, "analysis_commit", ANALYSIS_COMMIT)
    out.insert(4, "execution_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat())
    out.insert(5, "evidence_label", EVIDENCE_LABEL)
    out.insert(6, "station_status", STATION_STATUS)
    return out


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PARQUET}")

    input_sha256 = compute_sha256(INPUT_PARQUET)
    print(f"Loaded {INPUT_PARQUET.name} (SHA-256: {input_sha256})")

    df_raw = pd.read_parquet(INPUT_PARQUET)
    df_norm, _ = normalize_predictions_schema(df_raw)

    aligned, _ = check_case_alignment(df_norm)
    if not aligned:
        raise ValueError("Common cases misaligned!")

    print(f"Verified {len(df_norm)} total observations across models.")

    # 1. Compute fold-level metrics table
    df_fold_metrics = compute_fold_level_metrics(df_norm)
    df_fold_out = add_provenance_metadata(df_fold_metrics, INPUT_PARQUET, input_sha256)
    df_fold_out.to_csv(OUTPUT_DIR / "fold_stability_by_model_horizon_fold.csv", index=False)
    print("Exported fold_stability_by_model_horizon_fold.csv")

    # 2. Compute SARIMA fold summary table
    df_sarima_summary = summarize_sarima_fold_stability(df_fold_metrics)
    df_summary_out = add_provenance_metadata(df_sarima_summary, INPUT_PARQUET, input_sha256)
    df_summary_out.to_csv(OUTPUT_DIR / "fold_stability_summary_sarima.csv", index=False)
    print("Exported fold_stability_summary_sarima.csv")

    print(f"\nSUCCESS: Completed fold stability audit. Tables written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
