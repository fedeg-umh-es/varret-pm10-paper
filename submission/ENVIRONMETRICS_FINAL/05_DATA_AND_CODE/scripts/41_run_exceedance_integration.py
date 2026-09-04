#!/usr/bin/env python3
"""
41 — Exceedance Evaluation & Rank Reversal Module Integration
Runs canonical exceedance evaluation on outputs/reproduction/predictions_rolling_origin.parquet.
Exports standardized source tables with SHA-256 metadata under outputs/source_tables/.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.exceedance_adapter import (
    check_case_alignment,
    check_duplicates,
    classify_rank_reversal,
    compute_contingency_metrics,
    compute_kendall_taub,
    normalize_predictions_schema,
)
INPUT_PARQUET = ROOT_DIR / "outputs" / "reproduction" / "predictions_rolling_origin.parquet"
OUTPUT_DIR = ROOT_DIR / "outputs" / "source_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRODUCER_COMMIT = "4909e048e0b9f516031b9e217be0b806fa9dfb8b"
ANALYSIS_COMMIT = "4909e048e0b9f516031b9e217be0b806fa9dfb8b"
EVIDENCE_LABEL = "B_HIGH_SOURCE_PROVENANCE_PENDING"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def add_metadata(df: pd.DataFrame, input_file: Path, input_sha256: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "input_file", input_file.name)
    out.insert(1, "input_sha256", input_sha256)
    out.insert(2, "producer_commit", PRODUCER_COMMIT)
    out.insert(3, "analysis_commit", ANALYSIS_COMMIT)
    out.insert(4, "execution_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat())
    out.insert(5, "evidence_label", EVIDENCE_LABEL)
    return out


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PARQUET}")

    input_sha256 = compute_sha256(INPUT_PARQUET)
    print(f"Loaded {INPUT_PARQUET.name} (SHA-256: {input_sha256})")

    df_raw = pd.read_parquet(INPUT_PARQUET)
    df_norm, schema_meta = normalize_predictions_schema(df_raw)
    print(f"Normalized schema: {len(df_norm)} rows across models {sorted(df_norm['model'].unique())}")

    # 1. Duplicate check
    df_dups = check_duplicates(df_norm)
    df_dup_report = add_metadata(
        pd.DataFrame([{
            "duplicate_count": len(df_dups),
            "status": "DUPLICATES_FOUND" if len(df_dups) > 0 else "NO_DUPLICATES",
        }]),
        INPUT_PARQUET,
        input_sha256,
    )
    df_dup_report.to_csv(OUTPUT_DIR / "duplicate_report.csv", index=False)
    print(f"Exported duplicate_report.csv (duplicates: {len(df_dups)})")

    # 2. Case alignment check
    aligned, df_align_info = check_case_alignment(df_norm)
    df_align_report = add_metadata(df_align_info, INPUT_PARQUET, input_sha256)
    df_align_report.to_csv(OUTPUT_DIR / "case_alignment_report.csv", index=False)
    print(f"Exported case_alignment_report.csv (aligned: {aligned})")

    # 3. Exceedance counts by horizon
    count_rows = []
    for h in sorted(df_norm["horizon"].unique()):
        sub_h = df_norm[df_norm["horizon"] == h]
        n_obs = len(sub_h)
        n_exceed_p75 = int((sub_h["y_true"] > sub_h["p75_train"]).sum())
        n_exceed_abs50 = int((sub_h["y_true"] > 50.0).sum())
        count_rows.append({
            "horizon": h,
            "n_obs": n_obs,
            "n_exceed_p75_train": n_exceed_p75,
            "rate_p75_train": n_exceed_p75 / n_obs if n_obs > 0 else np.nan,
            "n_exceed_abs_50": n_exceed_abs50,
            "rate_abs_50": n_exceed_abs50 / n_obs if n_obs > 0 else np.nan,
        })
    df_counts = add_metadata(pd.DataFrame(count_rows), INPUT_PARQUET, input_sha256)
    df_counts.to_csv(OUTPUT_DIR / "exceedance_counts_by_horizon.csv", index=False)
    print("Exported exceedance_counts_by_horizon.csv")

    # 4. Compute Event Metrics by Model and Horizon
    metric_rows = []

    # Policy 1: PRIMARY_FIXED_THRESHOLD (fold_train_p75)
    # Policy 2: POST_HOC_DIAGNOSTIC (abs_50)
    policies = [
        ("PRIMARY_FIXED_THRESHOLD", "fold_train_p75", "VERIFIED_PRIMARY", None),
        ("POST_HOC_DIAGNOSTIC", "abs_50", "PENDING_DOMAIN_VERIFICATION", 50.0),
    ]

    for policy_name, policy_type, thresh_status, fixed_val in policies:
        for (model, h), group in df_norm.groupby(["model", "horizon"], sort=True):
            y_true = group["y_true"].to_numpy(dtype=float)
            y_pred = group["y_pred"].to_numpy(dtype=float)
            y_pers = group["y_persistence"].to_numpy(dtype=float)

            rmse_m = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            rmse_p = float(np.sqrt(np.mean((y_pers - y_true) ** 2)))
            skill_rmse = float(1.0 - (rmse_m / rmse_p))

            thresh = fixed_val if fixed_val is not None else group["p75_train"].to_numpy(dtype=float)
            m_dict = compute_contingency_metrics(y_true, y_pred, thresh)

            metric_rows.append({
                "policy_name": policy_name,
                "threshold_policy": policy_type,
                "threshold_status": thresh_status,
                "model": model,
                "horizon": h,
                "rmse": rmse_m,
                "rmse_persistence": rmse_p,
                "skill_rmse": skill_rmse,
                **m_dict,
            })

    df_event_metrics = add_metadata(pd.DataFrame(metric_rows), INPUT_PARQUET, input_sha256)
    df_event_metrics.to_csv(OUTPUT_DIR / "event_metrics_by_model_horizon.csv", index=False)
    print("Exported event_metrics_by_model_horizon.csv")

    # 5. Rank Reversal Table
    reversal_rows = []
    models = sorted(df_norm["model"].unique())

    if len(models) == 2 and aligned:
        m1, m2 = models[0], models[1]
        for policy_name in ["PRIMARY_FIXED_THRESHOLD", "POST_HOC_DIAGNOSTIC"]:
            sub_pol = df_event_metrics[df_event_metrics["policy_name"] == policy_name]
            for h in sorted(df_norm["horizon"].unique()):
                r1 = sub_pol[(sub_pol["model"] == m1) & (sub_pol["horizon"] == h)].iloc[0]
                r2 = sub_pol[(sub_pol["model"] == m2) & (sub_pol["horizon"] == h)].iloc[0]

                skill_diff = r1["skill_rmse"] - r2["skill_rmse"]
                csi_diff = r1["csi"] - r2["csi"]
                pod_diff = r1["pod"] - r2["pod"]

                rev_csi = classify_rank_reversal(skill_diff, csi_diff, 0.0)
                rev_pod = classify_rank_reversal(skill_diff, pod_diff, 0.0)

                sub_h = df_norm[df_norm["horizon"] == h]
                p1_vals = sub_h[sub_h["model"] == m1]["y_pred"].values
                p2_vals = sub_h[sub_h["model"] == m2]["y_pred"].values
                tau_val = compute_kendall_taub(p1_vals, p2_vals)

                reversal_rows.append({
                    "policy_name": policy_name,
                    "horizon": h,
                    "model_1": m1,
                    "model_2": m2,
                    "skill_diff_m1_minus_m2": skill_diff,
                    "csi_diff_m1_minus_m2": csi_diff,
                    "pod_diff_m1_minus_m2": pod_diff,
                    "rank_reversal_csi": rev_csi,
                    "rank_reversal_pod": rev_pod,
                    "kendall_taub_prediction_series": tau_val,
                })
    else:
        status_label = "NOT_EVALUABLE_NO_COMMON_CASES" if not aligned else "NOT_EVALUABLE_SINGLE_MODEL"
        for h in sorted(df_norm["horizon"].unique()):
            reversal_rows.append({
                "policy_name": "PRIMARY_FIXED_THRESHOLD",
                "horizon": h,
                "model_1": models[0] if models else "NONE",
                "model_2": models[1] if len(models) > 1 else "NONE",
                "skill_diff_m1_minus_m2": np.nan,
                "csi_diff_m1_minus_m2": np.nan,
                "pod_diff_m1_minus_m2": np.nan,
                "rank_reversal_csi": status_label,
                "rank_reversal_pod": status_label,
                "kendall_taub_prediction_series": np.nan,
            })

    df_reversal = add_metadata(pd.DataFrame(reversal_rows), INPUT_PARQUET, input_sha256)
    df_reversal.to_csv(OUTPUT_DIR / "rank_reversal_table.csv", index=False)
    print("Exported rank_reversal_table.csv")

    print(f"\nSUCCESS: Completed exceedance integration. All source tables exported to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
