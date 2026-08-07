#!/usr/bin/env python3
"""
44 — Publication Source Tables Exporter
Packages frozen evidence from commit 95c9cbdc8c582f5657523c404afa58e61f5e1137 into manuscript-traceable publication source tables.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

SOURCE_TABLES_DIR = ROOT_DIR / "outputs" / "source_tables"
PUB_TABLES_DIR = ROOT_DIR / "outputs" / "publication_tables"
PUB_TABLES_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_COMMIT = "95c9cbdc8c582f5657523c404afa58e61f5e1137"
INPUT_PARQUET = ROOT_DIR / "outputs" / "reproduction" / "predictions_rolling_origin.parquet"
EVIDENCE_LABEL = "B_HIGH_SOURCE_PROVENANCE_PENDING"
STATION_STATUS = "MISSING_FROM_SOURCE"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def add_publication_metadata(df: pd.DataFrame, input_sha256: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "source_commit", FROZEN_COMMIT)
    out.insert(1, "input_file", INPUT_PARQUET.name)
    out.insert(2, "input_sha256", input_sha256)
    out.insert(3, "execution_timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat())
    out.insert(4, "evidence_label", EVIDENCE_LABEL)
    out.insert(5, "station_status", STATION_STATUS)
    return out


def main() -> None:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PARQUET}")

    input_sha256 = compute_sha256(INPUT_PARQUET)
    print(f"Loaded {INPUT_PARQUET.name} for publication packaging (SHA-256: {input_sha256})")

    # Load canonical source tables
    df_event_metrics = pd.read_csv(SOURCE_TABLES_DIR / "event_metrics_by_model_horizon.csv")
    df_fidelity = pd.read_csv(SOURCE_TABLES_DIR / "dynamic_fidelity_by_model_horizon.csv")
    df_reversal = pd.read_csv(SOURCE_TABLES_DIR / "rank_reversal_table.csv")
    df_sarima_summary = pd.read_csv(SOURCE_TABLES_DIR / "fold_stability_summary_sarima.csv")
    df_ghost = pd.read_csv(SOURCE_TABLES_DIR / "ghost_skill_audit_table.csv")

    # -------------------------------------------------------------------------
    # Layer 1: Error-Based Performance (pub_table_1_error_metrics.csv)
    # Filter to primary threshold policy rows
    primary_event_rows = df_event_metrics[df_event_metrics["policy_name"] == "PRIMARY_FIXED_THRESHOLD"].copy()

    layer1_df = primary_event_rows[[
        "model", "horizon", "n", "rmse", "rmse_persistence", "skill_rmse"
    ]].copy()
    layer1_df = layer1_df.rename(columns={"n": "N"})
    layer1_df = add_publication_metadata(layer1_df, input_sha256)
    layer1_df.to_csv(PUB_TABLES_DIR / "pub_table_1_error_metrics.csv", index=False)
    print("Exported Layer 1: pub_table_1_error_metrics.csv")

    # -------------------------------------------------------------------------
    # Layer 2: Dynamic Fidelity (pub_table_2_dynamic_fidelity.csv)
    layer2_df = df_fidelity[[
        "model", "horizon", "N", "variance_retention", "std_ratio", "alpha_kge",
        "correlation", "amplitude_ratio", "temporal_variability", "event_amplitude_retention"
    ]].copy()
    layer2_df = add_publication_metadata(layer2_df, input_sha256)
    layer2_df.to_csv(PUB_TABLES_DIR / "pub_table_2_dynamic_fidelity.csv", index=False)
    print("Exported Layer 2: pub_table_2_dynamic_fidelity.csv")

    # -------------------------------------------------------------------------
    # Layer 3: Operational Event Representation (pub_table_3_event_metrics.csv)
    layer3_df = primary_event_rows[[
        "model", "horizon", "n", "tp", "fp", "fn", "tn", "pod", "far", "pofd",
        "csi", "precision", "event_bias", "exceedance_intensity_error"
    ]].copy()
    layer3_df = layer3_df.rename(columns={"n": "N"})
    layer3_df = add_publication_metadata(layer3_df, input_sha256)
    layer3_df.to_csv(PUB_TABLES_DIR / "pub_table_3_event_metrics.csv", index=False)
    print("Exported Layer 3: pub_table_3_event_metrics.csv")

    # -------------------------------------------------------------------------
    # Layer 4: Structural & Fold Evidence (pub_table_4_ghost_skill_structure.csv)
    layer4_df = df_ghost[[
        "model", "horizon", "N", "rmse_skill", "variance_retention", "temporal_variability",
        "POD", "CSI", "ghost_skill_status"
    ]].copy()

    # Merge rank reversal info for paread models
    rev_primary = df_reversal[df_reversal["policy_name"] == "PRIMARY_FIXED_THRESHOLD"].copy()

    reversal_map = {}
    for _, row in rev_primary.iterrows():
        h = int(row["horizon"])
        reversal_map[h] = {
            "rank_reversal_csi": row["rank_reversal_csi"],
            "rank_reversal_pod": row["rank_reversal_pod"],
            "kendall_taub_prediction_series": row["kendall_taub_prediction_series"],
        }

    layer4_df["rank_reversal_csi"] = layer4_df["horizon"].map(lambda h: reversal_map.get(h, {}).get("rank_reversal_csi", "NA"))
    layer4_df["rank_reversal_pod"] = layer4_df["horizon"].map(lambda h: reversal_map.get(h, {}).get("rank_reversal_pod", "NA"))
    layer4_df["kendall_taub_prediction_series"] = layer4_df["horizon"].map(lambda h: reversal_map.get(h, {}).get("kendall_taub_prediction_series", np.nan))

    # Merge SARIMA fold stability info
    sarima_sum_map = {}
    for _, row in df_sarima_summary.iterrows():
        key = (row["model"], int(row["horizon"]))
        sarima_sum_map[key] = {
            "folds_with_positive_skill": row["folds_with_positive_skill"],
            "folds_with_concordant_degradation": row["folds_with_concordant_degradation"],
            "dynamic_collapse_all_folds": row["dynamic_collapse_all_folds"],
            "complete_event_failure_all_folds": row["complete_event_failure_all_folds"],
            "stability_pattern": row["stability_pattern"],
        }

    layer4_df["folds_with_positive_skill"] = layer4_df.apply(lambda r: sarima_sum_map.get((r["model"], r["horizon"]), {}).get("folds_with_positive_skill", "NA"), axis=1)
    layer4_df["folds_with_concordant_degradation"] = layer4_df.apply(lambda r: sarima_sum_map.get((r["model"], r["horizon"]), {}).get("folds_with_concordant_degradation", "NA"), axis=1)
    layer4_df["dynamic_collapse_all_folds"] = layer4_df.apply(lambda r: sarima_sum_map.get((r["model"], r["horizon"]), {}).get("dynamic_collapse_all_folds", "NA"), axis=1)
    layer4_df["complete_event_failure_all_folds"] = layer4_df.apply(lambda r: sarima_sum_map.get((r["model"], r["horizon"]), {}).get("complete_event_failure_all_folds", "NA"), axis=1)
    layer4_df["stability_pattern"] = layer4_df.apply(lambda r: sarima_sum_map.get((r["model"], r["horizon"]), {}).get("stability_pattern", "NA"), axis=1)

    layer4_df = add_publication_metadata(layer4_df, input_sha256)
    layer4_df.to_csv(PUB_TABLES_DIR / "pub_table_4_ghost_skill_structure.csv", index=False)
    print("Exported Layer 4: pub_table_4_ghost_skill_structure.csv")

    print(f"\nSUCCESS: Completed publication packaging. 4 publication source tables exported to {PUB_TABLES_DIR}")


if __name__ == "__main__":
    main()
