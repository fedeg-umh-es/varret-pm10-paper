"""Unit tests for Manuscript Evidence Map Registry traceability and column verification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PUB_DIR = ROOT / "outputs" / "publication_tables"
REGISTRY_FILE = PUB_DIR / "manuscript_evidence_map_registry.csv"
EVID_COMMIT = "95c9cbdc8c582f5657523c404afa58e61f5e1137"
PACK_COMMIT = "f233a2080d8ff0428ef5bc1bd80cf8a62ddc6a78"


def test_registry_file_exists() -> None:
    assert REGISTRY_FILE.exists()


def test_registry_columns_and_commits() -> None:
    df_reg = pd.read_csv(REGISTRY_FILE)
    assert len(df_reg) >= 5
    assert (df_reg["evidence_source_commit"] == EVID_COMMIT).all()
    assert (df_reg["publication_packaging_commit"] == PACK_COMMIT).all()
    assert (df_reg["evidence_label"] == "B_HIGH_SOURCE_PROVENANCE_PENDING").all()
    assert (df_reg["station_status"] == "MISSING_FROM_SOURCE").all()


def test_registry_target_table_columns_exist() -> None:
    df_reg = pd.read_csv(REGISTRY_FILE)

    for _, row in df_reg.iterrows():
        table_file = PUB_DIR / row["source_table_file"]
        assert table_file.exists(), f"Source table missing: {row['source_table_file']}"

        df_target = pd.read_csv(table_file)
        cols = [c.strip() for c in row["source_columns"].split(",")]

        for col in cols:
            assert col in df_target.columns, f"Column '{col}' missing in {row['source_table_file']}"

        # Verify model and horizon row exist in target table
        matched = df_target[(df_target["model"] == row["model"]) & (df_target["horizon"] == int(row["horizon"]))]
        assert len(matched) > 0, f"No row matching model={row['model']} and horizon={row['horizon']} in {row['source_table_file']}"
