"""Unit tests for P4 Manuscript Architecture registry and evidence map traceability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PUB_DIR = ROOT / "outputs" / "publication_tables"
ARCH_FILE = PUB_DIR / "manuscript_architecture_registry.csv"
EVID_REG_FILE = PUB_DIR / "manuscript_evidence_map_registry.csv"
EVID_COMMIT = "95c9cbdc8c582f5657523c404afa58e61f5e1137"
PACK_COMMIT = "f233a2080d8ff0428ef5bc1bd80cf8a62ddc6a78"


def test_architecture_file_exists() -> None:
    assert ARCH_FILE.exists()


def test_architecture_rows_and_commits() -> None:
    df_arch = pd.read_csv(ARCH_FILE)
    assert len(df_arch) == 6  # 5 result blocks + 1 synthesis
    assert (df_arch["evidence_source_commit"] == EVID_COMMIT).all()
    assert (df_arch["publication_packaging_commit"] == PACK_COMMIT).all()
    assert (df_arch["evidence_label"] == "B_HIGH_SOURCE_PROVENANCE_PENDING").all()


def test_architecture_source_table_linkage() -> None:
    df_arch = pd.read_csv(ARCH_FILE)

    for _, row in df_arch.iterrows():
        table_path = PUB_DIR / row["primary_source_table"]
        assert table_path.exists(), f"Source table missing: {row['primary_source_table']}"

        df_target = pd.read_csv(table_path)
        matched = df_target[(df_target["model"] == row["target_model"]) & (df_target["horizon"] == int(row["target_horizon"]))]
        assert len(matched) > 0, f"No matching row for model={row['target_model']} and horizon={row['target_horizon']} in {row['primary_source_table']}"


def test_evidence_map_registry_alignment() -> None:
    df_arch = pd.read_csv(ARCH_FILE)
    df_evid = pd.read_csv(EVID_REG_FILE)

    # Verify that all primary source tables referenced in architecture exist in evidence map registry
    arch_tables = set(df_arch["primary_source_table"])
    evid_tables = set(df_evid["source_table_file"])
    assert arch_tables.issubset(evid_tables)
