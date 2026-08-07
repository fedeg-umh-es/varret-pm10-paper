"""Unit tests for publication source tables, 1:1 mathematical identity, and provenance metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PUB_DIR = ROOT / "outputs" / "publication_tables"
SOURCE_DIR = ROOT / "outputs" / "source_tables"
FROZEN_COMMIT = "95c9cbdc8c582f5657523c404afa58e61f5e1137"


def test_publication_tables_exist() -> None:
    expected_files = [
        "pub_table_1_error_metrics.csv",
        "pub_table_2_dynamic_fidelity.csv",
        "pub_table_3_event_metrics.csv",
        "pub_table_4_ghost_skill_structure.csv",
    ]
    for filename in expected_files:
        filepath = PUB_DIR / filename
        assert filepath.exists(), f"Missing publication source table: {filename}"


def test_publication_provenance_metadata() -> None:
    for filepath in PUB_DIR.glob("pub_table_*.csv"):
        df = pd.read_csv(filepath)
        assert "source_commit" in df.columns
        assert (df["source_commit"] == FROZEN_COMMIT).all()
        assert "evidence_label" in df.columns
        assert (df["evidence_label"] == "B_HIGH_SOURCE_PROVENANCE_PENDING").all()
        assert "station_status" in df.columns
        assert (df["station_status"] == "MISSING_FROM_SOURCE").all()


def test_layer1_identity() -> None:
    df_pub = pd.read_csv(PUB_DIR / "pub_table_1_error_metrics.csv")
    df_src = pd.read_csv(SOURCE_DIR / "event_metrics_by_model_horizon.csv")
    df_src_primary = df_src[df_src["policy_name"] == "PRIMARY_FIXED_THRESHOLD"]

    assert len(df_pub) == len(df_src_primary)
    for model in ["lightgbm", "sarima"]:
        for h in [1, 6, 24, 48]:
            pub_row = df_pub[(df_pub["model"] == model) & (df_pub["horizon"] == h)].iloc[0]
            src_row = df_src_primary[(df_src_primary["model"] == model) & (df_src_primary["horizon"] == h)].iloc[0]

            assert pytest.approx(pub_row["rmse"], abs=1e-6) == src_row["rmse"]
            assert pytest.approx(pub_row["skill_rmse"], abs=1e-6) == src_row["skill_rmse"]


def test_layer2_identity() -> None:
    df_pub = pd.read_csv(PUB_DIR / "pub_table_2_dynamic_fidelity.csv")
    df_src = pd.read_csv(SOURCE_DIR / "dynamic_fidelity_by_model_horizon.csv")

    assert len(df_pub) == len(df_src)
    for model in ["lightgbm", "sarima"]:
        for h in [1, 6, 24, 48]:
            pub_row = df_pub[(df_pub["model"] == model) & (df_pub["horizon"] == h)].iloc[0]
            src_row = df_src[(df_src["model"] == model) & (df_src["horizon"] == h)].iloc[0]

            assert pytest.approx(pub_row["variance_retention"], abs=1e-6) == src_row["variance_retention"]
            assert pytest.approx(pub_row["temporal_variability"], abs=1e-6) == src_row["temporal_variability"]


def test_layer4_sarima_classifications() -> None:
    df_pub = pd.read_csv(PUB_DIR / "pub_table_4_ghost_skill_structure.csv")

    sarima_24 = df_pub[(df_pub["model"] == "sarima") & (df_pub["horizon"] == 24)].iloc[0]
    assert sarima_24["ghost_skill_status"] == "STRONG_GHOST_SKILL_CANDIDATE_WITH_FOLD_HETEROGENEITY"
    assert sarima_24["stability_pattern"] == "GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS"

    sarima_48 = df_pub[(df_pub["model"] == "sarima") & (df_pub["horizon"] == 48)].iloc[0]
    assert sarima_48["ghost_skill_status"] == "GHOST_SKILL_DIAGNOSTIC_SATISFIED_IN_RECOVERED_SINGLE_SERIES"
    assert sarima_48["stability_pattern"] == "GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS"
