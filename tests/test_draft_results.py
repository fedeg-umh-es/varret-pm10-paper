"""Unit tests for verifying that numerical values in draft_results.md match publication source tables."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DRAFT_RESULTS_FILE = ROOT / "draft_results.md"
PUB_DIR = ROOT / "outputs" / "publication_tables"


def test_draft_results_exists() -> None:
    assert DRAFT_RESULTS_FILE.exists()


def test_cited_numerical_values_in_draft() -> None:
    text = DRAFT_RESULTS_FILE.read_text(encoding="utf-8")

    # Load publication source tables
    df_l1 = pd.read_csv(PUB_DIR / "pub_table_1_error_metrics.csv")
    df_l2 = pd.read_csv(PUB_DIR / "pub_table_2_dynamic_fidelity.csv")
    df_l3 = pd.read_csv(PUB_DIR / "pub_table_3_event_metrics.csv")
    df_l4 = pd.read_csv(PUB_DIR / "pub_table_4_ghost_skill_structure.csv")

    # 1. SARIMA 48h skill RMSE: +0.1325
    sarima_48_l1 = df_l1[(df_l1["model"] == "sarima") & (df_l1["horizon"] == 48)].iloc[0]
    assert "+0.1325" in text
    assert pytest.approx(sarima_48_l1["skill_rmse"], abs=1e-4) == 0.1325

    # 2. SARIMA 48h pooled variance retention: 0.0037 (0.37%)
    sarima_48_l2 = df_l2[(df_l2["model"] == "sarima") & (df_l2["horizon"] == 48)].iloc[0]
    assert "0.0037" in text or "0.37%" in text
    assert pytest.approx(sarima_48_l2["variance_retention"], abs=1e-4) == 0.0037

    # 3. SARIMA 48h temporal variability: 0.0223
    assert "0.0223" in text
    assert pytest.approx(sarima_48_l2["temporal_variability"], abs=1e-4) == 0.0223

    # 4. SARIMA 48h POD and CSI: 0.0000
    sarima_48_l3 = df_l3[(df_l3["model"] == "sarima") & (df_l3["horizon"] == 48)].iloc[0]
    assert "0.0000" in text
    assert pytest.approx(sarima_48_l3["pod"], abs=1e-4) == 0.0
    assert pytest.approx(sarima_48_l3["csi"], abs=1e-4) == 0.0

    # 5. SARIMA 24h skill RMSE: +0.0450
    sarima_24_l1 = df_l1[(df_l1["model"] == "sarima") & (df_l1["horizon"] == 24)].iloc[0]
    assert "+0.0450" in text
    assert pytest.approx(sarima_24_l1["skill_rmse"], abs=1e-4) == 0.0450

    # 6. Check that forbidden terms do not appear
    forbidden_terms = [
        "across stations",
        "across PM10 networks",
        "17 stations",
        "universal ghost skill",
        "Grade A",
        "Skill_VP",
    ]
    for term in forbidden_terms:
        assert term not in text, f"Forbidden term '{term}' found in draft_results.md"
