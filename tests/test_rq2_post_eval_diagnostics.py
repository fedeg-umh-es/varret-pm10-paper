"""Unit and integrity tests for RQ2 post-evaluation diagnostic experiment."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"

EXPECTED_STATIONS = 17
EXPECTED_MODELS = 5
EXPECTED_HORIZONS = 7
EXPECTED_TOTAL_CELLS = 595
EXPECTED_STATION_HORIZONS = 119

TARGET_MODELS = {"hgb_direct", "ridge_direct", "sarima", "seasonal_naive", "stl_ridge_direct"}


def test_canonical_cell_table_structure() -> None:
    """Verifies skill_fidelity_event_cells.csv has exactly 595 rows with no duplicates."""
    path = TABLES_DIR / "skill_fidelity_event_cells.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path)
    assert len(df) == EXPECTED_TOTAL_CELLS, f"Expected {EXPECTED_TOTAL_CELLS} rows, got {len(df)}"
    assert df["station"].nunique() == EXPECTED_STATIONS
    assert set(df["model"].unique()) == TARGET_MODELS
    assert set(df["horizon"].unique()) == set(range(1, 8))

    # Check uniqueness of keys
    dups = df.duplicated(subset=["station", "model", "horizon"])
    assert not dups.any(), f"Found {dups.sum()} duplicated (station, model, horizon) keys"


def test_skill_and_alpha_fidelity_to_master() -> None:
    """Ensures skill and alpha match source master_diagnostic_table.csv exactly."""
    cell_path = TABLES_DIR / "skill_fidelity_event_cells.csv"
    master_path = TABLES_DIR / "master_diagnostic_table.csv"

    cells = pd.read_csv(cell_path)
    master = pd.read_csv(master_path)
    master = master[master["model"].isin(TARGET_MODELS)].copy()

    merged = cells.merge(
        master,
        left_on=["station", "model", "horizon"],
        right_on=["station_id", "model", "horizon"],
        suffixes=("_cell", "_master"),
    )
    assert len(merged) == EXPECTED_TOTAL_CELLS

    # Check exact equality (within floating point precision)
    np.testing.assert_allclose(
        merged["persistence_relative_rmse_skill"].to_numpy(),
        merged["skill"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
        err_msg="Skill values mismatch between cells table and master table",
    )
    np.testing.assert_allclose(
        merged["alpha_cell"].to_numpy(),
        merged["alpha_master"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
        err_msg="Alpha values mismatch between cells table and master table",
    )


def test_event_metrics_integrity_and_na_preservation() -> None:
    """Verifies event metrics: FAR direction, CSI validity, and NA preservation."""
    cell_path = TABLES_DIR / "skill_fidelity_event_cells.csv"
    cells = pd.read_csv(cell_path)

    # 1. Event support is positive for all cells under p75
    assert (cells["observed_exceedance_count"] > 0).all(), "Observed count should be > 0 for all p75 cells"

    # 2. CSI is bounded in [0, 1]
    valid_csi = cells["csi"].dropna()
    assert (valid_csi >= 0.0).all() and (valid_csi <= 1.0).all(), "CSI out of bounds [0, 1]"

    # 3. Recall is bounded in [0, 1]
    valid_rec = cells["recall"].dropna()
    assert (valid_rec >= 0.0).all() and (valid_rec <= 1.0).all(), "Recall out of bounds [0, 1]"

    # 4. FAR is NaN when predicted_exceedance_count == 0, and not coerced to 0
    zero_pred_flags = cells[cells["predicted_exceedance_count"] == 0]
    assert len(zero_pred_flags) > 0, "Expected some cells with 0 predicted flags"
    assert zero_pred_flags["far"].isna().all(), "FAR must remain NaN when predicted flags are 0"

    # 5. Diagnostic categories partition the dataset completely
    category_counts = cells["diagnostic_category"].value_counts()
    assert category_counts.sum() == EXPECTED_TOTAL_CELLS

    # Positive skill collapsed count check
    expected_pos_collapsed = ((cells["persistence_relative_rmse_skill"] > 0) & (cells["alpha"] < 0.5)).sum()
    assert category_counts.get("positive_skill_collapsed", 0) == expected_pos_collapsed


def test_rank_reversal_by_station_horizon() -> None:
    """Checks rank reversal table across 119 station-horizons."""
    path = TABLES_DIR / "rank_reversal_by_station_horizon.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path)
    assert len(df) == EXPECTED_STATION_HORIZONS, f"Expected {EXPECTED_STATION_HORIZONS} evaluations, got {len(df)}"
    assert df["station"].nunique() == EXPECTED_STATIONS
    assert set(df["horizon"].unique()) == set(range(1, 8))

    # All best models must belong to target models
    assert df["best_model_skill"].isin(TARGET_MODELS).all()
    assert df["best_model_csi"].isin(TARGET_MODELS).all()

    # Reversal rate is high and non-zero
    reversal_rate = df["winner_changed"].mean()
    assert 0.80 <= reversal_rate <= 0.95, f"Expected reversal rate around 89%, got {reversal_rate:.1%}"


def test_rank_reversal_pairwise() -> None:
    """Checks pairwise rank reversal table for all 10 model pairs across 119 evaluations."""
    path = TABLES_DIR / "rank_reversal_pairwise.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path)
    expected_pairs_count = EXPECTED_STATION_HORIZONS * 10  # 119 * 10 = 1190
    assert len(df) == expected_pairs_count, f"Expected {expected_pairs_count} rows, got {len(df)}"

    # Reversal logic check: (skill_diff > 0 & csi_diff < 0) or (skill_diff < 0 & csi_diff > 0)
    for _, row in df.iterrows():
        s_diff = row["skill_diff_m1_minus_m2"]
        c_diff = row["csi_diff_m1_minus_m2"]
        expected_reversal = (s_diff > 0 and c_diff < 0) or (s_diff < 0 and c_diff > 0)
        assert row["is_reversal_csi"] == expected_reversal, f"Reversal flag logic error in row: {row}"


def test_falsification_table_integrity() -> None:
    """Verifies that seasonal_naive and stl_ridge_direct serve as valid falsification cases."""
    path = TABLES_DIR / "variance_retention_falsification.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path).set_index("model")
    assert len(df) == EXPECTED_MODELS

    # seasonal_naive preserves variance (alpha ~ 1.0) but has non-positive skill
    assert df.loc["seasonal_naive", "median_alpha"] >= 0.9
    assert df.loc["seasonal_naive", "median_skill"] <= 0.0
    assert df.loc["seasonal_naive", "collapse_rate_pct"] == 0.0

    # stl_ridge_direct preserves/inflates variance (alpha > 1.2) but has catastrophic negative skill
    assert df.loc["stl_ridge_direct", "median_alpha"] > 1.2
    assert df.loc["stl_ridge_direct", "median_skill"] < -1.0
    assert df.loc["stl_ridge_direct", "collapse_rate_pct"] == 0.0
    assert df.loc["stl_ridge_direct", "median_far"] > 0.70


def test_generated_figures_exist() -> None:
    """Ensures both candidate figures exist in PDF and PNG format with non-zero size."""
    expected_figures = [
        "skill_fidelity_event_map.pdf",
        "skill_fidelity_event_map.png",
        "rank_reversal_by_horizon.pdf",
        "rank_reversal_by_horizon.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure {fig_name} is too small / empty"


def test_audit_reports_exist_and_contain_required_sections() -> None:
    """Ensures audit reports exist and contain all required scientific sections."""
    gate_path = AUDIT_DIR / "rq2_event_utility_novelty_gate.md"
    assert gate_path.exists(), f"Missing {gate_path}"
    content = gate_path.read_text(encoding="utf-8")

    required_sections = [
        "1. Research Question Tested (RQ2)",
        "2. Empirical Data Support",
        "3. Main Result",
        "4. Rank Reversal",
        "5. Incremental Diagnostic Information",
        "6. Falsification Analysis",
        "7. Formal Gate Decision",
        "8. Claims Allowed",
        "9. Claims Forbidden",
        "GO",
    ]
    for sec in required_sections:
        assert sec in content, f"Missing section in gate report: {sec}"
