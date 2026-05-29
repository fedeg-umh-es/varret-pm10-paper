"""Tests for KGE horizon-wise diagnostics.

AUDIT SOURCE: adapted from manuscript_assets/paper_c_kge/paper.tex formulas
and existing variance-retention schema tests in tests/test_variance_retention_schema.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.kge_diagnostics import compute_kge_components, compute_kge_skill, kge_horizon_table


def test_kge_perfect_forecast() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = compute_kge_components(y, y)
    assert result["KGE_h"] == pytest.approx(1.0)
    assert result["r_h"] == pytest.approx(1.0)
    assert result["alpha_h"] == pytest.approx(1.0)
    assert result["beta_h"] == pytest.approx(1.0)


def test_alpha_equals_phi_formula() -> None:
    y_true = np.array([1.0, 2.0, 4.0, 8.0])
    y_pred = np.array([1.0, 3.0, 5.0, 7.0])
    result = compute_kge_components(y_true, y_pred)
    phi = np.std(y_pred, ddof=0) / np.std(y_true, ddof=0)
    assert result["alpha_h"] == pytest.approx(phi)


def test_kge_skill_defined_against_persistence() -> None:
    assert np.isfinite(compute_kge_skill(0.6, 0.3))


def _valid_forecast_df() -> pd.DataFrame:
    rows = []
    y_true = [10.0, 12.0, 14.0, 16.0, 18.0]
    for model, y_pred in {
        "persistence": [9.0, 11.0, 13.0, 15.0, 17.0],
        "ridge_direct": [10.1, 11.8, 14.2, 15.9, 18.1],
    }.items():
        for idx, (obs, pred) in enumerate(zip(y_true, y_pred)):
            rows.append(
                {
                    "dataset": "station_a",
                    "model": model,
                    "fold": idx,
                    "origin_date": f"2020-01-{idx + 1:02d}",
                    "horizon": 1,
                    "date": f"2020-01-{idx + 2:02d}",
                    "y_true": obs,
                    "y_pred": pred,
                }
            )
    return pd.DataFrame(rows)


def test_kge_table_expected_columns() -> None:
    table = kge_horizon_table(_valid_forecast_df())
    expected = {"station", "model", "h", "r_h", "alpha_h", "beta_h", "KGE_h", "KGE_skill_h"}
    assert expected.issubset(table.columns)


def test_no_nans_for_valid_input() -> None:
    table = kge_horizon_table(_valid_forecast_df())
    cols = ["r_h", "alpha_h", "beta_h", "KGE_h", "KGE_skill_h"]
    assert not table[cols].isna().any().any()


def test_generated_kge_csv_schema_when_present() -> None:
    path = Path(__file__).resolve().parents[1] / "results" / "kge_horizon_table.csv"
    if not path.exists():
        pytest.skip("KGE CSV has not been generated yet.")
    df = pd.read_csv(path)
    assert len(df) == 595
    assert {"station", "model", "h", "phi_h", "alpha_h", "KGE_h", "KGE_skill_h"}.issubset(df.columns)

