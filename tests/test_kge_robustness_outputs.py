"""Schema checks for KGE robustness closeout outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("path", "columns"),
    [
        (
            ROOT / "results" / "tie_sensitivity_rankings.csv",
            {
                "rank_method",
                "median_spearman_skill_vs_kge",
                "median_spearman_phi_vs_r",
                "rank_reversal_rate",
                "top_skill_top_kge_mismatch_cells",
            },
        ),
        (
            ROOT / "results" / "horizon_breakdown_kge.csv",
            {
                "h",
                "median_skill_h",
                "median_phi_h",
                "median_r_h",
                "median_beta_h",
                "median_KGE_h",
                "rank_reversal_rate",
            },
        ),
        (
            ROOT / "results" / "station_type_breakdown_kge.csv",
            {
                "station_type",
                "n_stations",
                "n_station_horizon_cells",
                "median_spearman_skill_vs_kge",
                "median_spearman_phi_vs_r",
                "rank_reversal_rate",
            },
        ),
    ],
)
def test_kge_robustness_csv_schema_when_present(path: Path, columns: set[str]) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} has not been generated yet.")
    df = pd.read_csv(path)
    assert columns.issubset(df.columns)
    assert not df.empty

