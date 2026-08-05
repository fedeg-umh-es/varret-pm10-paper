import numpy as np
import pandas as pd
import pytest

from src.evaluation.p4_exceedance.ranking_comparison import (
    ranking_comparison,
    RANKING_COMPARISON_COLUMNS,
)


def _metrics(rows):
    return pd.DataFrame(rows)


class TestScenario6EmptySchemaSingleModel:
    def test_single_model_returns_stable_schema_not_evaluable(self):
        df = _metrics(
            [
                {"station": "elche", "horizon": 1, "model": "hgb", "skill": 0.3, "csi": 0.5},
            ]
        )
        result = ranking_comparison(
            df, metric_continuous_col="skill", metric_event_col="csi"
        )
        assert list(result.columns) == list(RANKING_COMPARISON_COLUMNS)
        assert len(result) == 1
        assert result.loc[0, "evaluation_status"] == "NOT_EVALUABLE_SINGLE_MODEL"
        assert pd.isna(result.loc[0, "kendall_tau"])
        assert result.loc[0, "n_pairs"] == 0


class TestScenario7NoComparableModelsAtAll:
    def test_empty_input_returns_empty_but_well_formed_frame(self):
        df = _metrics([]).assign(station=[], horizon=[], model=[], skill=[], csi=[])
        result = ranking_comparison(
            df, metric_continuous_col="skill", metric_event_col="csi"
        )
        assert list(result.columns) == list(RANKING_COMPARISON_COLUMNS)
        assert len(result) == 0

    def test_does_not_raise_keyerror_on_sort_values_horizon(self):
        # This is the literal B2 regression: pd.DataFrame([]).sort_values("horizon")
        # used to raise KeyError. ranking_comparison must never do that.
        df = pd.DataFrame(columns=["station", "horizon", "model", "skill", "csi"])
        result = ranking_comparison(
            df, metric_continuous_col="skill", metric_event_col="csi"
        )
        assert "horizon" in result.columns
        assert len(result) == 0


class TestScenario11KendallTauBWithTies:
    def test_ties_use_kendall_tau_b(self):
        df = _metrics(
            [
                {"station": "s", "horizon": 1, "model": "a", "skill": 0.5, "csi": 0.4},
                {"station": "s", "horizon": 1, "model": "b", "skill": 0.5, "csi": 0.4},
                {"station": "s", "horizon": 1, "model": "c", "skill": 0.2, "csi": 0.1},
            ]
        )
        result = ranking_comparison(
            df, metric_continuous_col="skill", metric_event_col="csi"
        )
        row = result.iloc[0]
        assert row["evaluation_status"] == "EVALUATED"
        assert row["n_models"] == 3
        from scipy.stats import kendalltau

        expected_tau, expected_p = kendalltau(
            [0.5, 0.5, 0.2], [0.4, 0.4, 0.1], variant="b"
        )
        assert row["kendall_tau"] == pytest.approx(expected_tau)
        assert row["kendall_pvalue"] == pytest.approx(expected_p)


class TestNoRaiseOnZeroModelsPerGroup:
    def test_incomplete_ranking_when_a_model_is_missing_event_metric(self):
        df = _metrics(
            [
                {"station": "s", "horizon": 1, "model": "a", "skill": 0.5, "csi": 0.4},
                {"station": "s", "horizon": 1, "model": "b", "skill": 0.3, "csi": np.nan},
            ]
        )
        result = ranking_comparison(
            df, metric_continuous_col="skill", metric_event_col="csi"
        )
        assert result.iloc[0]["evaluation_status"] == "NOT_EVALUABLE_INCOMPLETE_RANKING"

    def test_misaligned_group_reported_without_computing_tau(self):
        df = _metrics(
            [
                {"station": "s", "horizon": 1, "model": "a", "skill": 0.5, "csi": 0.4},
                {"station": "s", "horizon": 1, "model": "b", "skill": 0.3, "csi": 0.2},
            ]
        )
        result = ranking_comparison(
            df,
            metric_continuous_col="skill",
            metric_event_col="csi",
            misaligned_groups={("s", 1)},
        )
        row = result.iloc[0]
        assert row["evaluation_status"] == "NOT_EVALUABLE_NO_COMMON_CASES"
        assert pd.isna(row["kendall_tau"])

    def test_missing_required_columns_raises_valueerror(self):
        df = pd.DataFrame({"horizon": [1], "model": ["a"], "skill": [0.1], "csi": [0.2]})
        with pytest.raises(ValueError, match="missing required column"):
            ranking_comparison(df, metric_continuous_col="skill", metric_event_col="csi")

    def test_sorted_by_horizon(self):
        df = _metrics(
            [
                {"station": "s", "horizon": 3, "model": "a", "skill": 0.5, "csi": 0.4},
                {"station": "s", "horizon": 3, "model": "b", "skill": 0.3, "csi": 0.2},
                {"station": "s", "horizon": 1, "model": "a", "skill": 0.5, "csi": 0.4},
                {"station": "s", "horizon": 1, "model": "b", "skill": 0.3, "csi": 0.2},
            ]
        )
        result = ranking_comparison(
            df, metric_continuous_col="skill", metric_event_col="csi"
        )
        assert list(result["horizon"]) == [1, 3]
