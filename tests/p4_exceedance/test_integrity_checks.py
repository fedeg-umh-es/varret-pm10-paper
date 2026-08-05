import pandas as pd
import pytest

from src.evaluation.p4_exceedance.integrity_checks import (
    detect_duplicates,
    check_common_support,
    COMMON_SUPPORT_SENSITIVITY,
)


def _base_rows():
    return [
        {"station": "elche", "model": "a", "origin_date": "2020-01-01", "target_date": "2020-01-02", "horizon": 1, "fold_id": 0},
        {"station": "elche", "model": "a", "origin_date": "2020-01-02", "target_date": "2020-01-03", "horizon": 1, "fold_id": 0},
        {"station": "elche", "model": "b", "origin_date": "2020-01-01", "target_date": "2020-01-02", "horizon": 1, "fold_id": 0},
        {"station": "elche", "model": "b", "origin_date": "2020-01-02", "target_date": "2020-01-03", "horizon": 1, "fold_id": 0},
    ]


class TestScenario8Duplicates:
    def test_no_duplicates_reports_clean(self):
        df = pd.DataFrame(_base_rows())
        report = detect_duplicates(df)
        assert not report.has_duplicates
        assert report.n_duplicates == 0
        assert len(report.duplicate_keys) == 0

    def test_exact_duplicate_key_detected_and_not_dropped(self):
        rows = _base_rows()
        rows.append(dict(rows[0]))  # exact duplicate key for model a
        df = pd.DataFrame(rows)
        report = detect_duplicates(df)
        assert report.has_duplicates
        assert report.n_duplicates == 2  # both copies flagged
        assert "a" in report.affected_models
        assert "elche" in report.affected_stations
        # rows are not removed by the detector itself
        assert len(df) == len(rows)

    def test_missing_key_columns_raises(self):
        df = pd.DataFrame({"model": ["a"], "horizon": [1]})
        with pytest.raises(ValueError):
            detect_duplicates(df)


class TestScenario9MisalignedCases:
    def test_misaligned_models_reported_with_missing_cases(self):
        rows = _base_rows()
        # model b is missing the second case that model a has
        rows = [r for r in rows if not (r["model"] == "b" and r["origin_date"] == "2020-01-02")]
        df = pd.DataFrame(rows)
        report = check_common_support(df)
        assert not report.is_aligned
        assert ("elche", 1) in report.misaligned_groups
        missing_for_b = report.misalignment_table[report.misalignment_table["model"] == "b"]
        assert len(missing_for_b) == 1
        assert missing_for_b.iloc[0]["origin_date"] == "2020-01-02"

    def test_model_entirely_missing_from_one_fold_is_detected(self):
        # Regression test: fold_id must be part of the case key, not the
        # group key, otherwise a model with zero rows for a given fold
        # simply does not appear in that group and the gap goes unnoticed.
        rows = _base_rows()
        rows = [r for r in rows if not (r["model"] == "b" and r["fold_id"] == 0 and r["origin_date"] == "2020-01-02")]
        df = pd.DataFrame(rows)
        report = check_common_support(df)
        assert not report.is_aligned
        missing_for_b = report.misalignment_table[report.misalignment_table["model"] == "b"]
        assert len(missing_for_b) == 1

    def test_fold_id_as_group_column_is_rejected(self):
        df = pd.DataFrame(_base_rows())
        with pytest.raises(ValueError, match="fold_id"):
            check_common_support(df, group_columns=["station", "horizon", "fold_id"])

    def test_aligned_models_report_no_misalignment(self):
        df = pd.DataFrame(_base_rows())
        report = check_common_support(df)
        assert report.is_aligned
        assert report.misaligned_groups == []
        assert len(report.misalignment_table) == 0

    def test_common_support_sensitivity_mode_computes_intersection_only_when_requested(self):
        rows = _base_rows()
        rows = [r for r in rows if not (r["model"] == "b" and r["origin_date"] == "2020-01-02")]
        df = pd.DataFrame(rows)

        strict_report = check_common_support(df, mode="STRICT")
        assert strict_report.common_support_table is None

        sensitivity_report = check_common_support(df, mode=COMMON_SUPPORT_SENSITIVITY)
        assert sensitivity_report.common_support_table is not None
        assert len(sensitivity_report.common_support_table) == 1
        assert sensitivity_report.common_support_table.iloc[0]["origin_date"] == "2020-01-01"


class TestScenario10NotEvaluableWithoutCommonSupport:
    def test_single_model_group_is_skipped_not_flagged_misaligned(self):
        df = pd.DataFrame(
            [{"station": "s", "model": "a", "origin_date": "2020-01-01", "target_date": "2020-01-02", "horizon": 1}]
        )
        report = check_common_support(df, group_columns=["station", "horizon"])
        assert report.is_aligned

    def test_invalid_mode_raises(self):
        df = pd.DataFrame(_base_rows())
        with pytest.raises(ValueError):
            check_common_support(df, mode="SILENT_INTERSECTION")
