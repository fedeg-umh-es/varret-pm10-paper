import pandas as pd
import pytest

from src.evaluation.p4_exceedance.contract import (
    validate_contract,
    classify_evidence_status,
    ProducerEvidence,
)


def _valid_df():
    return pd.DataFrame(
        {
            "station": ["elche"],
            "model": ["hgb"],
            "horizon": [1],
            "y_true": [10.0],
            "y_pred": [12.0],
            "origin_date": ["2020-01-01"],
            "target_date": ["2020-01-02"],
        }
    )


class TestContractValidation:
    def test_valid_row_level_table_passes(self):
        report = validate_contract(_valid_df())
        assert report.is_valid
        assert report.missing_required == []
        assert report.missing_alternative_groups == []

    def test_missing_required_column_detected(self):
        df = _valid_df().drop(columns=["y_pred"])
        report = validate_contract(df)
        assert not report.is_valid
        assert "y_pred" in report.missing_required

    def test_missing_timestamp_alternative_group_detected(self):
        df = _valid_df().drop(columns=["origin_date"])
        report = validate_contract(df)
        assert not report.is_valid
        assert any("origin_date" in g for g in report.missing_alternative_groups)

    def test_optional_fields_not_invented(self):
        report = validate_contract(_valid_df())
        assert "baseline" in report.missing_optional
        assert "producer_repository" in report.missing_optional
        assert "fold_id" not in report.present_optional


class TestEvidenceClassification:
    def test_synthetic_data_source_is_demo_synthetic(self):
        assert classify_evidence_status("synthetic") == "DEMO_SYNTHETIC"

    def test_real_data_without_producer_evidence_stays_unverified(self):
        assert classify_evidence_status("real") == "REAL_DATA_UNVERIFIED"

    def test_real_data_with_incomplete_producer_evidence_stays_unverified(self):
        evidence = ProducerEvidence(
            rolling_origin_protocol="documented",
            producer_repository="fedeg-umh-es/varret-pm10-paper",
        )
        assert classify_evidence_status("real", evidence) == "REAL_DATA_UNVERIFIED"
        assert "producer_commit" in evidence.missing_fields()

    def test_real_data_with_complete_producer_evidence_is_audited(self):
        evidence = ProducerEvidence(
            rolling_origin_protocol="documented in docs/protocol.md",
            preprocessing_train_only="verified train-only per fold",
            baseline_explicit="persistence",
            producer_repository="fedeg-umh-es/varret-pm10-paper",
            producer_commit="abc1234",
            dataset="e1_rr_daily",
            station="casa_de_campo",
            period="2020-01-01/2020-12-31",
            fold="rolling_origin_fold_0",
        )
        assert evidence.is_complete()
        assert classify_evidence_status("real", evidence) == "REAL_DATA_AUDITED"

    def test_invalid_data_source_rejected(self):
        with pytest.raises(ValueError):
            classify_evidence_status("fabricated")
