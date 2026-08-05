from src.evaluation.p4_exceedance.classification import (
    classify_reversal,
    classify_reversal_from_row,
    detect_event_submetric_conflict,
)


class TestScenario12ClassificationYes:
    def test_reversal_present_classified_yes(self):
        label = classify_reversal(
            both_families_tested=True,
            evaluation_status="EVALUATED",
            n_reversals=1,
            n_pairs=3,
        )
        assert label == "YES"


class TestScenario13ClassificationNo:
    def test_no_reversal_classified_no(self):
        label = classify_reversal(
            both_families_tested=True,
            evaluation_status="EVALUATED",
            n_reversals=0,
            n_pairs=3,
        )
        assert label == "NO"


class TestScenario14ClassificationTradeOffOnly:
    def test_event_submetric_conflict_classified_trade_off_only_not_yes(self):
        label = classify_reversal(
            both_families_tested=True,
            evaluation_status="EVALUATED",
            n_reversals=1,
            n_pairs=3,
            event_submetrics_conflict=True,
        )
        assert label == "TRADE_OFF_ONLY"

    def test_detect_event_submetric_conflict_pod_far_disagreement(self):
        submetric_values = {
            "pod": {"model_a": 0.9, "model_b": 0.5},
            "far": {"model_a": 0.4, "model_b": 0.1},
        }
        higher_is_better = {"pod": True, "far": False}
        assert detect_event_submetric_conflict(submetric_values, higher_is_better) is True

    def test_detect_event_submetric_conflict_agreement_is_not_a_conflict(self):
        submetric_values = {
            "pod": {"model_a": 0.9, "model_b": 0.5},
            "far": {"model_a": 0.1, "model_b": 0.4},
        }
        higher_is_better = {"pod": True, "far": False}
        assert detect_event_submetric_conflict(submetric_values, higher_is_better) is False


class TestNotTestedNotEvaluableUnclear:
    def test_not_tested_when_event_family_never_computed(self):
        label = classify_reversal(both_families_tested=False)
        assert label == "NOT_TESTED"

    def test_not_evaluable_propagates_from_ranking_status(self):
        label = classify_reversal(
            both_families_tested=True,
            evaluation_status="NOT_EVALUABLE_SINGLE_MODEL",
        )
        assert label == "NOT_EVALUABLE"

    def test_unclear_when_pairs_missing_despite_evaluated_status(self):
        label = classify_reversal(
            both_families_tested=True,
            evaluation_status="EVALUATED",
            n_reversals=None,
            n_pairs=None,
        )
        assert label == "UNCLEAR"

    def test_classify_reversal_from_row_reads_ranking_comparison_row(self):
        row = {"evaluation_status": "EVALUATED", "n_reversals": 0, "n_pairs": 1}
        assert classify_reversal_from_row(row) == "NO"
