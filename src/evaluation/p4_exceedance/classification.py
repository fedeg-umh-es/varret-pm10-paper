"""Rank-reversal classification (Fase 6).

Turns a ranking_comparison result (continuous metric vs. a single event
metric) plus, optionally, a check for disagreement *among* event
sub-metrics (e.g. POD vs FAR) into one of six explicit labels. A
POD/FAR-style trade-off is never silently folded into YES.
"""

from __future__ import annotations

CLASSIFICATION_LABELS = (
    "YES",
    "NO",
    "TRADE_OFF_ONLY",
    "NOT_TESTED",
    "NOT_EVALUABLE",
    "UNCLEAR",
)


def detect_event_submetric_conflict(
    submetric_values: dict,
    higher_is_better: dict,
) -> bool:
    """True if event sub-metrics disagree on which model is best.

    submetric_values: {submetric_name: {model: value}}
    higher_is_better: {submetric_name: bool}, e.g. {"pod": True, "far": False}
    """
    if len(submetric_values) < 2:
        return False

    best_per_submetric = {}
    for name, values in submetric_values.items():
        if not values:
            continue
        direction = 1 if higher_is_better.get(name, True) else -1
        best_model = max(values, key=lambda m: direction * values[m])
        best_per_submetric[name] = best_model

    return len(set(best_per_submetric.values())) > 1


def classify_reversal(
    *,
    both_families_tested: bool,
    evaluation_status: str | None = None,
    n_reversals: float | None = None,
    n_pairs: float | None = None,
    event_submetrics_conflict: bool | None = None,
) -> str:
    """Classify a single (station, horizon) ranking-comparison outcome.

    Parameters
    ----------
    both_families_tested:
        False when the event-metric family (or the continuous-metric
        family) was never computed for this comparison at all — as
        opposed to being computed but found NOT_EVALUABLE.
    evaluation_status:
        The `evaluation_status` value from ranking_comparison's output
        row, e.g. "EVALUATED" or one of the NOT_EVALUABLE_* labels.
    n_reversals, n_pairs:
        From the same row.
    event_submetrics_conflict:
        Optional result of detect_event_submetric_conflict, when the
        caller has multiple event sub-metrics (e.g. POD and FAR) to
        check for an internal trade-off. When True, this takes
        precedence over a continuous-vs-event reversal verdict.
    """
    if not both_families_tested:
        return "NOT_TESTED"

    if evaluation_status is None or evaluation_status.startswith("NOT_EVALUABLE"):
        return "NOT_EVALUABLE"

    if evaluation_status != "EVALUATED":
        return "UNCLEAR"

    if event_submetrics_conflict:
        return "TRADE_OFF_ONLY"

    if n_reversals is None or n_pairs is None or n_pairs == 0:
        return "UNCLEAR"

    if n_reversals > 0:
        return "YES"

    return "NO"


def classify_reversal_from_row(
    row: dict,
    *,
    both_families_tested: bool = True,
    event_submetrics_conflict: bool | None = None,
) -> str:
    """Convenience wrapper reading fields off a ranking_comparison output row."""
    return classify_reversal(
        both_families_tested=both_families_tested,
        evaluation_status=row.get("evaluation_status"),
        n_reversals=row.get("n_reversals"),
        n_pairs=row.get("n_pairs"),
        event_submetrics_conflict=event_submetrics_conflict,
    )
