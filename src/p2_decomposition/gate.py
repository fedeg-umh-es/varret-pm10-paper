"""Mechanical evaluation of the ``P2_PAPER_GO`` gate.

Contract: ``P2_PAIRED_DECOMPOSITION_CONTRACT.md`` section 15, canon section 15.

Nine conditions can be settled from mechanical evidence.
``NON_TRIVIAL_INTERPRETATION_FOUND`` is a scientific judgement and is pinned to
``PENDING_SCIENTIFIC_REVIEW``; :func:`evaluate_gate` refuses to emit any other
status for it, so no run can declare ``P2_PAPER_GO`` on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "GATE_CONDITIONS",
    "GATE_STATUSES",
    "HUMAN_JUDGEMENT_CONDITIONS",
    "GateCondition",
    "GateEvaluationError",
    "evaluate_gate",
    "overall_status",
]

GATE_CONDITIONS: tuple[str, ...] = (
    "PAIRED_SUPPORT_VALID",
    "TRAIN_ONLY_VALID",
    "NO_ORACLE_SELECTION",
    "MSE_IDENTITY_VERIFIED",
    "MISSINGNESS_TESTS_PASS",
    "P_SENSITIVITY_COMPLETED",
    "BLOCK_BOOTSTRAP_COMPLETED",
    "SYNTHETIC_VALIDATION_COMPLETED",
    "RESULT_REPLICATES_ACROSS_MORE_THAN_ONE_STATION",
    "NON_TRIVIAL_INTERPRETATION_FOUND",
)

GATE_STATUSES: tuple[str, ...] = ("PASS", "FAIL", "BLOCKED", "PENDING_SCIENTIFIC_REVIEW")

#: Conditions that may never be settled mechanically.
HUMAN_JUDGEMENT_CONDITIONS: frozenset[str] = frozenset({"NON_TRIVIAL_INTERPRETATION_FOUND"})


class GateEvaluationError(RuntimeError):
    """Raised on an attempt to settle a human-judgement condition mechanically."""


@dataclass(frozen=True)
class GateCondition:
    """One gate condition with its status, evidence and notes."""

    condition: str
    status: str
    evidence_paths: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.condition not in GATE_CONDITIONS:
            raise ValueError(f"unknown gate condition {self.condition!r}")
        if self.status not in GATE_STATUSES:
            raise ValueError(f"unknown gate status {self.status!r}")
        if (
            self.condition in HUMAN_JUDGEMENT_CONDITIONS
            and self.status != "PENDING_SCIENTIFIC_REVIEW"
        ):
            raise GateEvaluationError(
                f"{self.condition} is a scientific judgement and must remain "
                "PENDING_SCIENTIFIC_REVIEW until a human decides it after "
                "inspecting the results."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "evidence_paths": list(self.evidence_paths),
            "notes": self.notes,
        }


def evaluate_gate(conditions: list[GateCondition]) -> dict[str, object]:
    """Assemble ``paper_go_gate.json`` from the supplied conditions.

    Every condition in :data:`GATE_CONDITIONS` must be present exactly once; a
    silently omitted condition would read as an unnoticed pass.
    """
    supplied = [condition.condition for condition in conditions]
    duplicates = {name for name in supplied if supplied.count(name) > 1}
    if duplicates:
        raise ValueError(f"duplicate gate conditions: {sorted(duplicates)}")
    missing = set(GATE_CONDITIONS) - set(supplied)
    if missing:
        raise ValueError(f"gate is incomplete, missing: {sorted(missing)}")

    by_name = {condition.condition: condition for condition in conditions}
    payload: dict[str, object] = {
        name: by_name[name].as_dict() for name in GATE_CONDITIONS
    }
    counts = {status: 0 for status in GATE_STATUSES}
    for condition in conditions:
        counts[condition.status] += 1
    payload["_summary"] = {
        "counts": counts,
        "p2_paper_status": overall_status(conditions),
        "rule": (
            "NON_TRIVIAL_INTERPRETATION_FOUND is a human scientific judgement. "
            "P2_PAPER_GO cannot be declared by this pipeline under any "
            "combination of mechanical results."
        ),
    }
    return payload


def overall_status(conditions: list[GateCondition]) -> str:
    """Overall manuscript status. Never returns ``P2_PAPER_GO``."""
    statuses = {condition.condition: condition.status for condition in conditions}
    if any(status == "FAIL" for status in statuses.values()):
        return "NO-GO — MECHANICAL CONDITION FAILED"
    if any(status == "BLOCKED" for status in statuses.values()):
        return "NO-GO — MECHANICAL CONDITION BLOCKED"
    return "NO-GO PENDING SCIENTIFIC REVIEW"
