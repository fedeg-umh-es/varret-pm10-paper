"""Row-level input contract (Fase 9).

Defines the canonical row-level schema this diagnostic module expects,
validates a given table against it without inventing missing fields, and
classifies the evidentiary status of a run.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd

REQUIRED_COLUMNS = (
    "station",
    "model",
    "horizon",
    "y_true",
    "y_pred",
)

# At least one column from each alternative group must be present.
REQUIRED_ALTERNATIVES = (
    ("origin_date", "origin_time"),
    ("target_date", "target_time", "date"),
)

OPTIONAL_COLUMNS = (
    "fold_id",
    "baseline",
    "producer_repository",
    "producer_commit",
)

EVIDENCE_STATUSES = (
    "DEMO_SYNTHETIC",
    "REAL_DATA_UNVERIFIED",
    "REAL_DATA_AUDITED",
)


@dataclass
class ContractValidationReport:
    is_valid: bool
    missing_required: list
    missing_alternative_groups: list
    present_optional: list
    missing_optional: list

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "missing_required": list(self.missing_required),
            "missing_alternative_groups": [list(g) for g in self.missing_alternative_groups],
            "present_optional": list(self.present_optional),
            "missing_optional": list(self.missing_optional),
        }


def validate_contract(df: pd.DataFrame) -> ContractValidationReport:
    columns = set(df.columns)

    missing_required = [c for c in REQUIRED_COLUMNS if c not in columns]
    missing_groups = [
        group for group in REQUIRED_ALTERNATIVES
        if not any(c in columns for c in group)
    ]
    present_optional = [c for c in OPTIONAL_COLUMNS if c in columns]
    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in columns]

    is_valid = not missing_required and not missing_groups

    return ContractValidationReport(
        is_valid=is_valid,
        missing_required=missing_required,
        missing_alternative_groups=missing_groups,
        present_optional=present_optional,
        missing_optional=missing_optional,
    )


@dataclass
class ProducerEvidence:
    """Evidence about the pipeline that produced the predictions.

    Every field is None (unknown) or "PENDING_VERIFICATION" until a human
    or an auditing process supplies it. None of these fields are ever
    inferred or fabricated by this module.
    """

    rolling_origin_protocol: str | None = None
    preprocessing_train_only: str | None = None
    baseline_explicit: str | None = None
    producer_repository: str | None = None
    producer_commit: str | None = None
    dataset: str | None = None
    station: str | None = None
    period: str | None = None
    fold: str | None = None

    def is_complete(self) -> bool:
        unresolved = {None, "PENDING_VERIFICATION", ""}
        return all(
            getattr(self, f.name) not in unresolved for f in fields(self)
        )

    def missing_fields(self) -> list:
        unresolved = {None, "PENDING_VERIFICATION", ""}
        return [f.name for f in fields(self) if getattr(self, f.name) in unresolved]


def classify_evidence_status(
    data_source: str,
    producer_evidence: ProducerEvidence | None = None,
) -> str:
    """Classify a run as DEMO_SYNTHETIC / REAL_DATA_UNVERIFIED / REAL_DATA_AUDITED.

    `data_source` must be declared explicitly by the caller ("synthetic" or
    "real"); this module cannot infer from the numbers alone whether data
    is synthetic. REAL_DATA_AUDITED additionally requires a complete
    ProducerEvidence record — otherwise real data stays REAL_DATA_UNVERIFIED
    regardless of how many tests pass.
    """
    if data_source not in ("synthetic", "real"):
        raise ValueError(f"Unsupported data_source '{data_source}'; expected 'synthetic' or 'real'.")

    if data_source == "synthetic":
        return "DEMO_SYNTHETIC"

    if producer_evidence is not None and producer_evidence.is_complete():
        return "REAL_DATA_AUDITED"

    return "REAL_DATA_UNVERIFIED"
