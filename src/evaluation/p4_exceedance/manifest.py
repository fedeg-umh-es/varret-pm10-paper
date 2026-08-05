"""Evaluation manifest builder (Fase 11).

Produces the run manifest that ties a diagnostic execution to its inputs,
schema adaptation, integrity checks, threshold policy, bootstrap
configuration, and evidence status. Unknown values are recorded as
``None`` (JSON null) or the literal string ``"PENDING_VERIFICATION"`` —
never guessed.
"""

from __future__ import annotations

PENDING = "PENDING_VERIFICATION"

PROJECT = "P4 — Ghost Skill & Dynamic Fidelity"
ANALYSIS_ROLE = "Auxiliary exceedance and rank-reversal diagnostic"

REQUIRED_MANIFEST_FIELDS = (
    "project",
    "analysis_role",
    "input_predictions_path",
    "input_predictions_sha256",
    "input_schema",
    "schema_adapter",
    "producer_repository",
    "producer_commit",
    "dataset",
    "station",
    "period",
    "resolution",
    "forecast_horizons",
    "rolling_origin_protocol",
    "preprocessing_protocol",
    "event_threshold",
    "threshold_source",
    "threshold_mode",
    "block_length",
    "block_length_unit",
    "block_length_justification",
    "random_seed",
    "models",
    "baselines",
    "metrics_continuous",
    "metrics_event",
    "common_case_status",
    "duplicate_status",
    "execution_timestamp",
    "execution_label",
)

_DEFAULTS = {field: None for field in REQUIRED_MANIFEST_FIELDS}
_DEFAULTS.update(
    {
        "project": PROJECT,
        "analysis_role": ANALYSIS_ROLE,
        "producer_repository": PENDING,
        "producer_commit": PENDING,
        "rolling_origin_protocol": PENDING,
        "preprocessing_protocol": PENDING,
        "block_length": 14,
        "block_length_unit": "days",
        "block_length_justification": "PROVISIONAL_DEFAULT_NOT_JUSTIFIED_BY_ACF_OR_EPISODE_DURATION",
        "execution_label": PENDING,
    }
)


def build_manifest(**fields) -> dict:
    """Build a manifest dict. project/analysis_role are always fixed.

    Raises ValueError on unknown field names (typo protection) rather than
    silently accepting arbitrary keys.
    """
    unknown = set(fields) - set(REQUIRED_MANIFEST_FIELDS)
    if unknown:
        raise ValueError(f"Unknown manifest field(s): {sorted(unknown)}")

    manifest = dict(_DEFAULTS)
    manifest.update(fields)
    manifest["project"] = PROJECT
    manifest["analysis_role"] = ANALYSIS_ROLE
    return manifest
