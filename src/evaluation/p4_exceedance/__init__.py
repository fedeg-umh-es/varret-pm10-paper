"""
Project: P4 - Ghost Skill & Dynamic Fidelity
Role: auxiliary exceedance and rank-reversal diagnostic
Evidence status: REAL_DATA_UNVERIFIED until the producer pipeline is audited

This package is a diagnostic add-on. It measures exceedances, hit rate,
false alarm rate, precision, recall, CSI, event bias, exceedance intensity
error, and rank changes between continuous and event metrics. It is not a
standalone proof of ghost skill; ghost skill is established at the P4
skill/alpha/skill_vp level (see docs/e1_rr_post_evaluation_contract.md and
docs/p4_exceedance_module.md).

Origin note: this package was written directly from a detailed functional
specification supplied by the repository owner on 2026-08-05. The original
external module (evaluacion_excedencias_pm10.py), its documentation, and
its prior technical audit live outside this repository (on the owner's
local machine) and were not accessible from the environment this code was
written in. This is therefore a fresh implementation against that
specification, not a line-for-line port. See
docs/p4_exceedance_module.md for the full traceability note.
"""

from .schema_adapter import adapt_schema, SchemaAdapterError
from .contract import validate_contract, classify_evidence_status, ProducerEvidence
from .integrity_checks import detect_duplicates, check_common_support
from .ranking_comparison import ranking_comparison, RANKING_COMPARISON_COLUMNS
from .classification import classify_reversal
from .threshold_sweep import diagnostic_sweep, calibrated_threshold_result, regulatory_threshold_result
from .bootstrap import block_bootstrap
from .manifest import build_manifest

__all__ = [
    "adapt_schema",
    "SchemaAdapterError",
    "validate_contract",
    "classify_evidence_status",
    "ProducerEvidence",
    "detect_duplicates",
    "check_common_support",
    "ranking_comparison",
    "RANKING_COMPARISON_COLUMNS",
    "classify_reversal",
    "diagnostic_sweep",
    "calibrated_threshold_result",
    "regulatory_threshold_result",
    "block_bootstrap",
    "build_manifest",
]
