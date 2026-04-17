"""Baseline-relative skill calculations for P33."""

from __future__ import annotations


def relative_skill(model_error: float, baseline_error: float) -> float:
    """Compute skill relative to a baseline error scale."""
    if baseline_error <= 0:
        raise ValueError("baseline_error must be positive.")
    return 1.0 - (model_error / baseline_error)
