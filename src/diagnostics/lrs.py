"""Leakage Risk Score draft for temporal-audit purposes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeakageRiskComponents:
    """Auditable components contributing to the leakage risk score."""

    preprocessing_risk: float
    feature_risk: float
    validation_risk: float
    target_construction_risk: float

    def total(self) -> float:
        """Return the average component risk on a 0-1 scale."""
        components = (
            self.preprocessing_risk,
            self.feature_risk,
            self.validation_risk,
            self.target_construction_risk,
        )
        for value in components:
            if not 0.0 <= value <= 1.0:
                raise ValueError("All risk components must be between 0 and 1.")
        return sum(components) / len(components)


def leakage_risk_score(components: LeakageRiskComponents) -> float:
    """Compute the overall Leakage Risk Score on a 0-1 scale."""
    return components.total()


def leakage_risk_label(score: float) -> str:
    """Map the score to a coarse audit label."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be between 0 and 1.")
    if score < 0.2:
        return "low"
    if score < 0.5:
        return "moderate"
    if score < 0.8:
        return "high"
    return "critical"
