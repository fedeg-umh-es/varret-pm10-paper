"""Auxiliary diagnostic wrapper for Skill_VP."""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.metrics import skill_vp as compute_skill_vp


@dataclass(frozen=True)
class SkillVPDiagnostic:
    """Container for auxiliary diagnostic interpretation."""

    skill: float
    alpha: float
    skill_vp: float
    interpretation: str


def summarize_skill_vp(skill: float, alpha: float) -> SkillVPDiagnostic:
    """Return Skill_VP and a concise interpretation string."""
    diagnostic = compute_skill_vp(skill=skill, alpha=alpha)
    if alpha < 0.5:
        label = "positive skill with severe variance collapse" if skill > 0 else "negative skill with variance collapse"
    elif alpha > 1.5:
        label = "variance inflation"
    else:
        label = "variance-retention compatible"
    return SkillVPDiagnostic(skill=skill, alpha=alpha, skill_vp=diagnostic, interpretation=label)
