"""Conceptual tests for baseline-relative skill."""

from src.evaluation.skill import relative_skill


def test_relative_skill_positive_when_model_beats_baseline() -> None:
    """Skill must be positive when the model error is smaller than the baseline error."""
    assert relative_skill(8.0, 10.0) > 0.0
