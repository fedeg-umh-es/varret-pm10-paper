"""Simple validation checks shared across P33 modules."""


def ensure_positive(value: float, name: str) -> None:
    """Raise an error if a numeric value is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
