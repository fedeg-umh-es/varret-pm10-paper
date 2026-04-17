"""Text summary helpers for P33 runs."""

from pathlib import Path


def write_summary(path: str | Path, text: str) -> None:
    """Write a plain-text summary file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
