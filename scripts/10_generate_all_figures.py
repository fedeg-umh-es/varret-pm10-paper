#!/usr/bin/env python3
"""Run available figure generators for the expanded diagnostic workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> None:
    print("$", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    commands = [
        [sys.executable, "scripts/generate_figures_3_to_6.py"],
        [sys.executable, "scripts/plot_threshold_sensitivity.py"],
        [sys.executable, "scripts/plot_exceedance_figure.py"],
        [sys.executable, "scripts/plot_murphy_decomposition.py"],
    ]
    for command in commands:
        try:
            _run(command)
        except FileNotFoundError as exc:
            print(f"[SKIP] {command}: {exc}")
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] figure command failed ({exc.returncode}): {' '.join(command)}")
    print("Figure generation pass complete.")


if __name__ == "__main__":
    main()
