"""Run the minimal P33 paper pipeline in order."""

import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    "scripts/01_validate_raw_data.py",
    "scripts/02_build_processed_datasets.py",
    "scripts/03_run_baselines.py",
    "scripts/04_run_linear_models.py",
    "scripts/05_run_boosting_model.py",
    "scripts/06_build_skill_tables.py",
    "scripts/07_build_variance_retention_table.py",
    "scripts/08_build_run_summary.py",
]


def main() -> None:
    """Execute the P33 pipeline scripts sequentially."""
    for script in SCRIPT_ORDER:
        print(f"Running {script}")
        subprocess.run([sys.executable, str(Path(script))], check=True)


if __name__ == "__main__":
    main()
