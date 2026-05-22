"""Run the minimal P33 paper pipeline in order."""

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = [
    ["scripts/01_validate_raw_data.py"],
    ["scripts/01_generate_e1_rr_lags_only_predictions.py", "--n-jobs", "7"],
    ["scripts/07_build_variance_retention_table.py"],
    ["scripts/08_build_run_summary.py"],
    ["scripts/08_build_e1_rr_variance_retention_report.py"],
    ["scripts/09_build_e1_rr_latex_table.py"],
    ["scripts/10_build_figures.py"],
]


def main() -> None:
    """Execute the P33 pipeline scripts sequentially."""
    env = os.environ.copy()
    repo_root = str(Path.cwd())
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    for command in SCRIPT_ORDER:
        script = command[0]
        args = command[1:]
        print(f"Running {script}")
        subprocess.run([sys.executable, str(Path(script)), *args], check=True, env=env)


if __name__ == "__main__":
    main()
