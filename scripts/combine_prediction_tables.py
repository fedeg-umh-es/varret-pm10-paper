#!/usr/bin/env python3
"""Combine full-origin base predictions with sparse-origin SARIMA predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine base and SARIMA prediction/skill tables.")
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--base-skill", type=Path, required=True)
    parser.add_argument("--sarima-predictions", type=Path, required=True)
    parser.add_argument("--sarima-skill", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--skill-output", type=Path, required=True)
    args = parser.parse_args()

    base_predictions = _read(args.base_predictions)
    base_skill = _read(args.base_skill)
    sarima_predictions = _read(args.sarima_predictions)
    sarima_skill = _read(args.sarima_skill)

    sarima_only = sarima_predictions[sarima_predictions["model"].eq("sarima")].copy()
    predictions = pd.concat([base_predictions, sarima_only], ignore_index=True)
    predictions = predictions.sort_values(["dataset", "model", "horizon", "origin_date"]).reset_index(drop=True)

    skill = pd.concat([base_skill, sarima_skill], ignore_index=True)
    skill = skill.drop_duplicates(["dataset", "model", "horizon"], keep="last")
    skill = skill.sort_values(["dataset", "model", "horizon"]).reset_index(drop=True)

    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.skill_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions_output, index=False)
    skill.to_csv(args.skill_output, index=False)
    print(f"Wrote {args.predictions_output} with {len(predictions)} rows")
    print(f"Wrote {args.skill_output} with {len(skill)} rows")


if __name__ == "__main__":
    main()
