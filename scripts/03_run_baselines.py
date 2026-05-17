"""Run persistence and seasonal_persistence baselines under rolling-origin evaluation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.schema import PREDICTIONS_COLUMNS
from src.models.persistence import PersistenceModel
from src.models.seasonal_persistence import SeasonalPersistenceModel
from src.splits.rolling_origin import generate_rolling_origin_folds


def _run_model(df: pd.DataFrame, folds, dataset_name: str, model_name: str, predict_fn) -> pd.DataFrame:
    rows = []
    for fold in folds:
        history = df["y"].iloc[fold.train_indices].values
        y_true = float(df["y"].iloc[fold.test_index])
        y_pred = float(predict_fn(history, fold.horizon))
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold.fold,
                "origin_date": str(fold.origin_date.date()),
                "horizon": fold.horizon,
                "date": str(fold.target_date.date()),
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
    return pd.DataFrame(rows, columns=PREDICTIONS_COLUMNS)


def main() -> None:
    cfg_ds = yaml.safe_load(Path("configs/datasets/pm10.yaml").read_text())["dataset"]
    cfg_ro = yaml.safe_load(Path("configs/evaluation/rolling_origin.yaml").read_text())["rolling_origin"]

    processed_path = Path(cfg_ds["processed_path"])
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset missing: {processed_path}. Run scripts/02 first.")

    df = pd.read_parquet(processed_path)
    folds = generate_rolling_origin_folds(
        df,
        date_column="date",
        min_train_size=cfg_ro["min_train_size"],
        max_horizon=cfg_ro["max_horizon"],
        step_size=cfg_ro.get("step_size", 1),
    )
    print(f"Generated {len(folds)} rolling-origin (origin, horizon) pairs")

    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = cfg_ds["name"]

    persist = PersistenceModel()
    pred_p = _run_model(
        df, folds, dataset_name, "persistence",
        lambda history, horizon: persist.predict(history, horizon)[horizon - 1],
    )
    pred_p.to_parquet(out_dir / "persistence.parquet", index=False)
    print(f"persistence: {len(pred_p)} predictions written → {out_dir / 'persistence.parquet'}")

    season = SeasonalPersistenceModel(season_length=7)
    pred_s = _run_model(
        df, folds, dataset_name, "seasonal_persistence",
        lambda history, horizon: season.predict(history, horizon)[horizon - 1],
    )
    pred_s.to_parquet(out_dir / "seasonal_persistence.parquet", index=False)
    print(f"seasonal_persistence: {len(pred_s)} predictions written")


if __name__ == "__main__":
    main()
