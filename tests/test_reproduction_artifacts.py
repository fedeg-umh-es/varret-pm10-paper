from pathlib import Path

import pandas as pd

from scripts.run_paper_a_empirical import aggregate_metrics, expanding_folds


ROOT = Path(__file__).resolve().parents[1]


def test_rolling_origin_artifact_is_aligned_and_train_thresholded() -> None:
    predictions = pd.read_parquet(
        ROOT / "outputs/reproduction/predictions_rolling_origin.parquet"
    )
    data = pd.read_csv(ROOT / "data/processed/casa_de_campo_pm10_2023.csv")
    lead = (
        pd.to_datetime(predictions["target_time"])
        - pd.to_datetime(predictions["origin_time"])
    ).dt.total_seconds() / 3600
    assert lead.equals(predictions["horizon"].astype(float))
    assert not predictions.duplicated(["model", "fold", "origin_time", "horizon"]).any()

    observed = data["pm10"]
    for fold, train_end, _ in expanding_folds(len(data)):
        expected = observed.iloc[:train_end].dropna().quantile(0.75)
        actual = predictions.loc[predictions.fold.eq(fold), "p75_train"].unique()
        assert len(actual) == 1
        assert actual[0] == expected


def test_committed_metrics_recompute_from_row_level_predictions() -> None:
    predictions = pd.read_parquet(
        ROOT / "outputs/reproduction/predictions_rolling_origin.parquet"
    )
    expected = pd.read_csv(ROOT / "outputs/reproduction/metrics_rolling_origin.csv")
    actual, _ = aggregate_metrics(predictions)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12)
