import numpy as np
import pandas as pd

from scripts.run_paper_a_empirical import causal_inputs, expanding_folds, make_features
from src.data.preprocess_pm10 import impute_series, normalize_zscore


def test_causal_fill_never_uses_future_value() -> None:
    values = pd.Series([np.nan, 1.0, np.nan, 9.0])
    result = causal_inputs(values)
    assert pd.isna(result.iloc[0])
    assert result.iloc[2] == 1.0


def test_expanding_folds_are_temporally_ordered_and_disjoint() -> None:
    folds = expanding_folds(8760)
    assert folds[0] == (0, 4380, 5256)
    assert folds[-1] == (4, 7884, 8760)
    assert all(previous[2] == current[1] for previous, current in zip(folds, folds[1:]))


def test_features_at_origin_do_not_change_when_future_changes() -> None:
    index = pd.date_range("2023-01-01", periods=240, freq="h")
    original = pd.Series(np.arange(240, dtype=float), index=index)
    changed = original.copy()
    changed.iloc[201:] = -9999
    left = make_features(original, index, horizon=24).iloc[:201]
    right = make_features(changed, index, horizon=24).iloc[:201]
    pd.testing.assert_frame_equal(left, right)


def test_legacy_preprocessor_is_causal_and_train_fitted() -> None:
    series = pd.Series([1.0, np.nan, 3.0, 1000.0])
    filled = impute_series(series)
    assert filled.iloc[1] == 1.0
    normalized, params = normalize_zscore(filled, fit_series=filled.iloc[:3])
    assert params["mean"] < 10
    assert normalized.iloc[3] > 100
