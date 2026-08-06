"""Paired-support and oracle-prohibition tests.

Covers ``test_unique_prediction_keys``, ``test_identical_paired_targets``,
``test_global_common_support_across_p``, ``test_model_curves_are_individual``
and ``test_no_oracle_selection``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.p2_decomposition.pairing import (
    OracleSelectionError,
    assert_no_duplicate_keys,
    assert_paired_consistency,
    build_common_support,
    load_model_predictions,
    select_evaluated_models,
)

METHODS = ("persistence", "ar1", "ar7", "ar14", "ar21", "ridge_direct", "hgb_direct")


def make_long_frame(
    n_origins: int = 40,
    methods: tuple[str, ...] = METHODS,
    horizons: tuple[int, ...] = (1, 2),
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2020-01-01", periods=n_origins, freq="D")
    records = []
    for origin in origins:
        for horizon in horizons:
            y_true = float(rng.normal(30.0, 5.0))
            for method in methods:
                y_pred = y_true + float(rng.normal(0.0, 3.0))
                records.append(
                    {
                        "station": "T",
                        "model": method,
                        "fold_or_window_id": origin.strftime("%Y-%m-%d"),
                        "origin_date": origin,
                        "target_date": origin + pd.Timedelta(days=horizon),
                        "horizon": horizon,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "forecast_available_at_origin": True,
                        "squared_error": (y_true - y_pred) ** 2,
                    }
                )
    return pd.DataFrame(records)


def test_unique_prediction_keys() -> None:
    frame = make_long_frame()
    assert_no_duplicate_keys(frame, extra=("model",))

    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="not one-to-one"):
        assert_no_duplicate_keys(duplicated, extra=("model",))


def test_identical_paired_targets() -> None:
    frame = make_long_frame()
    assert_paired_consistency(frame, methods=METHODS, y_true_tolerance=1e-9)

    poisoned = frame.copy()
    poisoned.loc[poisoned.index[0], "y_true"] += 1.0
    with pytest.raises(ValueError, match="y_true differs across methods"):
        assert_paired_consistency(poisoned, methods=METHODS, y_true_tolerance=1e-9)


def test_paired_consistency_requires_every_method_on_every_case() -> None:
    frame = make_long_frame()
    dropped = frame[~((frame["model"] == "ar21") & (frame["horizon"] == 1))]
    with pytest.raises(ValueError, match="paired support broken"):
        assert_paired_consistency(dropped, methods=METHODS, y_true_tolerance=1e-9)


def test_global_common_support_across_p() -> None:
    """The principal support must not move when p moves.

    AR(21) needs 21 consecutive observed days, so it is unavailable at more
    origins than AR(7). The global intersection takes the strictest requirement
    once, and every lag order is then evaluated on that same fixed sample.
    """
    frame = make_long_frame(n_origins=40)
    unavailable_ar21 = (frame["model"] == "ar21") & (
        frame["origin_date"] < pd.Timestamp("2020-01-11")
    )
    unavailable_ar14 = (frame["model"] == "ar14") & (
        frame["origin_date"] < pd.Timestamp("2020-01-06")
    )
    frame.loc[unavailable_ar21 | unavailable_ar14, "forecast_available_at_origin"] = False
    frame.loc[unavailable_ar21 | unavailable_ar14, "squared_error"] = np.nan

    support = build_common_support(
        frame,
        station="T",
        support_type="GLOBAL_COMMON",
        methods=METHODS,
        y_true_tolerance=1e-9,
    )
    # 40 origins minus the 10 where AR(21) is unavailable, times two horizons.
    assert support.n_cases == 60
    assert set(support.keys["origin_date"].min() == pd.Timestamp("2020-01-11") for _ in [0]) == {True}

    restricted = frame.merge(
        support.keys[list(support.keys.columns[:5])],
        on=list(support.keys.columns[:5]),
        how="inner",
    )
    # Every lag order sees exactly the same origins on this support.
    per_method = restricted.groupby("model")["origin_date"].apply(
        lambda s: tuple(sorted(s.unique()))
    )
    assert per_method.nunique() == 1


def test_order_specific_support_is_larger_and_separately_labelled() -> None:
    frame = make_long_frame(n_origins=40)
    mask = (frame["model"] == "ar21") & (frame["origin_date"] < pd.Timestamp("2020-01-11"))
    frame.loc[mask, "forecast_available_at_origin"] = False
    frame.loc[mask, "squared_error"] = np.nan

    global_support = build_common_support(
        frame, station="T", support_type="GLOBAL_COMMON", methods=METHODS,
        y_true_tolerance=1e-9,
    )
    order_specific = build_common_support(
        frame,
        station="T",
        support_type="ORDER_SPECIFIC_p7",
        methods=("persistence", "ar1", "ar7", "ridge_direct", "hgb_direct"),
        y_true_tolerance=1e-9,
    )
    assert order_specific.n_cases > global_support.n_cases
    assert order_specific.support_type != global_support.support_type


def test_model_curves_are_individual() -> None:
    """Each declared model keeps its own losses; nothing is averaged across models."""
    frame = make_long_frame()
    support = build_common_support(
        frame, station="T", support_type="GLOBAL_COMMON", methods=METHODS,
        y_true_tolerance=1e-9,
    )
    paired = frame.merge(support.keys[list(support.keys.columns[:5])],
                         on=list(support.keys.columns[:5]), how="inner")
    losses = paired.groupby(["horizon", "model"])["squared_error"].mean().unstack()

    assert "ridge_direct" in losses.columns
    assert "hgb_direct" in losses.columns
    assert not any(name.startswith("best") for name in losses.columns)
    assert losses.loc[1, "ridge_direct"] != pytest.approx(losses.loc[1, "hgb_direct"])


def test_no_oracle_selection() -> None:
    declared = ["ridge_direct", "hgb_direct", "sarima"]
    available = {"ridge_direct", "hgb_direct", "persistence"}

    selected = select_evaluated_models(declared, available_models=available)
    assert selected == ["ridge_direct", "hgb_direct"]

    with pytest.raises(OracleSelectionError, match="oracle envelope"):
        select_evaluated_models(
            declared, available_models=available, allow_test_loss_selection=True
        )

    losses = pd.DataFrame({"model": ["ridge_direct"], "mse": [1.0]})
    with pytest.raises(OracleSelectionError, match="Evaluated losses"):
        select_evaluated_models(declared, available_models=available, losses=losses)


def test_load_model_predictions_validates_schema(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"dataset": ["d"], "model": ["m"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_model_predictions(
            path,
            station="T",
            dataset_id="d",
            horizons=[1],
            source_sha256="x",
            producer_repository="r",
            producer_commit="c",
        )


def test_load_model_predictions_rejects_wrong_target_offset(tmp_path) -> None:
    path = tmp_path / "offset.csv"
    pd.DataFrame(
        {
            "dataset": ["d"],
            "model": ["persistence"],
            "fold": ["2020-01-01"],
            "origin_date": ["2020-01-01"],
            "horizon": [1],
            "date": ["2020-01-05"],  # not origin + 1
            "y_true": [10.0],
            "y_pred": [9.0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="target_date != origin_date \\+ horizon"):
        load_model_predictions(
            path,
            station="T",
            dataset_id="d",
            horizons=[1],
            source_sha256="x",
            producer_repository="r",
            producer_commit="c",
        )


def test_real_prediction_artefact_has_unique_keys() -> None:
    """Regression guard on the actual repository input."""
    frame = load_model_predictions(
        "outputs/metrics/predictions_zarra_emep.csv",
        station="Zarra EMEP",
        dataset_id="e1_rr_zarra_emep",
        horizons=[1, 2, 3, 4, 5, 6, 7],
        source_sha256="unchecked",
        producer_repository="fedeg-umh-es/varret-pm10-paper",
        producer_commit="unchecked",
    )
    assert_no_duplicate_keys(frame, extra=("model",))
    assert set(frame["horizon"].unique()) == {1, 2, 3, 4, 5, 6, 7}
