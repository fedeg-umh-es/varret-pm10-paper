"""Directed regression tests for temporal pairing, DM/HAC/HLN, and bootstrap."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.variance import (
    _bootstrap_alpha_ci,
    _circular_block_indices,
    build_variance_retention_summary,
)
from src.evaluation.pairing import pair_model_and_baseline


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dm_significance", ROOT / "scripts" / "05_dm_significance.py")
assert SPEC and SPEC.loader
DM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DM)


def _prediction_rows(model: str, origins: list[str], horizon: int = 1) -> pd.DataFrame:
    dates = pd.to_datetime(origins) + pd.to_timedelta(horizon, unit="D")
    truth = np.arange(10.0, 10.0 + len(origins))
    prediction = truth if model == "candidate" else truth + 1.0
    return pd.DataFrame(
        {
            "dataset": "station_a",
            "model": model,
            "fold": origins,
            "origin_date": origins,
            "horizon": horizon,
            "date": dates.strftime("%Y-%m-%d"),
            "y_true": truth,
            "y_pred": prediction,
        }
    )


def test_pairing_uses_only_common_finite_pairs() -> None:
    model = _prediction_rows("candidate", ["2024-01-01", "2024-01-02", "2024-01-03"])
    baseline = _prediction_rows("persistence", ["2024-01-01", "2024-01-03"])
    baseline["y_true"] = [10.0, 12.0]
    baseline.loc[baseline.index[-1], "y_pred"] = np.nan

    paired = pair_model_and_baseline(model, baseline, context="fixture")

    assert paired["origin_date"].tolist() == ["2024-01-01"]
    assert paired.attrs["candidate_pairs"] == 2
    assert paired.attrs["invalid_pairs_dropped"] == 1


def test_pairing_rejects_duplicate_keys() -> None:
    model = _prediction_rows("candidate", ["2024-01-01", "2024-01-02"])
    baseline = pd.concat(
        [_prediction_rows("persistence", ["2024-01-01", "2024-01-02"])] * 2,
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate baseline pairing keys"):
        pair_model_and_baseline(model, baseline, context="fixture")


def test_pairing_rejects_accidentally_reordered_origins() -> None:
    model = _prediction_rows("candidate", ["2024-01-02", "2024-01-01"])
    baseline = _prediction_rows("persistence", ["2024-01-01", "2024-01-02"])
    with pytest.raises(ValueError, match="not in chronological order"):
        pair_model_and_baseline(model, baseline, context="fixture")


def test_dm_separates_horizons_and_records_overlap_lag() -> None:
    frames = []
    for horizon in (1, 3):
        origins = [f"2024-01-{day:02d}" for day in range(1, 11)]
        frames.extend(
            [
                _prediction_rows("candidate", origins, horizon),
                _prediction_rows("persistence", origins, horizon),
            ]
        )
    table = DM.build_dm_table(pd.concat(frames, ignore_index=True))
    assert table["horizon"].tolist() == [1, 3]
    assert table["hac_lag"].tolist() == [0, 2]
    assert table["n_pairs"].tolist() == [10, 10]


def test_dm_sign_and_known_mean_favor_model() -> None:
    d = np.array([0.5, 1.0, 1.5, 0.8, 1.2, 0.7, 1.1, 0.9])
    result = DM._dm_hln_result(d, horizon=1, hac_lag=0)
    assert result["dm_status"] == "ok"
    assert result["mean_loss_diff"] == pytest.approx(np.mean(d))
    assert result["dm_stat"] > 0
    assert 0 <= result["dm_pval_raw"] <= 1


def test_dm_handles_autocorrelation_and_explicit_lag() -> None:
    innovations = np.array([1.0, -0.4, 0.7, -0.2, 0.9, -0.3, 0.5, -0.1, 0.8, -0.2])
    d = np.empty_like(innovations)
    d[0] = innovations[0]
    for idx in range(1, len(d)):
        d[idx] = 0.7 * d[idx - 1] + innovations[idx]
    result = DM._dm_hln_result(d, horizon=4, hac_lag=3)
    assert result["dm_status"] == "ok"
    assert result["hac_lag"] == 3
    assert result["long_run_variance"] > 0


def test_dm_hln_can_be_enabled_or_disabled() -> None:
    d = np.array([0.2, 0.8, 0.4, 1.1, 0.6, 1.3, 0.7, 1.0])
    corrected = DM._dm_hln_result(d, horizon=2, hac_lag=1, apply_hln=True)
    uncorrected = DM._dm_hln_result(d, horizon=2, hac_lag=1, apply_hln=False)
    assert corrected["hln_applied"] is True
    assert uncorrected["hln_applied"] is False
    assert abs(corrected["dm_stat"]) < abs(uncorrected["dm_stat"])


@pytest.mark.parametrize(
    ("d", "horizon", "status"),
    [
        (np.array([1.0, 1.0, 1.0, 1.0]), 1, "all_loss_differentials_equal"),
        (np.array([0.1, 0.2]), 2, "insufficient_sample"),
        (np.array([0.1, np.nan, 0.2]), 1, "nonfinite_loss_differential"),
    ],
)
def test_dm_degenerate_cases_have_explicit_status(
    d: np.ndarray,
    horizon: int,
    status: str,
) -> None:
    assert DM._dm_hln_result(d, horizon=horizon)["dm_status"] == status


def test_sparse_origins_remove_false_multistep_overlap() -> None:
    origins = pd.Series(pd.date_range("2024-01-01", periods=8, freq="14D"))
    lag, stride = DM._overlap_hac_lag(origins, horizon=7)
    assert lag == 0
    assert stride == 14.0


class _StubRng:
    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        assert (low, high, size) == (0, 6, 2)
        return np.array([2, 0])


def test_circular_bootstrap_preserves_order_inside_blocks() -> None:
    indices = _circular_block_indices(6, 3, _StubRng())
    assert indices.tolist() == [2, 3, 4, 0, 1, 2]


def test_block_bootstrap_is_reproducible_and_preserves_true_prediction_pairs() -> None:
    group = _prediction_rows(
        "candidate",
        [f"2024-01-{day:02d}" for day in range(1, 21)],
        horizon=3,
    )
    first = _bootstrap_alpha_ci(group, n_boot=100, seed=123, block_length=4)
    second = _bootstrap_alpha_ci(group, n_boot=100, seed=123, block_length=4)
    assert first == second
    assert first[0] <= first[1]


def test_bootstrap_summary_does_not_mix_stations_or_horizons() -> None:
    frames = []
    skills = []
    for dataset, horizon in (("station_a", 1), ("station_b", 2)):
        frame = _prediction_rows("candidate", [f"2024-01-{day:02d}" for day in range(1, 11)], horizon)
        frame["dataset"] = dataset
        frames.append(frame)
        skills.append({"dataset": dataset, "model": "candidate", "horizon": horizon, "skill": 0.1})
    summary = build_variance_retention_summary(pd.concat(frames), pd.DataFrame(skills))
    assert set(zip(summary["dataset"], summary["horizon"])) == {("station_a", 1), ("station_b", 2)}
