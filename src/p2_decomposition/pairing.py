"""Paired support construction and validation.

Contract: ``P2_PAIRED_DECOMPOSITION_CONTRACT.md`` sections 1, 7 and 9.

Every compared loss must live on the same case. This module builds the paired
support, proves one-to-one uniqueness, and refuses any construction that would
change the verification sample between lag orders without disclosure. It also
holds the oracle-selection guard: the evaluated model list is a declared fixed
list, and any attempt to derive it from the evaluated test loss raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "PAIRED_KEY",
    "OracleSelectionError",
    "PairedSupport",
    "assert_no_duplicate_keys",
    "assert_paired_consistency",
    "build_common_support",
    "load_model_predictions",
    "select_evaluated_models",
]

#: Minimal atomic paired key. ``model`` completes the row identity.
PAIRED_KEY: tuple[str, ...] = (
    "station",
    "fold_or_window_id",
    "origin_date",
    "target_date",
    "horizon",
)

_REQUIRED_SOURCE_COLUMNS = {
    "dataset",
    "model",
    "fold",
    "origin_date",
    "horizon",
    "date",
    "y_true",
    "y_pred",
}


class OracleSelectionError(RuntimeError):
    """Raised when a model would be chosen using the evaluated test loss."""


@dataclass(frozen=True)
class PairedSupport:
    """A validated set of paired cases plus the methods it is valid for."""

    station: str
    support_type: str
    methods: tuple[str, ...]
    keys: pd.DataFrame  # PAIRED_KEY columns plus y_true

    @property
    def n_cases(self) -> int:
        return int(len(self.keys))

    def counts_by_horizon(self) -> pd.Series:
        return self.keys.groupby("horizon").size()


def load_model_predictions(
    path: str | Path,
    *,
    station: str,
    dataset_id: str,
    horizons: tuple[int, ...] | list[int],
    source_sha256: str,
    producer_repository: str,
    producer_commit: str,
) -> pd.DataFrame:
    """Read a row-level prediction artefact into the canonical atomic schema.

    The schema validation is deliberately strict: an unexpected column layout
    means the artefact is not what the provenance manifest claims, and silently
    coercing it would break the paired contract.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    missing = _REQUIRED_SOURCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    observed_datasets = set(frame["dataset"].astype(str).unique())
    if observed_datasets != {dataset_id}:
        raise ValueError(
            f"{path}: expected a single dataset {dataset_id!r}, found {sorted(observed_datasets)}"
        )

    out = pd.DataFrame(
        {
            "station": station,
            "dataset_id": frame["dataset"].astype(str),
            "model": frame["model"].astype(str),
            "fold_or_window_id": frame["fold"].astype(str),
            "origin_date": pd.to_datetime(frame["origin_date"]),
            "target_date": pd.to_datetime(frame["date"]),
            "horizon": frame["horizon"].astype(int),
            "y_true": pd.to_numeric(frame["y_true"], errors="coerce").astype(float),
            "y_pred": pd.to_numeric(frame["y_pred"], errors="coerce").astype(float),
        }
    )
    out = out[out["horizon"].isin(list(horizons))].copy()

    offset = (out["target_date"] - out["origin_date"]).dt.days
    bad = offset != out["horizon"]
    if bool(bad.any()):
        raise ValueError(
            f"{path}: {int(bad.sum())} row(s) where target_date != origin_date + horizon"
        )

    out["forecast_available_at_origin"] = out["y_pred"].notna()
    out["squared_error"] = np.where(
        out["y_true"].notna() & out["y_pred"].notna(),
        (out["y_true"] - out["y_pred"]) ** 2,
        np.nan,
    )
    out["source_artifact"] = str(path)
    out["source_sha256"] = source_sha256
    out["producer_repository"] = producer_repository
    out["producer_commit"] = producer_commit

    assert_no_duplicate_keys(out, extra=("model",))
    return out


def assert_no_duplicate_keys(frame: pd.DataFrame, extra: tuple[str, ...] = ()) -> None:
    """Raise unless ``PAIRED_KEY + extra`` identifies rows one-to-one."""
    key = list(PAIRED_KEY) + list(extra)
    missing = [column for column in key if column not in frame.columns]
    if missing:
        raise KeyError(f"paired key columns missing from frame: {missing}")
    duplicated = frame.duplicated(subset=key, keep=False)
    if bool(duplicated.any()):
        sample = frame.loc[duplicated, key].head(5).to_dict("records")
        raise ValueError(
            f"paired key is not one-to-one: {int(duplicated.sum())} duplicated row(s); "
            f"examples {sample}"
        )


def assert_paired_consistency(
    frame: pd.DataFrame,
    *,
    methods: tuple[str, ...] | list[str],
    y_true_tolerance: float,
) -> None:
    """Validate the paired contract on a long frame restricted to one support.

    Checks, in order: every method present on every case; identical
    ``target_date``; identical ``y_true`` within tolerance; identical
    availability; no missing squared error inside the support.
    """
    methods = list(methods)
    key = list(PAIRED_KEY)

    per_case = frame.groupby(key, observed=True)["model"].nunique()
    if not bool((per_case == len(methods)).all()):
        offending = int((per_case != len(methods)).sum())
        raise ValueError(
            f"paired support broken: {offending} case(s) do not carry all "
            f"{len(methods)} methods {methods}"
        )

    spread = frame.groupby(key, observed=True)["y_true"].agg(["min", "max"])
    delta = (spread["max"] - spread["min"]).abs()
    if bool((delta > y_true_tolerance).any()):
        worst = float(delta.max())
        raise ValueError(
            f"y_true differs across methods on the same case (max spread {worst:.3g} "
            f"> tolerance {y_true_tolerance:.3g})"
        )

    if not bool(frame["forecast_available_at_origin"].all()):
        raise ValueError(
            f"{int((~frame['forecast_available_at_origin']).sum())} row(s) inside the "
            "support are not available at origin"
        )

    if bool(frame["squared_error"].isna().any()):
        raise ValueError(
            f"{int(frame['squared_error'].isna().sum())} missing squared error(s) inside "
            "the selected support"
        )


def build_common_support(
    frame: pd.DataFrame,
    *,
    station: str,
    support_type: str,
    methods: tuple[str, ...] | list[str],
    y_true_tolerance: float,
) -> PairedSupport:
    """Intersect cases across ``methods`` and return the validated support.

    The intersection is taken on the full paired key, so a case survives only
    when *every* method has an available forecast for the same origin, target
    and horizon.
    """
    methods = tuple(methods)
    key = list(PAIRED_KEY)

    available = frame[
        frame["model"].isin(methods)
        & frame["forecast_available_at_origin"]
        & frame["squared_error"].notna()
    ]
    counts = available.groupby(key, observed=True)["model"].nunique()
    complete_keys = counts[counts == len(methods)].index

    if len(complete_keys) == 0:
        keys = pd.DataFrame(columns=key + ["y_true"])
    else:
        keys = (
            pd.DataFrame(index=complete_keys)
            .reset_index()
            .merge(
                available.groupby(key, observed=True)["y_true"].first().reset_index(),
                on=key,
                how="left",
            )
            .sort_values(["horizon", "origin_date"])
            .reset_index(drop=True)
        )

    support = PairedSupport(
        station=station, support_type=support_type, methods=methods, keys=keys
    )
    if support.n_cases:
        restricted = available.merge(keys[key], on=key, how="inner")
        assert_paired_consistency(
            restricted, methods=methods, y_true_tolerance=y_true_tolerance
        )
    return support


def select_evaluated_models(
    declared_models: tuple[str, ...] | list[str],
    *,
    available_models: set[str],
    allow_test_loss_selection: bool = False,
    losses: pd.DataFrame | None = None,
) -> list[str]:
    """Return the declared model list, refusing any test-loss-driven selection.

    ``losses`` is accepted only so that a caller who *tries* to pass evaluated
    losses into model selection gets a loud, traceable failure instead of a
    quietly optimistic result.
    """
    if allow_test_loss_selection:
        raise OracleSelectionError(
            "Selecting the evaluated model by test MSE is prohibited: "
            "M(station, horizon) = argmin test MSE is an oracle envelope "
            "(P2_PROJECT_CANON.md section 9)."
        )
    if losses is not None:
        raise OracleSelectionError(
            "Evaluated losses were passed to model selection. The primary model set "
            "must be a declared fixed list independent of the evaluated test loss."
        )
    return [model for model in declared_models if model in available_models]
