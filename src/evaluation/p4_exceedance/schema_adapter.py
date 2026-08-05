"""Schema adapter for row-level exceedance inputs (Fase 3 / B1).

Real prediction tables in this repository (outputs/metrics/predictions*.csv)
use ``date`` for the forecast target and ``origin_date`` for the issue time,
not the ``target_date`` name the diagnostic module expects. This module
performs an explicit, auditable adaptation instead of a silent rename: it
records which source column was used, converts both timestamps, and
validates horizon/temporal-resolution coherence before anything downstream
is allowed to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class SchemaAdapterError(ValueError):
    """Raised when the input cannot be adapted without guessing."""


@dataclass
class SchemaAdaptationReport:
    target_date_source: str
    origin_date_source: str
    resolution: str
    n_rows: int
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target_date_source": self.target_date_source,
            "origin_date_source": self.origin_date_source,
            "resolution": self.resolution,
            "n_rows": self.n_rows,
            "notes": list(self.notes),
        }


_DAY = pd.Timedelta(days=1)
_HOUR = pd.Timedelta(hours=1)


def _resolve_origin_column(df: pd.DataFrame) -> str:
    if "origin_date" in df.columns:
        return "origin_date"
    if "origin_time" in df.columns:
        return "origin_time"
    raise SchemaAdapterError(
        "No origin timestamp column found. Expected one of: "
        "'origin_date', 'origin_time'."
    )


def _resolve_target_column(df: pd.DataFrame) -> tuple[str, list]:
    notes: list = []
    if "target_date" in df.columns:
        return "target_date", notes
    if "target_time" in df.columns:
        notes.append(
            "No 'target_date' column found; 'target_time' was used as the "
            "target timestamp source."
        )
        return "target_time", notes
    if "date" in df.columns:
        notes.append(
            "No 'target_date' column found; 'date' was interpreted as the "
            "forecast target time (target_date = date)."
        )
        return "date", notes
    raise SchemaAdapterError(
        "No target timestamp column found. Expected one of: "
        "'target_date', 'target_time', 'date'."
    )


def _infer_resolution(delta: pd.Series, horizon: pd.Series) -> tuple[str, pd.Series]:
    """Return (resolution, mismatch_mask) for the best-fitting hypothesis.

    Both hypotheses are tested against every row; if neither holds for all
    rows, or both do (degenerate horizon=0 case, excluded upstream), the
    resolution is ambiguous and must not be guessed.
    """
    expected_daily = horizon.astype("int64") * _DAY
    expected_hourly = horizon.astype("int64") * _HOUR

    daily_mismatch = delta != expected_daily
    hourly_mismatch = delta != expected_hourly

    daily_ok = not daily_mismatch.any()
    hourly_ok = not hourly_mismatch.any()

    if daily_ok and not hourly_ok:
        return "daily", daily_mismatch
    if hourly_ok and not daily_ok:
        return "hourly", hourly_mismatch
    if daily_ok and hourly_ok:
        # Every horizon value would have to be 0 for both to hold; horizon>0
        # is enforced earlier, so this path should be unreachable.
        raise SchemaAdapterError(
            "Temporal resolution is ambiguous: both a daily and an hourly "
            "interpretation of horizon are consistent with the data. "
            "Declare the resolution explicitly via the 'resolution' argument."
        )
    raise SchemaAdapterError(
        "Temporal resolution could not be determined unambiguously: "
        "target_date is not coherent with origin_date + horizon days, nor "
        "with origin_date + horizon hours, for all rows. "
        f"{int(daily_mismatch.sum())} row(s) violate the daily hypothesis and "
        f"{int(hourly_mismatch.sum())} row(s) violate the hourly hypothesis. "
        "Declare the resolution explicitly via the 'resolution' argument "
        "or fix the source data; the adapter will not infer silently."
    )


def adapt_schema(
    df: pd.DataFrame,
    *,
    resolution: str | None = None,
) -> tuple[pd.DataFrame, SchemaAdaptationReport]:
    """Adapt a row-level prediction table to the canonical target_date schema.

    Parameters
    ----------
    df:
        Input table. Not mutated; the source CSV is never touched.
    resolution:
        Optional explicit declaration, one of {"daily", "hourly"}. When
        omitted, resolution is inferred and validated against every row;
        if it cannot be determined unambiguously, this raises
        SchemaAdapterError instead of guessing.

    Returns
    -------
    (adapted_df, report)
        adapted_df carries 'origin_date' and 'target_date' as tz-naive
        datetime64 columns (added, originals preserved) plus 'horizon' as
        an integer column. report documents what was inferred.
    """
    if resolution is not None and resolution not in ("daily", "hourly"):
        raise SchemaAdapterError(
            f"Unsupported resolution '{resolution}'; expected 'daily' or 'hourly'."
        )
    if "horizon" not in df.columns:
        raise SchemaAdapterError("Missing required column 'horizon'.")

    out = df.copy()

    origin_col = _resolve_origin_column(out)
    target_col, notes = _resolve_target_column(out)

    origin_dt = pd.to_datetime(out[origin_col], errors="raise")
    target_dt = pd.to_datetime(out[target_col], errors="raise")

    horizon = pd.to_numeric(out["horizon"], errors="raise")
    if not np.isfinite(horizon).all():
        raise SchemaAdapterError("Column 'horizon' contains non-finite values.")
    if (horizon <= 0).any():
        n_bad = int((horizon <= 0).sum())
        raise SchemaAdapterError(
            f"Column 'horizon' must be strictly positive; {n_bad} row(s) "
            "violate this."
        )
    horizon_int = horizon.astype("int64")
    if not np.array_equal(horizon_int.values, horizon.values):
        raise SchemaAdapterError("Column 'horizon' must contain integer values.")

    if not (target_dt > origin_dt).all():
        n_bad = int((target_dt <= origin_dt).sum())
        raise SchemaAdapterError(
            f"target_date must be strictly after origin_date; {n_bad} row(s) "
            "violate this (target_date <= origin_date)."
        )

    delta = target_dt - origin_dt

    if resolution is not None:
        expected = horizon_int * (_DAY if resolution == "daily" else _HOUR)
        mismatch = delta != expected
        if mismatch.any():
            raise SchemaAdapterError(
                f"Declared resolution='{resolution}' is not coherent with "
                f"target_date - origin_date for {int(mismatch.sum())} row(s)."
            )
        resolved = resolution
    else:
        resolved, _ = _infer_resolution(delta, horizon_int)

    out["origin_date"] = origin_dt
    out["target_date"] = target_dt
    out["horizon"] = horizon_int

    report = SchemaAdaptationReport(
        target_date_source=target_col,
        origin_date_source=origin_col,
        resolution=resolved,
        n_rows=len(out),
        notes=notes,
    )
    return out, report
