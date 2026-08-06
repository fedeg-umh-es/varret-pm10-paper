"""Diagnostic record collection for the P2 paired decomposition.

Contract: ``P2_PAIRED_DECOMPOSITION_CONTRACT.md`` section 6.

Every fitted ``Gamma_p`` contributes a record carrying its eigen-diagnostics,
rank, solver status, the (absent) regularisation and the minimum valid-pair
count backing it. Records are accumulated as plain dictionaries and converted
to frames at the end, so a long run never holds more than the diagnostics it
has actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .linear_references import ProjectionDiagnostics

__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "DiagnosticsCollector",
    "diagnostics_record",
]

DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "station",
    "fold_or_window_id",
    "origin_date",
    "p",
    "min_eigenvalue",
    "max_eigenvalue",
    "condition_number",
    "rank",
    "solver_status",
    "regularisation_type",
    "regularisation_value",
    "pair_count_min",
    "horizons_solved",
    "horizons_refused",
    "notes",
)


def diagnostics_record(
    diagnostics: ProjectionDiagnostics,
    *,
    station: str,
    fold_or_window_id: str,
    origin_date: object,
    horizons_solved: list[int],
    horizons_refused: list[int],
) -> dict[str, object]:
    """Build one diagnostics row.

    The eigenvalues, rank and condition number of ``Gamma_p`` do not depend on
    the horizon — ``Gamma_p`` is built from the autocovariances alone, and only
    the right-hand side ``c_h`` changes with ``h``. Emitting one row per
    horizon would therefore duplicate identical numbers, so the horizon
    dimension is recorded as the explicit lists of horizons that were solved and
    refused, and the per-horizon solver status is written separately to
    ``yule_walker_solver_status.parquet``.
    """
    return {
        "station": station,
        "fold_or_window_id": fold_or_window_id,
        "origin_date": origin_date,
        "p": diagnostics.p,
        "min_eigenvalue": diagnostics.min_eigenvalue,
        "max_eigenvalue": diagnostics.max_eigenvalue,
        "condition_number": diagnostics.condition_number,
        "rank": diagnostics.rank,
        "solver_status": diagnostics.solver_status,
        "regularisation_type": diagnostics.regularisation_type,
        "regularisation_value": diagnostics.regularisation_value,
        "pair_count_min": diagnostics.pair_count_min,
        "horizons_solved": ";".join(str(h) for h in sorted(horizons_solved)),
        "horizons_refused": ";".join(str(h) for h in sorted(horizons_refused)),
        "notes": " | ".join(diagnostics.notes),
    }


@dataclass
class DiagnosticsCollector:
    """Accumulates matrix diagnostics, per-horizon statuses and pair counts."""

    matrix_records: list[dict[str, object]] = field(default_factory=list)
    solver_records: list[dict[str, object]] = field(default_factory=list)
    pair_count_records: list[dict[str, object]] = field(default_factory=list)

    def add_matrix(self, record: dict[str, object]) -> None:
        self.matrix_records.append(record)

    def add_solver_status(
        self,
        *,
        station: str,
        fold_or_window_id: str,
        origin_date: object,
        p: int,
        horizon: int,
        solver_status: str,
    ) -> None:
        self.solver_records.append(
            {
                "station": station,
                "fold_or_window_id": fold_or_window_id,
                "origin_date": origin_date,
                "p": p,
                "horizon": horizon,
                "solver_status": solver_status,
            }
        )

    def add_pair_counts(
        self, *, station: str, origin_date: object, n_pairs, gamma
    ) -> None:
        for lag, (count, value) in enumerate(zip(n_pairs, gamma)):
            self.pair_count_records.append(
                {
                    "station": station,
                    "origin_date": origin_date,
                    "lag": int(lag),
                    "n_pairs": int(count),
                    "gamma": float(value),
                }
            )

    # -- materialisation ---------------------------------------------------
    def matrix_frame(self) -> pd.DataFrame:
        if not self.matrix_records:
            return pd.DataFrame(columns=list(DIAGNOSTIC_COLUMNS))
        return pd.DataFrame(self.matrix_records)[list(DIAGNOSTIC_COLUMNS)]

    def solver_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.solver_records)

    def pair_count_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.pair_count_records)

    def summary(self) -> pd.DataFrame:
        """Per ``(station, p)`` counts of valid and refused matrices."""
        frame = self.matrix_frame()
        if frame.empty:
            return frame
        grouped = (
            frame.groupby(["station", "p", "solver_status"], observed=True)
            .size()
            .rename("n_folds")
            .reset_index()
        )
        return grouped.sort_values(["station", "p", "solver_status"]).reset_index(drop=True)
