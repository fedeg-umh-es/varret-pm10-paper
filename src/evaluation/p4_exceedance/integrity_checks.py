"""Duplicate detection and case-alignment checks (Fase 5 / S4 / S5).

These checks are mandatory gates: ranking_comparison must not compute
Kendall tau or a reversal verdict on data that has unresolved duplicates
or model-to-model case misalignment. Nothing here silently drops rows or
silently intersects case sets — silent intersection is only available
through the explicitly separate COMMON_SUPPORT_SENSITIVITY mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DEFAULT_KEY_PRIORITY = (
    "station",
    "model",
    "origin_date",
    "origin_time",
    "target_date",
    "horizon",
    "fold_id",
)

# Columns that identify a "case" independent of which model produced it.
# fold_id is deliberately part of the *case* key, not the group key: in
# this repository fold_id is 1:1 with origin_date (one fold == one
# forecast origin), so grouping by fold_id would trivialize every group to
# a single case and silently hide a model that is entirely missing from a
# fold (it would just not appear in that group, rather than showing up as
# "missing" from it).
DEFAULT_CASE_KEY_PRIORITY = (
    "origin_date",
    "origin_time",
    "target_date",
    "horizon",
    "fold_id",
)

COMMON_SUPPORT_SENSITIVITY = "COMMON_SUPPORT_SENSITIVITY"


def _resolve_key_columns(df: pd.DataFrame, priority: tuple) -> list:
    cols = [c for c in priority if c in df.columns]
    return cols


@dataclass
class DuplicateReport:
    key_columns: list
    n_duplicates: int
    duplicate_keys: pd.DataFrame
    affected_models: list
    affected_stations: list

    @property
    def has_duplicates(self) -> bool:
        return self.n_duplicates > 0

    def to_dict(self) -> dict:
        return {
            "key_columns": list(self.key_columns),
            "n_duplicates": self.n_duplicates,
            "affected_models": list(self.affected_models),
            "affected_stations": list(self.affected_stations),
        }


def detect_duplicates(df: pd.DataFrame, key_columns: list | None = None) -> DuplicateReport:
    """Detect exact duplicate rows by the evaluation key.

    A duplicate is >1 row sharing (station, model, origin_*, target_date,
    horizon[, fold_id]). Duplicates are counted and returned for the
    caller to act on; they are never removed here.
    """
    if key_columns is None:
        key_columns = _resolve_key_columns(df, DEFAULT_KEY_PRIORITY)
        required_minimum = {"station", "model", "horizon"}
        if not required_minimum.issubset(set(key_columns)):
            missing = required_minimum - set(key_columns)
            raise ValueError(
                f"Cannot build a duplicate-detection key: missing columns {sorted(missing)}."
            )
        has_target = "target_date" in key_columns
        has_origin = ("origin_date" in key_columns) or ("origin_time" in key_columns)
        if not (has_target and has_origin):
            raise ValueError(
                "Cannot build a duplicate-detection key: need an origin timestamp "
                "and 'target_date' (run adapt_schema first)."
            )

    if not key_columns:
        raise ValueError("No usable key columns for duplicate detection.")

    dup_mask = df.duplicated(subset=key_columns, keep=False)
    dup_rows = df.loc[dup_mask, key_columns].drop_duplicates()
    n_duplicates = int(dup_mask.sum())

    affected_models = (
        sorted(df.loc[dup_mask, "model"].unique().tolist())
        if "model" in key_columns and dup_mask.any()
        else []
    )
    affected_stations = (
        sorted(df.loc[dup_mask, "station"].unique().tolist())
        if "station" in key_columns and dup_mask.any()
        else []
    )

    return DuplicateReport(
        key_columns=key_columns,
        n_duplicates=n_duplicates,
        duplicate_keys=dup_rows.reset_index(drop=True),
        affected_models=affected_models,
        affected_stations=affected_stations,
    )


@dataclass
class AlignmentReport:
    group_columns: list
    case_columns: list
    is_aligned: bool
    misaligned_groups: list = field(default_factory=list)
    misalignment_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    common_support_table: pd.DataFrame | None = None

    def to_dict(self) -> dict:
        return {
            "group_columns": list(self.group_columns),
            "case_columns": list(self.case_columns),
            "is_aligned": self.is_aligned,
            "misaligned_groups": list(self.misaligned_groups),
            "n_misaligned_rows": int(len(self.misalignment_table)),
        }


def check_common_support(
    df: pd.DataFrame,
    group_columns: list | None = None,
    case_columns: list | None = None,
    *,
    mode: str = "STRICT",
) -> AlignmentReport:
    """Check that every model shares exactly the same cases within each group.

    mode:
        "STRICT" (default): report misalignment, do not resolve it.
        COMMON_SUPPORT_SENSITIVITY: additionally compute the intersection
            case set per group as an explicit, separately labeled
            sensitivity view. This must never be used silently as the
            main analysis.
    """
    if mode not in ("STRICT", COMMON_SUPPORT_SENSITIVITY):
        raise ValueError(f"Unsupported mode '{mode}'.")

    if group_columns is None:
        group_columns = [c for c in ("station", "horizon") if c in df.columns]
    if not group_columns:
        raise ValueError("No usable group columns for common-support check.")
    if "fold_id" in group_columns:
        raise ValueError(
            "'fold_id' must not be a group column: it is treated as part of "
            "the case identity (see DEFAULT_CASE_KEY_PRIORITY) so that a "
            "model entirely missing from one fold is detected as a missing "
            "case rather than silently absent from a group."
        )

    if case_columns is None:
        case_columns = _resolve_key_columns(df, DEFAULT_CASE_KEY_PRIORITY)
    if not case_columns:
        raise ValueError("No usable case-identifying columns.")

    if "model" not in df.columns:
        raise ValueError("Column 'model' is required for common-support checks.")

    misaligned_groups = []
    misalignment_rows = []
    common_support_rows = []

    for group_key, group_df in df.groupby(group_columns, dropna=False):
        group_key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        per_model_cases = {
            model: set(map(tuple, model_df[case_columns].to_numpy()))
            for model, model_df in group_df.groupby("model")
        }
        if len(per_model_cases) <= 1:
            continue

        union_cases = set.union(*per_model_cases.values())
        intersection_cases = set.intersection(*per_model_cases.values())

        if union_cases != intersection_cases:
            misaligned_groups.append(group_key_tuple)
            for model, cases in per_model_cases.items():
                missing = union_cases - cases
                for case in missing:
                    row = dict(zip(group_columns, group_key_tuple))
                    row["model"] = model
                    row.update(dict(zip(case_columns, case)))
                    row["n_cases_model"] = len(cases)
                    row["n_cases_union"] = len(union_cases)
                    misalignment_rows.append(row)

        if mode == COMMON_SUPPORT_SENSITIVITY:
            for case in intersection_cases:
                row = dict(zip(group_columns, group_key_tuple))
                row.update(dict(zip(case_columns, case)))
                common_support_rows.append(row)

    misalignment_table = pd.DataFrame(
        misalignment_rows,
        columns=[*group_columns, "model", *case_columns, "n_cases_model", "n_cases_union"],
    )
    common_support_table = (
        pd.DataFrame(common_support_rows, columns=[*group_columns, *case_columns])
        if mode == COMMON_SUPPORT_SENSITIVITY
        else None
    )

    return AlignmentReport(
        group_columns=group_columns,
        case_columns=case_columns,
        is_aligned=(len(misaligned_groups) == 0),
        misaligned_groups=misaligned_groups,
        misalignment_table=misalignment_table,
        common_support_table=common_support_table,
    )
