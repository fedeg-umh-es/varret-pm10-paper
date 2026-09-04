#!/usr/bin/env python3
"""P4-v2 Phase 0B: freeze the PM10 daily contract and station panel.

This producer reads only the official PM10 source files and Phase-0 metadata.
It performs no forecasting, model fitting, predictive filtering, or forecast
metric calculation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MANIFESTS = ROOT / "manifests"
DOCS = ROOT / "docs"

YEARS = (2020, 2021, 2022, 2023)
DAILY_MIN_HOURS = 18
MIN_USABLE_YEARS = 3
MIN_USABLE_DAYS_PER_YEAR = 300
MIN_TOTAL_COVERAGE = 0.75
PM10_CODE = "10"
RAW_MANIFEST = MANIFESTS / "raw_data_manifest.json"
DAILY_CANDIDATE = PROCESSED / "pm10_daily_candidate.parquet"

PM10_FILES = {year: RAW / f"miteco_{year}_PM10.csv" for year in YEARS}

OUTPUTS = {
    "daily": PROCESSED / "pm10_daily_canonical.parquet",
    "panel": PROCESSED / "station_panel_frozen.csv",
    "ledger": RESULTS / "station_exclusion_ledger.csv",
    "identity": RESULTS / "station_identity_audit.csv",
    "manifest": MANIFESTS / "station_panel_manifest.json",
    "contract": DOCS / "P4_V2_STATION_INCLUSION_CONTRACT.md",
    "freeze": DOCS / "P4_V2_DATA_FREEZE.md",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def assert_inputs() -> None:
    if not RAW_MANIFEST.exists() or not DAILY_CANDIDATE.exists():
        raise FileNotFoundError("Phase-0 raw manifest or daily candidate is missing")
    for path in PM10_FILES.values():
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    # The Phase-0 contract is intentionally superseded by this frozen contract;
    # all newly created Phase-0B artefacts remain protected from overwrite.
    existing = [str(path) for key, path in OUTPUTS.items() if key != "contract" and path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite Phase-0B outputs: " + ", ".join(existing))


def parse_day(row: list[str]) -> date:
    return date(int(row[5]), int(row[6]), int(row[7]))


def main() -> None:
    assert_inputs()

    daily: dict[tuple[str, str], dict[str, Any]] = {}
    point_identities: defaultdict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    code_points: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    point_years: defaultdict[str, set[int]] = defaultdict(set)

    source_rows = 0
    duplicate_station_days = 0
    for year, path in PM10_FILES.items():
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh, delimiter=";")
            header = next(reader)
            for line_number, row in enumerate(reader, start=2):
                if len(row) != 32:
                    raise ValueError(f"Unexpected row width at {path}:{line_number}")
                if row[3].strip() != PM10_CODE:
                    raise ValueError(f"Non-PM10 row at {path}:{line_number}")
                source_day = parse_day(row)
                station_id = row[4].strip()
                if not station_id:
                    raise ValueError(f"Blank PUNTO_MUESTREO at {path}:{line_number}")
                day_key = (station_id, source_day.isoformat())
                if day_key in daily:
                    duplicate_station_days += 1
                    raise ValueError(f"Duplicate station/day at {path}:{line_number}: {day_key}")
                daily[day_key] = {
                    "station_id": station_id,
                    "date": source_day.isoformat(),
                    "sum": 0.0,
                    "n_valid_hours": 0,
                    "source_year": year,
                }
                identity = (row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip())
                point_identities[station_id].add(identity)
                code_points[(row[0].strip(), row[1].strip(), row[2].strip())].add(station_id)
                point_years[station_id].add(year)
                source_rows += 1
                for raw_value in row[8:32]:
                    raw_value = raw_value.strip()
                    if not raw_value:
                        continue
                    value = float(raw_value)
                    daily[day_key]["sum"] += value
                    daily[day_key]["n_valid_hours"] += 1

    if duplicate_station_days:
        raise ValueError(f"Duplicate station/day rows: {duplicate_station_days}")

    daily_rows = []
    for key in sorted(daily):
        row = daily[key]
        n_valid = int(row["n_valid_hours"])
        daily_rows.append({
            "station_id": row["station_id"],
            "date": row["date"],
            "pm10_daily": row["sum"] / n_valid if n_valid >= DAILY_MIN_HOURS else None,
            "n_valid_hours": n_valid,
            "daily_coverage": n_valid / 24.0,
            "source_year": int(row["source_year"]),
        })
    daily_schema = pa.schema([
        ("station_id", pa.string()),
        ("date", pa.string()),
        ("pm10_daily", pa.float64()),
        ("n_valid_hours", pa.int16()),
        ("daily_coverage", pa.float64()),
        ("source_year", pa.int16()),
    ])
    daily_table = pa.Table.from_pydict({
        "station_id": [row["station_id"] for row in daily_rows],
        "date": [row["date"] for row in daily_rows],
        "pm10_daily": [row["pm10_daily"] for row in daily_rows],
        "n_valid_hours": [row["n_valid_hours"] for row in daily_rows],
        "daily_coverage": [row["daily_coverage"] for row in daily_rows],
        "source_year": [row["source_year"] for row in daily_rows],
    }, schema=daily_schema)
    pq.write_table(daily_table, OUTPUTS["daily"], compression="zstd")

    metadata: dict[str, dict[str, str]] = {}
    with (PROCESSED / "station_metadata_all_pm10.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            metadata[row["station_id"]] = row

    identity_rows = []
    for station_id in sorted(point_identities):
        identities = sorted(point_identities[station_id])
        identity_rows.append({
            "station_id": station_id,
            "province": identities[0][0],
            "municipality": identities[0][1],
            "station_code": identities[0][2],
            "magnitude": identities[0][3],
            "years_present": ",".join(str(year) for year in sorted(point_years[station_id])),
            "unique_identity_count": len(identities),
            "id_stable": "YES" if len(identities) == 1 else "NO",
            "station_code_point_count": len(code_points[(identities[0][0], identities[0][1], identities[0][2])]),
            "identity_values": json.dumps(identities, ensure_ascii=False),
        })
    write_csv(OUTPUTS["identity"], list(identity_rows[0].keys()), identity_rows)

    daily_by_station: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily_rows:
        daily_by_station[row["station_id"]].append(row)

    metrics: dict[str, dict[str, Any]] = {}
    for station_id in sorted(point_identities):
        rows = daily_by_station[station_id]
        dates = sorted(date.fromisoformat(row["date"]) for row in rows)
        first_date = dates[0]
        last_date = dates[-1]
        support_days = (last_date - first_date).days + 1
        usable_rows = [row for row in rows if row["n_valid_hours"] >= DAILY_MIN_HOURS]
        usable_days_by_year = {
            year: sum(row["n_valid_hours"] >= DAILY_MIN_HOURS for row in rows if row["source_year"] == year)
            for year in YEARS
        }
        usable_years = sum(count >= MIN_USABLE_DAYS_PER_YEAR for count in usable_days_by_year.values())
        coverage_total = len(usable_rows) / support_days
        stable = len(point_identities[station_id]) == 1 and metadata[station_id]["metadata_conflict_flag"] == "NO"
        fails_id_stability = not stable
        fails_daily_support = len(usable_rows) == 0
        fails_year_count = sum(bool(point_years[station_id] and year in point_years[station_id]) for year in YEARS) < MIN_USABLE_YEARS
        fails_300_days_rule = usable_years < MIN_USABLE_YEARS
        fails_total_coverage = coverage_total < MIN_TOTAL_COVERAGE
        selected = not any((fails_id_stability, fails_daily_support, fails_year_count, fails_300_days_rule, fails_total_coverage))
        metrics[station_id] = {
            "station_id": station_id,
            "province": metadata[station_id]["provincia"],
            "municipality": metadata[station_id]["municipio"],
            "station_code": metadata[station_id]["estacion"],
            "first_date": first_date.isoformat(),
            "last_date": last_date.isoformat(),
            "support_days": support_days,
            "usable_days": len(usable_rows),
            "usable_years": usable_years,
            "coverage_total": coverage_total,
            "source_years_present": len(point_years[station_id]),
            "usable_days_2020": usable_days_by_year[2020],
            "usable_days_2021": usable_days_by_year[2021],
            "usable_days_2022": usable_days_by_year[2022],
            "usable_days_2023": usable_days_by_year[2023],
            "fails_id_stability": fails_id_stability,
            "fails_daily_support": fails_daily_support,
            "fails_year_count": fails_year_count,
            "fails_300_days_rule": fails_300_days_rule,
            "fails_total_coverage": fails_total_coverage,
            "selected": selected,
        }

    selected_rows = []
    for station_id in sorted(metrics):
        row = metrics[station_id]
        if not row["selected"]:
            continue
        selected_rows.append({
            "station_id": row["station_id"],
            "province": row["province"],
            "municipality": row["municipality"],
            "station_code": row["station_code"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
            "usable_days": row["usable_days"],
            "usable_years": row["usable_years"],
            "coverage_total": f"{row['coverage_total']:.12f}",
            "support_days": row["support_days"],
            "source_years_present": row["source_years_present"],
            "usable_days_2020": row["usable_days_2020"],
            "usable_days_2021": row["usable_days_2021"],
            "usable_days_2022": row["usable_days_2022"],
            "usable_days_2023": row["usable_days_2023"],
        })
    panel_fields = list(selected_rows[0].keys()) if selected_rows else []
    write_csv(OUTPUTS["panel"], panel_fields, selected_rows)

    ledger_rows = []
    for station_id in sorted(metrics):
        row = metrics[station_id]
        if row["selected"]:
            continue
        reason_codes = [
            name for name in (
                "fails_id_stability", "fails_daily_support", "fails_year_count",
                "fails_300_days_rule", "fails_total_coverage",
            ) if row[name]
        ]
        ledger_rows.append({
            "station_id": row["station_id"],
            "province": row["province"],
            "municipality": row["municipality"],
            "station_code": row["station_code"],
            "fails_id_stability": "YES" if row["fails_id_stability"] else "NO",
            "fails_daily_support": "YES" if row["fails_daily_support"] else "NO",
            "fails_year_count": "YES" if row["fails_year_count"] else "NO",
            "fails_300_days_rule": "YES" if row["fails_300_days_rule"] else "NO",
            "fails_total_coverage": "YES" if row["fails_total_coverage"] else "NO",
            "reason_codes": ";".join(reason_codes),
        })
    ledger_fields = list(ledger_rows[0].keys()) if ledger_rows else [
        "station_id", "province", "municipality", "station_code",
        "fails_id_stability", "fails_daily_support", "fails_year_count",
        "fails_300_days_rule", "fails_total_coverage", "reason_codes",
    ]
    write_csv(OUTPUTS["ledger"], ledger_fields, ledger_rows)

    contract_text = f"""# P4-v2 Station Inclusion Contract

**Status: FROZEN_BEFORE_FORECASTING**

This contract is frozen for the new P4-v2 PM10 benchmark before any forecasting model is trained or evaluated. It is a data-selection contract only.

## Canonical identity

- Pollutant: PM10, represented by `MAGNITUD=10` in the official source.
- Canonical `station_id`: the original `PUNTO_MUESTREO` field.
- A point is stable only if it maps to exactly one `(PROVINCIA, MUNICIPIO, ESTACION, MAGNITUD)` tuple across 2020–2023 and has no duplicate station/date source records. Distinct `PUNTO_MUESTREO` values are never merged merely because they share a station code.

## Daily aggregation

For each source station-day, `pm10_daily` is the arithmetic mean of valid hourly PM10 observations belonging to that UTC calendar day, represented by the MITECO `ANNO/MES/DIA` label and H01–H24 fields. A day is usable only when `n_valid_hours >= {DAILY_MIN_HOURS}`. Missing hourly cells are not imputed, interpolated, or forward-filled. A day below the threshold remains present in the canonical daily table with `pm10_daily` missing and is not counted as usable.

## Inclusion criteria

A PM10 `station_id` is included if and only if all conditions hold:

1. **Stable identity:** the canonical `PUNTO_MUESTREO` identifier passes the identity audit.
2. **Daily support:** daily values are usable only at the `{DAILY_MIN_HOURS}/24` rule above.
3. **Year support:** at least {MIN_USABLE_YEARS} calendar years in 2020–2023 contain at least {MIN_USABLE_DAYS_PER_YEAR} usable station-days each.
4. **Total usable-day coverage:** `coverage_total = usable_days / support_days >= {MIN_TOTAL_COVERAGE}`. `support_days` is the inclusive number of UTC calendar days from the station's first to last source-supported date in 2020–2023. This denominator counts days without a usable observation as missing and does not count dates outside the station's observed temporal support.
5. **Forecast-independence:** no model error, forecast skill, variance retention, event performance, ranking, or other predictive result is used.

## Rationale

- The original `PUNTO_MUESTREO` identifier avoids merging distinct measurement points and preserves the source's experimental series identity.
- The 18-hour rule prevents a daily mean from being driven by a small fraction of a day while retaining a transparent, fixed completeness rule.
- Three usable calendar years and 300 usable days per year require multi-year support for later rolling-origin design without selecting stations from predictive outcomes.
- The 0.75 usable-day coverage floor limits prolonged missingness over each station's actual 2020–2023 support.
- All thresholds are frozen data-contract choices, not regulatory limits, universal air-quality standards, or forecasting-performance thresholds.

## Frozen outcome

The deterministic application of this contract selects **{len(selected_rows)}** of **{len(metrics)}** PM10 sampling-point series. The count is an output of the frozen definitions and was not targeted.
"""
    OUTPUTS["contract"].write_text(contract_text, encoding="utf-8")

    hashes = {
        "raw_data_manifest_sha256": sha256(RAW_MANIFEST),
        "canonical_daily_parquet_sha256": sha256(OUTPUTS["daily"]),
        "frozen_panel_sha256": sha256(OUTPUTS["panel"]),
        "exclusion_ledger_sha256": sha256(OUTPUTS["ledger"]),
        "inclusion_contract_sha256": sha256(OUTPUTS["contract"]),
    }
    manifest = {
        "manifest_type": "P4-v2 frozen station panel manifest",
        "created_at_utc": utc_now(),
        "producer_script": "p4_v2/phase0b_freeze.py",
        "git_head": git_head(),
        "station_id_field": "PUNTO_MUESTREO",
        "pollutant": "PM10",
        "daily_rule": "arithmetic mean of valid hourly PM10 observations belonging to the MITECO UTC source calendar day; usable iff n_valid_hours >= 18; no imputation",
        "thresholds": {
            "min_valid_hours_per_day": DAILY_MIN_HOURS,
            "min_usable_years": MIN_USABLE_YEARS,
            "min_usable_days_per_year": MIN_USABLE_DAYS_PER_YEAR,
            "min_total_coverage": MIN_TOTAL_COVERAGE,
            "coverage_denominator": "inclusive UTC calendar days from first to last source-supported date per station",
        },
        "station_count_total_pm10_series": len(metrics),
        "station_count_frozen": len(selected_rows),
        "excluded_station_count": len(ledger_rows),
        "identity_audit": {
            "unique_punto_muestreo": len(point_identities),
            "unique_provincia_municipio_estacion": len(code_points),
            "point_to_multiple_station_codes": sum(len(value) > 1 for value in point_identities.values()),
            "station_code_to_multiple_points": sum(len(value) > 1 for value in code_points.values()),
            "unstable_point_ids": sum(len(value) > 1 for value in point_identities.values()),
        },
        **hashes,
    }
    write_json(OUTPUTS["manifest"], manifest)

    freeze_text = f"""# P4-v2 Data Freeze

**Status: FROZEN_BEFORE_FORECASTING**  
**Created:** `{manifest['created_at_utc']}`  
**Git HEAD:** `{manifest['git_head']}`

## Freeze statement

The station inclusion contract and station panel were frozen before any forecasting model was trained or evaluated. Future station exclusions are forbidden unless caused by a documented data-integrity failure unrelated to model performance. Any such change requires a new panel version and explicit audit trail.

## Frozen artefacts

- Canonical station identifier: `PUNTO_MUESTREO`.
- Frozen daily table: `p4_v2/data/processed/pm10_daily_canonical.parquet`.
- Frozen station panel: `p4_v2/data/processed/station_panel_frozen.csv`.
- Exclusion ledger: `p4_v2/results/station_exclusion_ledger.csv`.
- Identity audit: `p4_v2/results/station_identity_audit.csv`.
- Inclusion contract: `p4_v2/docs/P4_V2_STATION_INCLUSION_CONTRACT.md`.
- Frozen station count: **{len(selected_rows)}** of **{len(metrics)}** PM10 sampling-point series.

## Scientific boundary

This freeze contains only source-data identity, temporal support, daily completeness, and missingness decisions. No model, forecast, error metric, skill, variance-retention result, event result, rank, or eligibility outcome was used. Model training remains unauthorized until the Phase-1 protocol is separately frozen.

## Hashes

```text
raw_data_manifest_sha256       = {hashes['raw_data_manifest_sha256']}
canonical_daily_parquet_sha256 = {hashes['canonical_daily_parquet_sha256']}
frozen_panel_sha256             = {hashes['frozen_panel_sha256']}
exclusion_ledger_sha256         = {hashes['exclusion_ledger_sha256']}
inclusion_contract_sha256       = {hashes['inclusion_contract_sha256']}
```
"""
    OUTPUTS["freeze"].write_text(freeze_text, encoding="utf-8")

    print(json.dumps({
        "canonical_station_id": "PUNTO_MUESTREO",
        "unique_punto_muestreo": len(point_identities),
        "unique_station_codes": len(code_points),
        "station_code_to_multiple_points": sum(len(value) > 1 for value in code_points.values()),
        "unstable_point_ids": sum(len(value) > 1 for value in point_identities.values()),
        "daily_rows": len(daily_rows),
        "source_rows": source_rows,
        "frozen_station_count": len(selected_rows),
        "excluded_station_count": len(ledger_rows),
        "hashes": hashes,
        "data_selection_used_forecast_results": "NO",
        "model_training_authorized": "NO",
    }, indent=2))


if __name__ == "__main__":
    main()
