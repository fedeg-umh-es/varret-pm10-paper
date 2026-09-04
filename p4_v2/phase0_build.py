#!/usr/bin/env python3
"""P4-v2 Phase 0: MITECO PM10 extraction and data-quality audit.

This script performs no forecasting and computes no forecast metrics.  It reads
the official annual PM10 files, preserves source identifiers, omits empty hourly
cells, and produces deterministic data-availability artefacts for review.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
MANIFESTS = ROOT / "manifests"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

YEARS = (2020, 2021, 2022, 2023)
PM10_INPUTS = {year: RAW / f"miteco_{year}_PM10.csv" for year in YEARS}
EXPECTED_PM10_CODE = "10"
DAILY_THRESHOLDS = (0.75, 0.80, 0.90, 0.95, 1.00)
DAILY_MIN_HOURS = {threshold: math.ceil(24 * threshold) for threshold in DAILY_THRESHOLDS}

DATASET_ID = "19458583-9953-4fe7-a494-e2cc26e89e58"
OFFICIAL_SOURCES = {
    2020: {
        "official_page": "https://www.miteco.gob.es/es/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/evaluacion-y-datos-de-calidad-del-aire/datos/datos_oficiales_2020.html",
        "download_url": "https://www.miteco.gob.es/content/dam/miteco/es/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/horarios_2020_tcm30-531262.zip",
        "resource_id": "b2bcb58a-b85d-4820-8c14-2dc4326c5392",
        "source_filename": "PM10_HH_2020.csv",
        "zip_filename": "miteco_official_2020.zip",
        "zip_entry": "PM10_HH_2020.csv",
    },
    2021: {
        "official_page": "https://www.miteco.gob.es/es/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/evaluacion-y-datos-de-calidad-del-aire/datos/datos_oficiales_2021.html",
        "download_url": "https://www.miteco.gob.es/content/dam/miteco/es/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/datos_horarios_tcm30-545543.zip",
        "resource_id": "15eddab1-fb90-4709-80c6-50b60d82b4d1",
        "source_filename": "PM10_HH_2021.csv",
        "zip_filename": "miteco_official_2021.zip",
        "zip_entry": "Datos_horarios/PM10_HH_2021.csv",
    },
    2022: {
        "official_page": "https://www.miteco.gob.es/es/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/evaluacion-y-datos-de-calidad-del-aire/datos/datos-oficiales-2022.html",
        "download_url": "https://www.miteco.gob.es/content/dam/miteco/es/calidad-y-evaluacion-ambiental/sgalsi/atm%C3%B3sfera-y-calidad-del-aire/evaluaci%C3%B3n-2022/Datos%20horarios%202022.zip",
        "resource_id": "f160ba6e-df57-4d8a-a900-d17516832290",
        "source_filename": "PM10_HH_2022.csv",
        "zip_filename": "miteco_official_2022.zip",
        "zip_entry": "Datos horarios 2022/PM10_HH_2022.csv",
    },
    2023: {
        "official_page": "https://www.miteco.gob.es/en/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/evaluacion-y-datos-de-calidad-del-aire/datos/datos-oficiales-2023.html",
        "download_url": "https://www.miteco.gob.es/content/dam/miteco/es/calidad-y-evaluacion-ambiental/sgalsi/atm%C3%B3sfera-y-calidad-del-aire/evaluaci%C3%B3n-2023/Datos%20horarios%202023.zip",
        "resource_id": "26f082ac-1176-4253-8ff8-3926be818637",
        "source_filename": "PM10_HH_2023.csv",
        "zip_filename": "miteco_official_2023.zip",
        "zip_entry": "Datos horarios 2023/PM10_HH_2023.csv",
    },
}

FORMAT_DOCUMENT = "https://www.miteco.gob.es/content/dam/miteco/es/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/formatodatoscsvmiteco_web_tcm30-501420.pdf"

EXPECTED_HEADER = [
    "PROVINCIA", "MUNICIPIO", "ESTACION", "MAGNITUD", "PUNTO_MUESTREO",
    "ANNO", "MES", "DIA",
] + [f"H{i:02d}" for i in range(1, 25)]

HOURLY_COLUMNS = [
    "station_id", "timestamp_utc", "pm10", "quality_flag", "source_year",
    "provincia", "municipio", "estacion", "magnitud", "punto_muestreo",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {"path": str(path), "filename": path.name, "byte_size": stat.st_size, "sha256": sha256(path), "filesystem_mtime_utc": mtime}


def year_hours(year: int) -> int:
    return (date(year + 1, 1, 1) - date(year, 1, 1)).days * 24


def parse_source_date(row: list[str]) -> date:
    return date(int(row[5]), int(row[6]), int(row[7]))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def require_inputs() -> None:
    for year, path in PM10_INPUTS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size == 0:
            raise ValueError(f"Empty input: {path}")


def require_outputs_absent() -> None:
    outputs = [
        MANIFESTS / "raw_data_manifest.json",
        DOCS / "miteco_schema_audit.md",
        DOCS / "P4_V2_STATION_INCLUSION_CONTRACT.md",
        PROCESSED / "pm10_hourly.parquet",
        PROCESSED / "pm10_daily_candidate.parquet",
        PROCESSED / "station_metadata_all_pm10.csv",
        RESULTS / "station_coverage_audit.csv",
        RESULTS / "daily_aggregation_sensitivity.csv",
        RESULTS / "station_panel_candidate.csv",
        INTERIM / "phase0_observation_summary.json",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing Phase-0 outputs: " + ", ".join(existing))


def inspect_headers() -> tuple[dict[int, dict[str, Any]], dict[int, Counter[str]]]:
    schemas: dict[int, dict[str, Any]] = {}
    stats: dict[int, Counter[str]] = {}
    for year, path in PM10_INPUTS.items():
        counter: Counter[str] = Counter()
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh, delimiter=";")
            header = next(reader)
            if header != EXPECTED_HEADER:
                raise ValueError(f"Unexpected header in {path}: {header}")
            counter["rows"] = 0
            counter["hour_cells"] = 0
            counter["blank_hour_cells"] = 0
            counter["valid_hour_cells"] = 0
            counter["bad_width_rows"] = 0
            counter["bad_date_rows"] = 0
            counter["bad_numeric_cells"] = 0
            counter["station_ids"] = 0
            station_ids: set[str] = set()
            min_day: str | None = None
            max_day: str | None = None
            magnitude_counts: Counter[str] = Counter()
            for row in reader:
                counter["rows"] += 1
                if len(row) != len(header):
                    counter["bad_width_rows"] += 1
                    continue
                station_ids.add(row[4].strip())
                magnitude_counts[row[3].strip()] += 1
                counter["hour_cells"] += 24
                try:
                    source_day = parse_source_date(row)
                    min_day = min(min_day, source_day.isoformat()) if min_day else source_day.isoformat()
                    max_day = max(max_day, source_day.isoformat()) if max_day else source_day.isoformat()
                except Exception:
                    counter["bad_date_rows"] += 1
                for raw in row[8:32]:
                    if raw.strip() == "":
                        counter["blank_hour_cells"] += 1
                    else:
                        try:
                            float(raw)
                            counter["valid_hour_cells"] += 1
                        except ValueError:
                            counter["bad_numeric_cells"] += 1
            counter["station_ids"] = len(station_ids)
            schemas[year] = {
                "header": header,
                "delimiter": ";",
                "encoding_observed": "UTF-8",
                "year": year,
                "date_min": min_day,
                "date_max": max_day,
                "magnitude_counts": dict(magnitude_counts),
            }
            stats[year] = counter
    return schemas, stats


def parquet_writer(path: Path) -> pq.ParquetWriter:
    schema = pa.schema([
        ("station_id", pa.string()),
        ("timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("pm10", pa.float64()),
        ("quality_flag", pa.string()),
        ("source_year", pa.int16()),
        ("provincia", pa.string()),
        ("municipio", pa.string()),
        ("estacion", pa.string()),
        ("magnitud", pa.string()),
        ("punto_muestreo", pa.string()),
    ])
    return pq.ParquetWriter(path, schema=schema, compression="zstd")


def write_hourly_batch(writer: pq.ParquetWriter, batch: list[tuple[Any, ...]]) -> None:
    if not batch:
        return
    columns = list(zip(*batch))
    arrays = [
        pa.array(columns[0], type=pa.string()),
        pa.array(columns[1], type=pa.timestamp("us", tz="UTC")),
        pa.array(columns[2], type=pa.float64()),
        pa.array(columns[3], type=pa.string()),
        pa.array(columns[4], type=pa.int16()),
        pa.array(columns[5], type=pa.string()),
        pa.array(columns[6], type=pa.string()),
        pa.array(columns[7], type=pa.string()),
        pa.array(columns[8], type=pa.string()),
        pa.array(columns[9], type=pa.string()),
    ]
    writer.write_table(pa.Table.from_arrays(arrays, names=HOURLY_COLUMNS))


def main() -> None:
    require_inputs()
    require_outputs_absent()
    for directory in (PROCESSED, INTERIM, MANIFESTS, RESULTS, DOCS):
        directory.mkdir(parents=True, exist_ok=True)

    schemas, source_stats = inspect_headers()

    metadata: dict[str, dict[str, Any]] = {}
    metadata_conflicts: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    daily: dict[tuple[str, str], dict[str, Any]] = {}
    masks: dict[tuple[str, int], bytearray] = {}
    station_year_counts: Counter[tuple[str, int]] = Counter()
    duplicate_daily_records: Counter[int] = Counter()
    hourly_path = PROCESSED / "pm10_hourly.parquet"
    hourly_writer = parquet_writer(hourly_path)
    hourly_batch: list[tuple[Any, ...]] = []
    total_valid = 0

    try:
        for year, path in PM10_INPUTS.items():
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.reader(fh, delimiter=";")
                header = next(reader)
                if header != EXPECTED_HEADER:
                    raise ValueError(f"Unexpected header in {path}")
                for row_number, row in enumerate(reader, start=2):
                    if len(row) != 32:
                        raise ValueError(f"Bad row width {path}:{row_number}")
                    station_id = row[4].strip()
                    if not station_id:
                        raise ValueError(f"Blank station id {path}:{row_number}")
                    if row[3].strip() != EXPECTED_PM10_CODE:
                        raise ValueError(f"Unexpected MAGNITUD in {path}:{row_number}: {row[3]}")
                    source_day = parse_source_date(row)
                    day_key = (station_id, source_day.isoformat())
                    if day_key in daily:
                        duplicate_daily_records[year] += 1
                        raise ValueError(f"Duplicate station-day in {path}:{row_number}: {day_key}")
                    meta = {
                        "station_id": station_id,
                        "provincia": row[0].strip(),
                        "municipio": row[1].strip(),
                        "estacion": row[2].strip(),
                        "magnitud": row[3].strip(),
                        "punto_muestreo": station_id,
                    }
                    if station_id in metadata and any(metadata[station_id][field] != meta[field] for field in ("provincia", "municipio", "estacion", "magnitud")):
                        metadata_conflicts[station_id].append(meta)
                    metadata.setdefault(station_id, meta)
                    daily[day_key] = {
                        **meta,
                        "date_utc": source_day.isoformat(),
                        "source_year": year,
                        "sum": 0.0,
                        "valid_hour_count": 0,
                    }
                    mask_key = (station_id, year)
                    if mask_key not in masks:
                        masks[mask_key] = bytearray(year_hours(year))
                    mask = masks[mask_key]
                    station_year_counts[mask_key] += 1
                    for hour_index, raw_value in enumerate(row[8:32], start=1):
                        raw_value = raw_value.strip()
                        if raw_value == "":
                            continue
                        value = float(raw_value)
                        timestamp = datetime(year, source_day.month, source_day.day, tzinfo=timezone.utc) + timedelta(hours=hour_index)
                        offset = int((timestamp - datetime(year, 1, 1, 1, tzinfo=timezone.utc)).total_seconds() // 3600)
                        if offset < 0 or offset >= len(mask):
                            raise ValueError(f"Timestamp offset out of range {path}:{row_number}:{hour_index}")
                        if mask[offset]:
                            raise ValueError(f"Duplicate hourly timestamp {path}:{row_number}:{hour_index}")
                        mask[offset] = 1
                        daily[day_key]["sum"] += value
                        daily[day_key]["valid_hour_count"] += 1
                        hourly_batch.append((
                            station_id, timestamp, value, None, year,
                            row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip(), station_id,
                        ))
                        total_valid += 1
                        if len(hourly_batch) >= 50_000:
                            write_hourly_batch(hourly_writer, hourly_batch)
                            hourly_batch.clear()
        write_hourly_batch(hourly_writer, hourly_batch)
    finally:
        hourly_writer.close()

    if duplicate_daily_records:
        raise ValueError(f"Duplicate daily records found: {dict(duplicate_daily_records)}")

    daily_rows = []
    for row in daily.values():
        valid_count = int(row["valid_hour_count"])
        daily_rows.append({
            "station_id": row["station_id"],
            "date_utc": row["date_utc"],
            "pm10_daily": (row["sum"] / valid_count) if valid_count else None,
            "valid_hour_count": valid_count,
            "daily_coverage_fraction": valid_count / 24.0,
            "source_year": row["source_year"],
            "provincia": row["provincia"],
            "municipio": row["municipio"],
            "estacion": row["estacion"],
            "magnitud": row["magnitud"],
            "punto_muestreo": row["punto_muestreo"],
        })
    daily_rows.sort(key=lambda row: (row["station_id"], row["date_utc"]))
    daily_table = pa.Table.from_pylist(daily_rows)
    pq.write_table(daily_table, PROCESSED / "pm10_daily_candidate.parquet", compression="zstd")

    daily_rows_by_station_year: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in daily_rows:
        daily_rows_by_station_year[(row["station_id"], row["source_year"])].append(row)

    # Station metadata is limited to fields actually present in the official PM10 files.
    metadata_rows = []
    for station_id in sorted(metadata):
        station_years = sorted(year for year in YEARS if (station_id, year) in masks)
        year_dates = [row["date_utc"] for row in daily_rows if row["station_id"] == station_id]
        metadata_rows.append({
            **metadata[station_id],
            "years_present": ",".join(map(str, station_years)),
            "source_year_count": len(station_years),
            "first_source_date": min(year_dates) if year_dates else "",
            "last_source_date": max(year_dates) if year_dates else "",
            "metadata_conflict_flag": "YES" if station_id in metadata_conflicts else "NO",
            "metadata_conflict_details": json.dumps(metadata_conflicts.get(station_id, []), ensure_ascii=False),
        })
    metadata_fields = [
        "station_id", "provincia", "municipio", "estacion", "magnitud", "punto_muestreo",
        "years_present", "source_year_count", "first_source_date", "last_source_date",
        "metadata_conflict_flag", "metadata_conflict_details",
    ]
    write_csv(PROCESSED / "station_metadata_all_pm10.csv", metadata_fields, metadata_rows)

    # Coverage and missing-run audit.  Expected hours cover every calendar year in the source,
    # including years in which a station has no record; this avoids rewarding partial panels.
    coverage_rows = []
    for station_id in sorted(metadata):
        year_records = {}
        first_timestamp: str | None = None
        last_timestamp: str | None = None
        total_valid_station = 0
        for year in YEARS:
            key = (station_id, year)
            mask = masks.get(key, bytearray(year_hours(year)))
            valid = sum(mask)
            total_valid_station += valid
            first_offset = next((i for i, flag in enumerate(mask) if flag), None)
            last_offset = next((i for i in range(len(mask) - 1, -1, -1) if mask[i]), None)
            if first_offset is not None:
                first = datetime(year, 1, 1, 1, tzinfo=timezone.utc) + timedelta(hours=first_offset)
                last = datetime(year, 1, 1, 1, tzinfo=timezone.utc) + timedelta(hours=last_offset)
                first_timestamp = min(first_timestamp, first.isoformat().replace("+00:00", "Z")) if first_timestamp else first.isoformat().replace("+00:00", "Z")
                last_timestamp = max(last_timestamp, last.isoformat().replace("+00:00", "Z")) if last_timestamp else last.isoformat().replace("+00:00", "Z")
            daily_year = daily_rows_by_station_year[(station_id, year)]
            usable_days = sum(row["valid_hour_count"] >= DAILY_MIN_HOURS[0.75] for row in daily_year)
            year_records[year] = {
                "expected": year_hours(year),
                "valid": valid,
                "coverage": valid / year_hours(year),
                "source_days": len(daily_year),
                "usable_days_75": usable_days,
            }
        concatenated = bytearray()
        for year in YEARS:
            concatenated.extend(masks.get((station_id, year), bytearray(year_hours(year))))
        longest_missing = 0
        current_missing = 0
        for flag in concatenated:
            if flag:
                current_missing = 0
            else:
                current_missing += 1
                longest_missing = max(longest_missing, current_missing)
        row = {
            "station_id": station_id,
            "provincia": metadata[station_id]["provincia"],
            "municipio": metadata[station_id]["municipio"],
            "estacion": metadata[station_id]["estacion"],
            "first_timestamp_utc": first_timestamp or "",
            "last_timestamp_utc": last_timestamp or "",
            "expected_hourly_observations": sum(year_hours(year) for year in YEARS),
            "observed_valid_pm10_observations": total_valid_station,
            "coverage_fraction": total_valid_station / sum(year_hours(year) for year in YEARS),
            "number_of_source_years_present": sum(bool(year_records[year]["source_days"]) for year in YEARS),
            "longest_missing_run_hours": longest_missing,
            "days_with_sufficient_hourly_support_75pct": sum(year_records[year]["usable_days_75"] for year in YEARS),
        }
        for year in YEARS:
            record = year_records[year]
            row[f"expected_hours_{year}"] = record["expected"]
            row[f"valid_hours_{year}"] = record["valid"]
            row[f"coverage_{year}"] = record["coverage"]
            row[f"source_days_{year}"] = record["source_days"]
            row[f"usable_days_75pct_{year}"] = record["usable_days_75"]
        coverage_rows.append(row)
    coverage_fields = list(coverage_rows[0].keys()) if coverage_rows else []
    write_csv(RESULTS / "station_coverage_audit.csv", coverage_fields, coverage_rows)

    sensitivity_rows = []
    for threshold in DAILY_THRESHOLDS:
        minimum_hours = DAILY_MIN_HOURS[threshold]
        eligible_days = [row for row in daily_rows if row["valid_hour_count"] >= minimum_hours]
        station_ids = {row["station_id"] for row in eligible_days}
        sensitivity_rows.append({
            "daily_completeness_threshold": threshold,
            "minimum_valid_hours": minimum_hours,
            "station_days_remaining": len(eligible_days),
            "stations_remaining": len(station_ids),
            "station_days_with_value": sum(row["pm10_daily"] is not None for row in eligible_days),
        })
    write_csv(RESULTS / "daily_aggregation_sensitivity.csv", list(sensitivity_rows[0].keys()), sensitivity_rows)

    coverage_by_station = {row["station_id"]: row for row in coverage_rows}
    daily_by_station_year: defaultdict[tuple[str, int], int] = defaultdict(int)
    for row in daily_rows:
        if row["valid_hour_count"] >= DAILY_MIN_HOURS[0.75]:
            daily_by_station_year[(row["station_id"], row["source_year"])] += 1

    # Proposed contract: independent of all forecast outcomes.
    contract = {
        "status": "PROPOSED",
        "daily_completeness_threshold": 0.75,
        "minimum_valid_hours_per_day": 18,
        "minimum_usable_years": 3,
        "minimum_usable_days_per_year": 300,
        "minimum_overall_hourly_coverage_fraction": 0.75,
        "station_id_stability": "exact PUNTO_MUESTREO identifier; repeated metadata fields must not conflict",
        "missing_days": "retain as missing; do not impute; exclude from any later forecast origin requiring unavailable target/input support",
        "selection_basis": "data availability, temporal support, daily completeness, and identifier stability only",
        "not_used": ["model errors", "forecast skill", "variance retention", "event performance"],
        "rationale": "The thresholds are a pre-specified audit contract for sufficient multi-year support, not thresholds tuned to any forecasting outcome or to maximize station count.",
    }
    contract_text = """# P4-v2 Station Inclusion Contract\n\n**Status: PROPOSED — not frozen.**\n\nThis contract is a Phase-0 data-availability proposal for the new MITECO benchmark. It is independent of all model outputs and must be reviewed and frozen before forecasting.\n\n## Proposed inclusion rule\n\nA PM10 station is a candidate if and only if all conditions below hold:\n\n1. The original `PUNTO_MUESTREO` identifier is present in at least 3 of the 4 source years (2020–2023).\n2. The identifier's province, municipality, station, and pollutant code do not conflict across years.\n3. At least 3 source years contain at least 300 station-days with at least 18 valid hourly PM10 observations out of 24.\n4. Across the full 2020–2023 calendar span, valid hourly observations are at least 75% of the expected hourly slots; a missing source year therefore counts as zero coverage for that year.\n5. Daily PM10 is the arithmetic mean of valid hourly PM10 observations only when at least 18 of 24 hourly observations are present. Days below this threshold remain missing; no imputation or interpolation is permitted.\n\n## Explicit non-use of forecasting outcomes\n\nStation inclusion must not inspect model errors, persistence-relative skill, variance retention, event performance, ranking, or eligibility. The contract is proposed solely from source availability, temporal support, daily completeness, and identifier stability.\n\nThe numerical thresholds are a reviewable data-contract choice, not universal air-quality standards and not forecasting-performance thresholds.\n"""
    (DOCS / "P4_V2_STATION_INCLUSION_CONTRACT.md").write_text(contract_text, encoding="utf-8")

    panel_rows = []
    for row in coverage_rows:
        station_id = row["station_id"]
        usable_years = sum(daily_by_station_year[(station_id, year)] >= 300 for year in YEARS)
        source_years = int(row["number_of_source_years_present"])
        reasons = []
        if source_years < 3:
            reasons.append("fewer_than_3_source_years")
        if metadata[station_id] and station_id in metadata_conflicts:
            reasons.append("metadata_conflict")
        if usable_years < 3:
            reasons.append("fewer_than_3_years_with_300_usable_days")
        if float(row["coverage_fraction"]) < 0.75:
            reasons.append("overall_hourly_coverage_below_0.75")
        included = not reasons
        panel_rows.append({
            "station_id": station_id,
            "provincia": row["provincia"],
            "municipio": row["municipio"],
            "estacion": row["estacion"],
            "source_years_present": source_years,
            "usable_years_300_days_at_75pct": usable_years,
            "overall_coverage_fraction": row["coverage_fraction"],
            "metadata_conflict_flag": "YES" if station_id in metadata_conflicts else "NO",
            "included_under_proposed_contract": "YES" if included else "NO",
            "exclusion_reasons": ";".join(reasons),
        })
    panel_fields = list(panel_rows[0].keys()) if panel_rows else []
    write_csv(RESULTS / "station_panel_candidate.csv", panel_fields, panel_rows)

    source_manifest = {
        "manifest_type": "P4-v2 Phase-0 raw data manifest",
        "generated_at_utc": utc_now(),
        "dataset_title": "Datos horarios de calidad del aire",
        "dataset_id": DATASET_ID,
        "publisher": "Ministerio para la Transición Ecológica y el Reto Demográfico (MITECO)",
        "access_classification": "Public according to the official MITECO open-data catalogue metadata",
        "license_note": "The official resource pages expose public access; no explicit license string was populated in the resource metadata observed.",
        "format_document": FORMAT_DOCUMENT,
        "source_years": [],
    }
    for year in YEARS:
        source = OFFICIAL_SOURCES[year]
        zip_path = RAW / source["zip_filename"]
        csv_path = PM10_INPUTS[year]
        source_manifest["source_years"].append({
            "year": year,
            "official_page": source["official_page"],
            "official_download_url": source["download_url"],
            "catalog_resource_id": source["resource_id"],
            "downloaded_file": {
                **file_info(zip_path),
                "download_timestamp_utc": file_info(zip_path)["filesystem_mtime_utc"],
                "original_filename": source["zip_filename"],
                "container": "ZIP",
            },
            "extracted_pm10_file": {
                **file_info(csv_path),
                "derived_from": source["zip_filename"],
                "archive_entry": source["zip_entry"],
                "original_filename": source["source_filename"],
                "schema": schemas[year],
                "observed_stats": dict(source_stats[year]),
            },
        })
    write_json(MANIFESTS / "raw_data_manifest.json", source_manifest)

    station_count = len(metadata)
    station_code_count = len({(row["provincia"], row["municipio"], row["estacion"]) for row in metadata.values()})
    candidate_count = sum(row["included_under_proposed_contract"] == "YES" for row in panel_rows)
    observation_summary = {
        "generated_at_utc": utc_now(),
        "input_years": list(YEARS),
        "pm10_code_observed": EXPECTED_PM10_CODE,
        "station_count_union": station_count,
        "station_count_definition": "distinct original PUNTO_MUESTREO PM10 sampling-point IDs",
        "station_code_count_union": station_code_count,
        "candidate_station_count_under_proposed_contract": candidate_count,
        "valid_hourly_rows_written": total_valid,
        "daily_rows_written": len(daily_rows),
        "source_schemas_identical": len({tuple(value["header"]) for value in schemas.values()}) == 1,
        "quality_flag_field_in_source": False,
        "imputation_performed": False,
        "forecasting_performed": False,
        "forecast_metrics_computed": False,
        "source_stats_by_year": {str(year): dict(source_stats[year]) for year in YEARS},
        "daily_thresholds_audited": list(DAILY_THRESHOLDS),
        "data_gate": "WARNING",
        "data_gate_reason": "MITECO supports reproducible hourly/daily extraction, but the proposed station contract and missingness/time-origin decisions require explicit review before forecasting.",
    }
    write_json(INTERIM / "phase0_observation_summary.json", observation_summary)

    schema_lines = [
        "# MITECO schema audit — P4-v2 Phase 0",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Source",
        "",
        f"Official dataset: [Datos oficiales de calidad del aire](https://catalogo.datosabiertos.miteco.gob.es/catalogo/dataset/{DATASET_ID})",
        f"Format specification: [{FORMAT_DOCUMENT}]({FORMAT_DOCUMENT})",
        "",
        "The four annual PM10 files were extracted from official MITECO ZIP packages linked by the 2020, 2021, 2022 and 2023 official-data pages. No alternative dataset was used.",
        "",
        "## Observed source schema",
        "",
        "All four PM10 files had the same 32-column header:",
        "",
        "```text",
        ";".join(EXPECTED_HEADER),
        "```",
        "",
        "| Field group | Observed fields | Interpretation supported by source |",
        "|---|---|---|",
        "| Station identity | `PROVINCIA`, `MUNICIPIO`, `ESTACION`, `PUNTO_MUESTREO` | Preserved exactly; Phase 0 uses the original `PUNTO_MUESTREO` as `station_id` and retains the other identifiers. The union contains 415 distinct PM10 sampling-point IDs and 411 distinct province–municipality–`ESTACION` codes; four station codes expose two distinct PM10 sampling points, so they are not merged. |",
        "| Pollutant | `MAGNITUD` | All extracted rows have `MAGNITUD=10`, the PM10 file supplied by MITECO. |",
        "| Calendar fields | `ANNO`, `MES`, `DIA` | Source day label; no undocumented date repair was applied. |",
        "| Hourly observations | `H01` … `H24` | Numeric hourly values; the official format document states that hourly reference is UTC and hour 1 is the period ending 01:00, through hour 24 ending 24:00. |",
        "| Quality flags | No separate quality/validation columns observed in the extracted PM10 files | Empty hourly cells are retained as missing; no per-observation quality flag is fabricated. |",
        "| Units | No unit column in the PM10 files | Unit metadata must be taken from MITECO pollutant metadata; this phase does not infer a unit from values. |",
        "| Network/geography | No network, latitude or longitude columns in these files | Province, municipality and station codes are retained; network/geographic enrichment is not inferred. |",
        "",
        "## Encoding and missingness",
        "",
        "- Delimiter observed: semicolon (`;`).",
        "- Encoding observed: UTF-8 (read with BOM tolerance).",
        "- Empty hourly cells were observed and treated as unavailable observations; no imputation, interpolation or forward filling was performed.",
        "- All rows had width 32, numeric non-empty hourly values, valid calendar dates, and `MAGNITUD=10` in this audit.",
        "- The official specification is authoritative for semantic interpretation; this audit does not infer undocumented validation semantics from the CSV alone.",
        "",
        "## Yearly observations",
        "",
        "| Year | Source rows | Source points | Date range | Valid hourly cells | Empty hourly cells |",
        "|---:|---:|---:|---|---:|---:|",
    ]
    for year in YEARS:
        stat = source_stats[year]
        schema = schemas[year]
        schema_lines.append(f"| {year} | {stat['rows']} | {stat['station_ids']} | {schema['date_min']} to {schema['date_max']} | {stat['valid_hour_cells']} | {stat['blank_hour_cells']} |")
    schema_lines += [
        "",
        "## Derived canonical fields",
        "",
        "The processed hourly Parquet table contains `station_id`, UTC timestamp, PM10 value, a null `quality_flag` because no source flag exists, `source_year`, and the original source identifiers. The daily candidate table is computed as the arithmetic mean of observed hourly values for each source day; it does not impute incomplete days.",
        "",
        "This is a Phase-0 data contract audit only. No forecast model, error metric, event metric, variance-retention metric, ranking, or eligibility rule was computed.",
        "",
    ]
    (DOCS / "miteco_schema_audit.md").write_text("\n".join(schema_lines), encoding="utf-8")

    print(json.dumps({
        "source_years": list(YEARS),
        "pm10_stations_found": station_count,
        "candidate_stations": candidate_count,
        "valid_hourly_rows": total_valid,
        "daily_rows": len(daily_rows),
        "data_gate": "WARNING",
    }, indent=2))


if __name__ == "__main__":
    main()
