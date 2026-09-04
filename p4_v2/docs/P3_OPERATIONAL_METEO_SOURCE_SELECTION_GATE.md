# P3 Operational Meteorology Source Selection Gate

## 1. Scope and decision

`PROJECT = P3`  
`STATUS = PARK`  
`AURORA = OUT OF SCOPE`  

This is a source-selection and provenance audit only. No meteorological files were
downloaded into P3, no PM forecasts were run, and no PM performance metric was
computed.

The active question remains:

> Does meteorological information provide incremental PM10/PM2.5 predictive value
> when restricted to information genuinely available at each forecast origin?

The best candidate is the **NCEP GFS operational forecast, archived at 0.5-degree
Grid 004**. It does not pass the P3 operational provenance gate yet, because the
actual run files, complete overlap inventory, and P3-side hashes have not been
secured. P3 therefore remains parked.

## 2. Active P3 support

The active row-level P3 evidence is:

| Field | Audited value |
| --- | --- |
| Primary prediction file | `p4_v2/results/predictions_row_level_lags_only_primary_support.parquet` |
| Rows | 3,412,754 |
| Active stations | 323 |
| Station-origin pairs | 250,190 |
| Distinct UTC origin dates | 1,095 |
| Origin range | 2020-12-31 through 2023-12-30 |
| Target range | 2021-01-01 through 2023-12-31 |
| Horizons | 1..7 days |
| Active Madrid stations | 32, identified by the frozen province code `28` |
| Active Ireland stations | 0 |

`PM_ORIGINS_TOTAL` is reported as station-origin pairs because the future
meteorological join must be performed per station and origin. No candidate has a
verified origin-level meteorological file in the P3 repository, so
`PM_ORIGINS_WITH_OPERATIONAL_METEO` and its rate remain `NOT_VERIFIED`, rather than
being treated as zero or inferred from a product catalogue.

## 3. Origin-availability contract

For a P3 origin represented by a canonical daily timestamp at `00:00 UTC`, an
admissible meteorological forecast must satisfy:

```text
availability_time <= origin_time
valid_time = target_time
```

The run cycle, initialization time, issue/publication time, valid time, lead, and
the evidence used for the availability bound must be preserved per extracted field.
`initialization_time == issue_time` and `issue_time == availability_time` are not
assumed.

This document does not silently freeze a new modelling protocol. The `00:00 UTC`
alignment is the audit convention needed to test the existing date-based P3
origins; it must be confirmed when the first source files are acquired.

## 4. Candidate ranking

| Rank | Candidate | Historical classification for P3 | Provenance | Temporal alignment | Spatial alignment | Overall gate |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | NCEP GFS operational forecast, 0.5-degree Grid 004 | `ARCHIVED_AS_ISSUED` at product/archive level | `PASS` for source/cycle/lead; P3 file retention missing | `PARTIAL` | `PASS` | `PARTIAL` |
| 2 | ECMWF IFS medium-range control forecast formerly HRES | `UNKNOWN` for the P3-retained historical set | Official product schedule exists; exact historical P3 files missing | `PARTIAL` | `PASS` | `PARTIAL` |
| 3 | CAMS global atmospheric composition forecast meteorological fields | `ARCHIVED_AS_ISSUED` not established for P3-retained files | Product and forecast cycles documented; origin publication time missing | `PARTIAL` | `PASS` | `PARTIAL` |

The complete field-level inventory is in
`p4_v2/results/p3_operational_meteo_candidate_table.csv`.

## 5. Candidate 1 — NCEP GFS Grid 004

### Product and archive

NCEP GFS is an operational global forecast system. NCEI documents GFS forecast
records on the 0.5-degree Grid 004 from 2006 onward, with four cycles per day and
3-hourly forecasts through at least +192 hours. The NCEI TDS exposes historical
file-unit catalogues, and the NCEP product inventory exposes the GRIB naming needed
to identify cycle and forecast hour.

The GFS implementation is not a single immutable model version during the P3
period. The authoritative implementation log records the transition from GFS
v15.3 to v16.0 on 2021-03-22 12Z and the GFS v16.3 operational state from
2022-11-29. A future P3 extraction must pin the implementation version per run; it
must not collapse 2020--2023 into one undocumented `GFS` version.

### Availability and lead construction

NCEP documentation describes the early GFS run as beginning approximately 2 h 45 m
after the cycle time. NCEP product documentation also records that pressure GRIB
products can be delayed by up to 20 minutes. This gives a conservative operational
bound of approximately `cycle + 3 h 05 m` for the relevant product class, subject
to checking the exact archived file stream.

For the audit convention in which P3 origin dates represent `00:00 UTC`, the latest
pre-origin cycle can be the previous day's `18Z` run. For targets at `00:00 UTC` on
horizons 1..7, the corresponding leads are:

```text
h = 1..7  ->  30, 54, 78, 102, 126, 150, 174 forecast hours
```

All are within the documented 0.5-degree GFS forecast range. This is a deterministic
candidate alignment, not an executed PM experiment.

### Limitation preventing PASS

The P3 repository contains no GFS file, extraction manifest, per-run checksum, or
station-grid mapping. The audited NCEI catalogue demonstrates substantial historical
coverage, including 2020--2023 monthly/day entries, but the exact complete run-level
coverage for every P3 origin—especially the terminal 2023 period—has not been
secured locally. Therefore the candidate is primary but remains `PARTIAL`.

## 6. Candidate 2 — ECMWF IFS/HRES

ECMWF documents four forecast bases per day (`00/06/12/18 UTC`) and publishes
dissemination windows for atmospheric fields. The product provides global/sub-area
gridded atmospheric fields, with hourly steps through +90 h and coarser steps at
longer leads. It is therefore a credible meteorological source in principle.

The missing P3 evidence is the historical data package itself: exact 2020--2023
run files, per-run implementation/version, archive retrieval record, and P3-side
hashes. The official schedule is not a substitute for proof that the selected
historical stream and files were available at each PM origin. It remains a
secondary candidate and does not pass.

## 7. Candidate 3 — CAMS global forecast meteorological fields

CAMS global forecasts are atmospheric-composition forecasts produced twice daily;
the official dataset also exposes meteorological variables and states 2015--present
temporal coverage. This makes it a possible source of origin-indexed fields, but it
is not a pure NWP forcing archive: it is a chemistry/composition forecast system
with meteorological fields.

The official description does not supply a P3-usable publication/availability time
for each historical run, and the forecast is described as extending five days,
short of the frozen P3 seven-day horizon family. Annual system upgrades also mean
that per-run version identity must be retained. It cannot pass the current gate
without a narrower, explicitly approved support design; no such design is made here.

## 8. Madrid and Ireland feasibility

All three candidates have global or European coverage, so deterministic spatial
mapping to the 32 active Madrid-province stations is feasible in principle. The
active P3 panel is the frozen Spanish MITECO panel and contains no Ireland station.
Consequently:

```text
MADRID_FEASIBILITY = PASS (source-level spatial support; file-level extraction pending)
IRELAND_FEASIBILITY = FAIL (no Ireland station in the active P3 panel)
```

Ireland is not a reason to open a new region. It is simply outside the current P3
support.

## 9. Three-condition contrast

The source selection would support the intended conceptual contrast only if the
following three inputs were retained on the same station-origin-target keys:

1. `lags_only` — already executed and frozen;
2. `lags_meteo_retrospective` — not supplied by this gate;
3. `lags_meteo_operational` — not supplied by this gate.

GFS can support the operational arm in principle, and its origin/cycle/lead
metadata are sufficiently explicit to make the contrast scientifically meaningful.
However, because the retrospective arm is not selected here and GFS files are not
yet retained and joined, the current gate result is:

```text
THREE_CONDITION_CONTRAST = PARTIAL
AVAILABILITY_CAN_BE_TESTED_AS_SCIENTIFIC_FACTOR = YES (conditional on file-level provenance recovery)
```

Operational availability is not being called novel merely because it prevents
leakage. It becomes a scientific factor only if replacing retrospective fields by
the admissible GFS forecast changes the estimated meteorological increment on the
same PM support.

## 10. Hard pass assessment

The hard pass rule requires source identity, archived historical runs, cycle/init,
valid lead, availability or a conservative bound, spatial mapping, and actual P3
overlap. The current evidence satisfies the product-level parts most strongly for
GFS, but not the file-level and complete-overlap parts.

```text
SOURCE_PROVENANCE = PASS       (GFS product/archive identity)
INIT_CYCLE_PROVENANCE = PASS
VALID_LEAD_PROVENANCE = PASS
AVAILABILITY_PROVENANCE = PASS (bounded by official timing documentation)
LATENCY_PROVENANCE = BOUNDED
TEMPORAL_ALIGNMENT = PARTIAL
SPATIAL_ALIGNMENT = PASS at source level
OPERATIONAL_METEOROLOGY_PROVENANCE = PARTIAL
```

No candidate reaches `OPERATIONAL_METEOROLOGY_PROVENANCE = PASS` in the current
repository state. No experiment is authorized.

## 11. Required next action

Acquire a small, hash-pinned GFS Grid 004 pilot archive for the exact P3 origin
window and one fixed Madrid station, retaining the original GRIB metadata and a
run-level manifest. The pilot must verify file availability against the previous
18Z cycle, valid leads 30..174 h, and the fixed grid-to-station mapping before any
P3 meteorological model is trained.

This is a provenance test only. It must not calculate PM performance or alter the
frozen P3 panel, horizons, origins, or lags-only predictions.

## 12. Evidence sources

Official sources audited on 2026-08-16:

- NCEI GFS product record: <https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>
- NCEI GFS TDS catalogue: <https://www.ncei.noaa.gov/thredds/catalog/model-gfs.html>
- NCEP GFS product inventory: <https://www.nco.ncep.noaa.gov/pmb/products/gfs/>
- NCEP GFS implementation history: <https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs/implementations.php>
- NCEP GFS operational verification page: <https://www.emc.ncep.noaa.gov/users/verification/global/gfs/ops/main.php>
- ECMWF IFS Set I product and dissemination schedule: <https://www.ecmwf.int/en/forecasts/datasets/set-i>
- CAMS global atmospheric-composition forecasts: <https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview>

## 13. Governance outcome

```text
PRIMARY_OPERATIONAL_METEO_CANDIDATE = NCEP GFS operational forecast, NCEI 0.5-degree Grid 004
OPERATIONAL_METEOROLOGY_PROVENANCE = PARTIAL
P3_STATUS = PARK
```

Aurora remains parked and was not used as a candidate, predictor, or replacement
for operational meteorology.
