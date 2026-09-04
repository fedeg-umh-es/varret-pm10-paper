# P3-F1 Single-Origin GFS Operational-Provenance Pilot

`PROJECT = P3`  
`LINE = F — Forecasting & Operational Predictability`  
`POLLUTANT = PM10 ONLY`  
`CONDITION = lags_meteo_operational`  
`FORECASTING_EXECUTED = NO`  
`SCIENTIFIC_EXECUTION_PERFORMED = NO`  
`MANUSCRIPT_UPDATE_AUTHORIZED = NO`

## Audit scope and source authority

The requested `P3_PROJECT_CANON.md` was not present at the audited repository
paths. The active P3 evidence used for this provenance-only audit was the frozen
P3 protocol, frozen station panel, and canonical primary row-level support. No
legacy Elche data, Aurora product, reanalysis, retrospective meteorology, or
future observation was used.

The selected origin was not arbitrary. The earliest Madrid origin in the
canonical primary prediction support is station `28079024_10_47` at
`2020-12-31 00:00 UTC`. The primary parquet contains exactly 14 rows for that
station-origin pair: two model rows for each horizon 1--7.

## 1. Frozen PM origin

| Field | Value |
| --- | --- |
| station | `28079024_10_47` |
| station code | `28079024` |
| station name | Casa de Campo |
| origin | `2020-12-31T00:00:00Z` |
| timezone | UTC |
| horizons | 1--7 days |
| target timestamps | `2021-01-01T00:00:00Z` through `2021-01-07T00:00:00Z` |
| canonical support source | `p4_v2/results/predictions_row_level_lags_only_primary_support.parquet` |
| support verification | 14 rows, `sarima` and `xgboost_direct`, horizons 1--7 |

The station panel gives the same canonical series identifier and the support
interval `2020-01-01` to `2023-12-31`.

## 2. Candidate GFS cycles

The product documentation specifies four nominal cycles per day (`00Z`, `06Z`,
`12Z`, `18Z`). For a `00Z` PM origin, the four cycles on the immediately
preceding UTC date are the latest cycles that could be available before the
origin:

| init_time | cycle | archive day path | status for this audit |
| --- | --- | --- | --- |
| `2020-12-30T00:00:00Z` | 00Z | `202012/20201230` | catalog exposes only fct:000/003/006; required leads absent |
| `2020-12-30T06:00:00Z` | 06Z | `202012/20201230` | catalog exposes only fct:000/003/006; required leads absent |
| `2020-12-30T12:00:00Z` | 12Z | `202012/20201230` | catalog exposes only fct:000/003/006; required leads absent |
| `2020-12-30T18:00:00Z` | 18Z | `202012/20201230` | selected latest-pre-origin candidate; required leads absent |

The archive directory is official NCEI, but the day catalogue entries are titled
`GFS Grid 3`, not Grid 004. The exact 18Z day catalogue is:

`https://www.ncei.noaa.gov/thredds/catalog/model-gfs-004-files/202012/20201230/catalog.html`

No archive publication timestamp or dissemination log was found for the required
forecast files. The catalog modification timestamps are not treated as
availability timestamps.

## 3. Lead-time derivation

For the selected 18Z cycle, the leads are derived rather than assumed:

| PM horizon | PM target | init time | required lead |
| ---: | --- | --- | ---: |
| 1 | 2021-01-01 00Z | 2020-12-30 18Z | 30 h |
| 2 | 2021-01-02 00Z | 2020-12-30 18Z | 54 h |
| 3 | 2021-01-03 00Z | 2020-12-30 18Z | 78 h |
| 4 | 2021-01-04 00Z | 2020-12-30 18Z | 102 h |
| 5 | 2021-01-05 00Z | 2020-12-30 18Z | 126 h |
| 6 | 2021-01-06 00Z | 2020-12-30 18Z | 150 h |
| 7 | 2021-01-07 00Z | 2020-12-30 18Z | 174 h |

The equality checked is `GFS valid_time = PM target_time`. The required leads
would be f030 through f174. Direct requests for each corresponding NCEI file
returned HTTP 404. The catalog exposes only fct:000, fct:003, and fct:006 for
each preceding-day cycle. Therefore no required lead can be assigned.

## 4. Raw archived-file verification

No raw archived-as-issued artifact for any required lead was obtainable or
locatable. The available NCEI entries are not silently substituted: fct:006 is
not a required lead for this origin and is not a valid proxy for f030.

The requested Grid 004 identity also remains unresolved. The NCEI day entries
are titled `GFS Grid 3`; the prior inspected GRIB artifact under the same parent
catalog identified itself as `Global Forecast System 003` on a 1-degree regular
grid. It cannot be used to establish Grid 004/0.5-degree provenance.

## 5. Spatial mapping

The canonical Madrid station coordinate is taken from the official Madrid station
catalogue, record `CODIGO=28079024`:

- latitude: `40.419358`
- longitude: `-3.747345`
- PM10 advertised: yes

Source:
`https://datos.madrid.es/dataset/212629-0-estaciones-control-aire/resource/212629-0-estaciones-control-aire-csv/download/212629-0-estaciones-control-aire-csv.csv`

Because no valid Grid 004 raw file was available, no grid-cell coordinate or
station-to-grid distance is claimed. The mapping rule remains reserved as
nearest grid cell, fixed before any PM performance analysis; it was not applied
to a substitute product.

## 6. Machine-readable provenance table

The seven target rows are in:

`p4_v2/results/p3f1_single_origin_gfs_provenance.csv`

Blank raw filename/hash and `UNKNOWN` availability status denote absent evidence,
not an imputed or reconstructed file.

## 7. Gate decision

`RAW_FILE_TRACEABILITY = FAIL` — no required-lead raw file.  
`SOURCE_PROVENANCE = FAIL` — Grid 004 identity is not reconciled to an actual
archived artifact.  
`INIT_CYCLE_PROVENANCE = PASS` — candidate cycle timestamps are explicit in the
archive naming/catalogue.  
`AVAILABILITY_PROVENANCE = FAIL` — pre-origin publication of a required product
cannot be demonstrated.  
`LATENCY_PROVENANCE = FAIL` — no exact dissemination/publication timestamp.  
`VALID_LEAD_PROVENANCE = FAIL` — all required f030--f174 files are absent.  
`TEMPORAL_ALIGNMENT = FAIL` — no file satisfies the target-time mapping.  
`SPATIAL_ALIGNMENT = PARTIAL` — station coordinates are traceable, but no valid
Grid 004 artifact exists from which to select a cell.  
`LEAKAGE_AUDIT = NOT_ASSESSABLE` — no admissible operational forecast file can
be tested at the origin.

`SINGLE_ORIGIN_OPERATIONAL_PROVENANCE = FAIL`

The smallest missing piece is one traceable archived-as-issued **GFS Grid 004
0.5-degree file for a required lead (starting with f030) for the selected
2020-12-30 18Z cycle**, including a verifiable pre-origin availability timestamp
or conservative publication bound.

No forecasting or scientific execution was performed.

