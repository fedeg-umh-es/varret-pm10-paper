# P3 GFS Grid 004 Origin-Level Provenance Pilot

`PROJECT = P3`  
`POLLUTANT = PM10`  
`REGION = Madrid`  
`FORECASTING_EXECUTED = NO`  
`PM_PERFORMANCE_METRICS_COMPUTED = NO`  
`OVERLEAF_MODIFIED = NO`

## Pilot question

Can real archived-as-issued NCEP GFS Grid 004 forecasts be deterministically and
leakage-safely assigned to canonical Madrid PM10 P3 forecast origins?

## Frozen P3 subset

The pilot uses the active P3 row-level support, not legacy Elche or historical
PM10 files. The selected station is `28079008_10_47`, whose PM10 station code is
`28079008`. The first retained P3 origin for this station is
`2021-03-30 00:00 UTC`; the primary parquet contains both model rows for all
horizons 1--7 at that origin. The source panel identifies the station as present
through 2023-12-31.

The P3 origin convention is `00:00 UTC`. Following the existing source-selection
contract, the candidate pre-origin cycle is the previous day's `18Z` run:
`2021-03-29 18:00 UTC`. For target timestamps at `00:00 UTC`, horizons 1--7
require leads `+030, +054, +078, +102, +126, +150, +174` hours. The complete
assignment inventory is in
`p4_v2/results/p3_gfs_grid004_origin_assignment.csv`.

## Station mapping evidence

The active P3 station identifier and PM10 series are verified from the frozen
panel and primary prediction parquet. The station coordinate used only for this
provenance mapping comes from the official Madrid open station catalogue:

`https://datos.madrid.es/dataset/212629-0-estaciones-control-aire/resource/212629-0-estaciones-control-aire-csv/download/212629-0-estaciones-control-aire-csv.csv`

The retrieved record is `CODIGO=28079008`, `LATITUD=40.421553`,
`LONGITUD=-3.682316`, and advertises PM10. Its SHA-256 is recorded in the pilot
manifest. This is an external station-metadata mapping, not a replacement for
the canonical P3 PM10 observations.

The observed GRIB grid is regular latitude/longitude. A nearest-grid assignment
would therefore be deterministic for the observed artifact, but that artifact's
product identity is not admissible as Grid 004; no meteorological value was used
in P3 and no mapping was used for a forecast.

## Archived artifact recovered

The exact file retained for forensic verification is:

`p4_v2/data/raw/gfs_grid004_origin_pilot/gfs_3_20210329_1800_006.grb2`

Archive URL:

`https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-004-files/202103/20210329/gfs_3_20210329_1800_006.grb2`

The NCEI catalog entry is under the parent `model-gfs-004-files` catalogue, but
the entry title is `GFS Grid 3 2021-03-29 18:00 UTC fct:006.grb2`. Direct GRIB
inspection reports `Project = Global Forecast System 003`, `centre = kwbc`,
`dataDate = 20210329`, `dataTime = 1800`, `forecastTime = 6`, and valid time
`20210330 00:00 UTC`. The grid is `regular_ll`, `Ni=360`, `Nj=181`, with
1-degree increments. The file contains 743 GRIB messages and 102 distinct
short names. It is therefore an archived official GRIB artifact, but it is not
evidence of a Grid 004/0.5-degree file.

The NCEI day catalog for the same cycle exposes only fct:000, fct:003, and
fct:006 files. Direct HTTP checks for the required f030, f054, f078, f102,
f126, f150, and f174 paths returned 404. The retained f006 artifact is valid at
the pilot origin itself, not at any of the required h=1--7 target timestamps.

## Origin-availability assessment

Official NCEP documentation establishes four daily cycles at 00/06/12/18 UTC.
The prior P3 source-selection record uses a conservative approximate bound of
cycle time plus 3 hours 05 minutes for the relevant product stream. Under that
bound, the 2021-03-29 18Z cycle would be available before the 2021-03-30 00Z
origin. The GRIB itself does not encode dissemination latency, so this is a
bounded availability argument, not exact file-publication proof.

That bounded timing is not sufficient to pass the pilot because the required
Grid 004 identity and target leads cannot be established from the archived
artifact. Unknown or missing leads are not assigned by interpolation, carry
forward, or substitution from another cycle.

## Verdict

`TEMPORAL_ALIGNMENT = FAIL` for the required P3 horizons.  
`SPATIAL_ALIGNMENT = PARTIAL`: station coordinates and a deterministic nearest
grid rule are available for the observed artifact, but the artifact is a
1-degree Grid 003 file rather than verified Grid 004.  
`OPERATIONAL_METEOROLOGY_PROVENANCE = FAIL`.  
`P3_STATUS = PARK`.

The current P3 pilot cannot establish that real archived-as-issued NCEP GFS
Grid 004 forecasts can be assigned to canonical Madrid PM10 origins. No
forecasting, model fitting, PM metric, or scientific comparison was run.

## Official and local evidence

- NCEI GFS product description: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
- NCEI Grid-004 file-unit catalogue: https://www.ncei.noaa.gov/thredds/catalog/model-gfs-004-files/catalog.html
- NCEI pilot day catalogue: https://www.ncei.noaa.gov/thredds/catalog/model-gfs-004-files/202103/20210329/catalog.html
- NCEP GFS product inventory: https://www.nco.ncep.noaa.gov/pmb/products/gfs/
- Active P3 protocol: `p4_v2/docs/P3_MULTISTATION_PROTOCOL_FREEZE.md`
- Active P3 panel: `p4_v2/data/processed/station_panel_frozen.csv`
- Active P3 primary predictions: `p4_v2/results/predictions_row_level_lags_only_primary_support.parquet`
