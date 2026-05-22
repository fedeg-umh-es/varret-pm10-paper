# PIPELINE_REPORT - varret-pm10-paper

Generated: 2026-05-22

## Run Status

| Task | Status | Detail |
|---|---|---|
| Dataset replacement | OK | Replaced the synthetic placeholder with the real `pm10_clean.csv` source table |
| Raw data verification | OK | 2,350 observed daily PM10 rows, 2017-01-01 to 2024-12-31 |
| Rolling-origin predictions | OK | `outputs/metrics/predictions.csv`, 26,001 rows |
| Skill table | OK | `outputs/metrics/skill_summary.csv`, 14 rows |
| Variance-retention diagnostics | OK | `outputs/tables/variance_retention_summary.csv`, 14 rows |
| Manuscript table | OK | `outputs/tables/e1_rr_variance_retention_summary.tex` regenerated |
| Figures | OK | `figure1_reporting_gap_audit` and `figure2_skill_variance_retention` regenerated |

## Dataset

The active raw dataset is:

```text
data/raw/pm10_daily.csv
```

It was copied from the local real-data source:

```text
/Users/federicogarciacrespi/Documents/Inboxicloud/99_Por_Revisar_Manual/02_Duplicados_Carpetas/Investigacion  2/AE-pm10review/ae/pm10_clean.csv
```

Verified properties:

```text
columns: date, pm10, temp, hr, ws, wd
observed rows: 2350
date range: 2017-01-01 to 2024-12-31
complete daily span: 2922 days
missing calendar dates: 572
coverage: 80.42%
```

The E1-RR experiment remains lags-only. Meteorological columns are retained in
the raw file for provenance but are not used by the prediction generator.

## Gap Handling

The daily series is reindexed to a complete daily calendar inside
`scripts/01_generate_e1_rr_lags_only_predictions.py`. Rows with missing PM10
values, missing lagged predictors, or unavailable verification targets after
horizon alignment are excluded by listwise deletion. No interpolation, forward
fill, backward fill, or other imputation is applied.

## Rolling-Origin Run

Command:

```bash
python3 scripts/01_generate_e1_rr_lags_only_predictions.py \
  --input data/raw/pm10_daily.csv \
  --dataset e1_rr_daily \
  --start-year 2020 \
  --end-year 2024 \
  --n-jobs 7
```

Output:

```text
predictions rows: 26001
models: hgb_direct, persistence, ridge_direct
horizons: 1..7
verification date range: 2020-01-23 to 2024-12-31
minimum n per model/horizon diagnostic cell: 1215
```

The training boundary is temporal: for each origin and horizon, model fitting
uses only rows whose target date is at or before the forecast origin
(`target_date <= origin`).

## Main Diagnostic Results

| Model | Mean skill | Skill range | Mean alpha | Alpha range | Mean Skill_VP | Collapse | Inflation | Near-ideal | Low-sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hgb_direct` | 0.212 | 0.055-0.254 | 0.145 | 0.101-0.298 | 0.027 | 7/7 | 0/7 | 0/7 | 0/7 |
| `ridge_direct` | 0.242 | 0.107-0.280 | 0.072 | 0.023-0.265 | 0.013 | 7/7 | 0/7 | 0/7 | 0/7 |

Interpretation:

- Both models show positive persistence-relative RMSE skill at all horizons.
- Both models show variance collapse at all horizons under the configured
  threshold `alpha < 0.5`.
- `ridge_direct` has higher mean skill, while `hgb_direct` retains more
  variance.
- No inflation, near-ideal, or low-sample diagnostic flags are triggered.

## Manuscript Corrections Required

Replace any stale synthetic-run values with the real-data values above.

In particular, do not use:

```text
2323 rows
2017-01-01 to 2023-05-12
predictions.csv = 23016 rows
minimum n = 1096
hgb_direct mean alpha = 0.287 from the synthetic placeholder
ridge_direct mean alpha = 0.207 from the synthetic placeholder
```

Use instead:

```text
2350 observed rows
2017-01-01 to 2024-12-31
predictions.csv = 26001 rows
minimum n = 1215
hgb_direct mean skill = 0.212, mean alpha = 0.145, mean Skill_VP = 0.027
ridge_direct mean skill = 0.242, mean alpha = 0.072, mean Skill_VP = 0.013
```
