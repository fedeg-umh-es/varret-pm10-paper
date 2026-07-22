# Empirical reproducibility audit

## Outcome

The empirical results previously reported in `paper_a.tex` could not be traced
to committed prediction artifacts and were not reproduced. They have been
replaced by a clean rerun whose row-level forecasts are committed under
`outputs/reproduction/`.

## GitHub recovery search

The authenticated account and both accessible organizations were searched
across 34 repositories, including branches and relevant history. The closest
upstream experiment is the private repository
[`fedeg-umh-es/P1_PM10_Meteorology_Hstar`](https://github.com/fedeg-umh-es/P1_PM10_Meteorology_Hstar)
at commit `54569072ae525974838a84950cb0e31cf83cd5a2`.

Recovered upstream artifact:

- `results/e2_met_madrid_pm10/predictions/predictions_all_models.csv`
- size: 2,629,099 bytes
- SHA-256: `e4a7edd656385df4f160176f0952a410848dc456cd5842981f191124189ea85c`
- protocol: 2019--2022 training, 2023 testing, daily origins, horizons 1--24
- models: persistence, SARIMA and direct XGBoost variants (not LightGBM)

That artifact cannot be the source of the manuscript table: it has no 48-hour
horizon and its recovered RMSE skill is small or negative at key horizons. The
processed 2019--2023 training table referenced by the upstream run is not
committed in any searched branch. No GitHub Actions artifact contained the
missing Paper A predictions.

Other inspected repositories (`P33_variance_collapse`,
`PM10-Horizons-Diagnostic`, `madrid-pm10-rank-reversal`, and
`hstar-p3-prob-operational-probabilistic-predictability`) contain adjacent
daily experiments or older EEA data, but not the claimed 2023 hourly
LightGBM/SARIMA artifacts.

## Authoritative data recovery

The actual Casa de Campo series is supplied by the Ayuntamiento de Madrid, not
EEA. The reproduction uses the official 2023 annual archive:

- dataset: <https://datos.madrid.es/dataset/201200-0-calidad-aire-horario>
- resource: `201200-3-calidad-aire-horario-zip`
- SHA-256: `b3ee481e0a787239dd07b33e93b2da97e31e6b5123d3c659f49e14549fb62b2e`
- station: 024, Casa de Campo
- magnitude: 10, PM10
- regular grid: 8,760 hours
- provider-validated observations: 8,465
- missing or invalid targets: 295

## Leakage controls in the rerun

- Five expanding rolling-origin folds: initial 50% train and five disjoint 10%
  test windows.
- Separate direct LightGBM model for every fold and horizon.
- SARIMA parameters fitted only at the start of each fold; state updated
  sequentially as each new observation becomes available.
- Input gaps use forward fill only. No backward fill, global normalization, or
  future-dependent imputation is used.
- P75 event thresholds are fitted independently on each fold's training data.
- Invalid or absent verification targets are excluded rather than imputed.
- Persistence is evaluated on exactly the same `(fold, origin, target,
  horizon)` rows as each candidate model.

The legacy `src/data/preprocess_pm10.py` remains for historical context but is
not called by the reproduction because its full-series normalization and
backward fill violate this protocol.

## Canonical rerun artifacts

- `data/raw/madrid_air_hourly_2023.zip`
- `data/processed/casa_de_campo_pm10_2023.csv`
- `outputs/reproduction/predictions_rolling_origin.parquet`
- `outputs/reproduction/metrics_rolling_origin.csv`
- `outputs/reproduction/events_p75_rolling_origin.csv`
- corresponding `holdout` artifacts and run manifests

The rerun is deterministic for LightGBM (`random_seed=42`, one thread). SARIMA
is deterministic for the fixed model specification and data sequence.
Exact Python dependency versions are pinned in `requirements-reproduction.txt`.
