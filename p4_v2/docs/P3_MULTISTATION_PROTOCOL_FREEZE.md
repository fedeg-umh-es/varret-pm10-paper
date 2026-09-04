# P3 Multistation PM10 Dynamic-Fidelity Benchmark

## Protocol freeze

`PROJECT = P3`
`TECHNICAL_NAMESPACE = p4_v2`
`CONDITION = lags_only`
`FORECASTING_RESULTS_SEEN_BEFORE_FREEZE = NO`
`MODEL_TRAINING_AUTHORIZED = NO`

`p4_v2/` is retained as a legacy technical namespace for provenance continuity. It
is now governed by P3 for this benchmark. The legacy P4 benchmark and its prediction
outputs are not inputs to this protocol.

This document freezes the forecasting protocol before any model is trained or any
forecasting result is generated. It contains no forecasting result.

## 1. Frozen source panel

The source artefacts are the Phase-0 data freeze:

| Artefact | SHA-256 |
| --- | --- |
| `p4_v2/data/processed/pm10_daily_canonical.parquet` | `1d4fbf031ecece900803303944d0d15e64e5bbe22260823ea57e3f6bcb2f8a34` |
| `p4_v2/data/processed/station_panel_frozen.csv` | `bc2a921e610dc60976f0f9609de09a7d5733e367102f6fe14a2c11361ebb0d27` |
| `p4_v2/manifests/station_panel_manifest.json` | `f05d78c062fbdf4f7a8d30b57815893ecb19d411f8d5b7a52eee6d6bbd1a3924` |
| `p4_v2/docs/P4_V2_STATION_INCLUSION_CONTRACT.md` | `813e09009131e48d7ae4a2304597c2485211583a07f19c6b66fc2f15c5ab01f0` |

The frozen panel contains 324 PM10 sampling-point series. The canonical station
identifier is the original `PUNTO_MUESTREO`; distinct measurement points are not
merged. Daily values follow the Phase-0B contract: arithmetic mean of valid hourly
values on the source UTC calendar day, usable only with at least 18 valid hours,
with no imputation.

The analysis window is the station-specific inclusive support interval recorded in
the frozen panel and comes from the 2020--2023 source years. No station may be
removed using forecast performance.

## 2. Scientific condition

`lags_only` means that each forecast uses PM10 lags and origin-known calendar
information only. No meteorological variable is used. The condition excludes
retrospective meteorology, operational meteorology, and oracle meteorology. Any
later meteorological condition must reuse the frozen panel, origin construction, and
comparable target keys.

## 3. Rolling-origin validation

Validation is an expanding, strictly chronological, blocked rolling-origin design.
Origins are daily. Model refits occur at fixed 28-calendar-day block starts to make
the protocol feasible on the target CPU environment. This is a refit schedule, not
a shuffled-fold design.

### 3.1 Station-specific origin support

For station `s`, let `[d_first,s, d_last,s]` be its inclusive frozen-panel support
interval. The candidate origin grid is every UTC calendar date in that interval. An
origin `t` is data-eligible only when all required lag values are observed at:

`{t-1, t-2, t-3, t-6, t-7, t-14, t-28}`.

The current-day PM10 value at `t` need not be observed. If it is missing, valid
historical lags still permit a forecast and persistence uses the last valid PM10
value available at or before `t`. This is causal and does not fill a missing origin
with a future value.

The first eligible origin is the earliest candidate origin satisfying all of:

1. at least 365 calendar days have elapsed since `d_first,s`;
2. all seven lag values are available;
3. for every horizon `h=1,...,7`, at least 300 complete supervised training rows
   exist with valid lags and valid `y(t+h)`, where `t+h` is strictly before the
   first fold start.

The 365-day and 300-row requirements are fixed data-support requirements. They are
not optimized using model results.

### 3.2 Folds, origins, and targets

Let `f_0,s` be the first eligible origin for station `s`. Fold starts are
`f_k,s = f_0,s + 28 k` calendar days. Fold `k` contains every eligible daily
origin `t` with `f_k,s <= t < f_k,s + 28 days`. The origin stride is one day.
Calendar blocks with no eligible origins produce no prediction rows but do not alter
the deterministic date construction of later blocks.

At fold start `f_k,s`, expanding training uses all complete supervised rows whose
target time is strictly earlier than `f_k,s`. There is no fixed-width truncation,
random split, shuffle, or look-ahead. Each model is fitted once per station/fold and
used for all daily origins in that block.

For horizon `h`, a target key exists only when `y(t+h)` is a usable canonical daily
value. A missing target horizon is omitted for every model and never imputed. A
missing required lag is omitted from common support for every model. Station
eligibility at each origin is therefore based only on frozen data availability and
the predeclared lag/target rules.

The frozen data-support audit gives 10,555 non-empty station-fold blocks, 251,316
eligible origins, and 1,709,938 origin--horizon target keys before model-output
complete-case filtering. These are protocol support counts, not forecast results.
With seven direct XGBoost fits and one SARIMA fit per block, the theoretical fit
count is 84,440. It is not a performance-based selection.

## 4. Persistence baseline

For station `s`, origin `t`, and horizon `h`:

`y_persistence(s,t,h) = last valid PM10 value with timestamp <= t`.

The lookup uses the canonical daily series only, without interpolation,
forward-filling beyond the last observed value, or access to any timestamp after the
origin. Because lag 1 is required for an eligible origin, a valid baseline exists
for every eligible origin. The persistence value is written inline to every
row-level model prediction as `y_persistence`.

## 5. Required empirical comparators

### 5.1 SARIMA

SARIMA is a required fixed empirical comparator, not a tuned benchmark winner. The
same specification applies to every station and fold:

`SARIMA(1,0,1)(1,0,0,7)`

It uses daily seasonal period 7, no exogenous variables, no order search, no
station-specific tuning, `trend="n"`, `enforce_stationarity=False`,
`enforce_invertibility=False`, and `maxiter=120`. It is fitted at each
station/fold start on the regular daily calendar from station support through the
day before the fold start. Missing daily observations remain missing; they are not
imputed. The implementation must use a missing-observation-capable state-space
SARIMA fit and record convergence/failure metadata.

Within a 28-day block, the fitted model is not refit. Its state may be updated
causally with observations becoming available through each origin by a no-refit
append/update operation; no target value is ingested before its forecast origin. If
the implementation cannot update state without refitting, the fallback is the same
fixed fold fit for every origin in that block, documented in the run manifest before
execution. No fallback may change the order, training cutoff, or target keys.

### 5.2 Primary non-SARIMA model

The primary non-SARIMA comparator is `xgboost_direct`, one model per horizon. This
is the fixed CPU-feasible model family already used in the P3 research line; it is
not a new model-family benchmark. Parameters are fixed globally and a priori:

- `n_estimators = 300`
- `max_depth = 4`
- `learning_rate = 0.05`
- `subsample = 0.9`
- `colsample_bytree = 0.9`
- `objective = reg:squarederror`
- `random_state = 42`
- `n_jobs = 1`
- `tree_method = hist`

There is no station-specific manual tuning and no hyperparameter search. Each
horizon model is fitted only on complete supervised training rows before the fold
start. XGBoost is CPU-first and does not require CUDA.

## 6. Feature contract

The exact lag features are:

- `pm10_lag_1`
- `pm10_lag_2`
- `pm10_lag_3`
- `pm10_lag_6`
- `pm10_lag_7`
- `pm10_lag_14`
- `pm10_lag_28`

The exact origin-known calendar features, computed from target date `t+h`, are:

- `target_dow_sin`
- `target_dow_cos`
- `target_doy_sin`
- `target_doy_cos`
- `target_month_sin`
- `target_month_cos`

Encodings are deterministic: weekday is `(0,...,6)/7`, day of year is
`(dayofyear-1)/365.25`, and month is `(month-1)/12`, with sine and cosine applied
to `2*pi*phase`. Target-date calendar values are known at the origin and use no
future PM10 or meteorology. There are no future rolling statistics, global
imputation, or global scaling. Any future scaling addition must be fitted within
the training split only; this freeze authorizes no such addition.

## 7. Tuning and preprocessing

Tuning is `none`. All model orders, hyperparameters, lag choices, seasonal period,
and preprocessing choices are fixed before execution. No test-origin, future-target,
station-performance, or cross-condition information can influence them. Missing
PM10 values are not imputed to increase training or evaluation coverage.

## 8. Common support

The fundamental comparison key is:

`(station_id, condition, fold, origin, target_time, horizon)`.

For each station/fold/origin/horizon, comparable models receive the same lag
features, target value, target timestamp, event threshold, and inline persistence
baseline. The target-support table is constructed before model-specific metrics.
The final paired comparison set is the exact intersection of keys with finite
`y_pred` from every required comparator. A model-specific fit failure removes that
key from the paired comparison for all models and is logged; it cannot be used to
cherry-pick a model's valid rows. `y_true` and `y_persistence` must be identical
within each retained key. This is `COMMON_SUPPORT = PASS` by construction.

## 9. Metrics

Metrics are computed first at `station x model x horizon x condition`; stations are
not pooled before station-level metrics exist.

Primary error metrics:

- `MAE = mean(abs(y_pred - y_true))`
- `RMSE = sqrt(mean((y_pred - y_true)^2))`
- `bias = mean(y_pred - y_true)`
- `Skill_MAE = 1 - MAE_model / MAE_persistence`
- `Skill_RMSE = 1 - RMSE_model / RMSE_persistence`

Dynamic-fidelity diagnostics:

- `variance_retention = Var(y_pred) / Var(y_true)`
- `std_ratio = SD(y_pred) / SD(y_true)`
- `pearson_r = Pearson correlation(y_pred, y_true)`
- `amplitude_ratio = (max(y_pred)-min(y_pred)) / (max(y_true)-min(y_true))`

Variance and standard deviation use population denominators (`ddof=0`). A zero
denominator yields `NA`, not an imputed value. These are post-evaluation diagnostics
and do not replace MAE or RMSE. Any later Skill_VP calculation remains auxiliary
and is not required by this freeze.

## 10. Event definition

The primary event threshold is training-only `p75`. For each station/fold, compute
one threshold from nonmissing PM10 daily values strictly before the fold start,
using deterministic linear quantile interpolation (`numpy.quantile`,
`method="linear"`). The same threshold is used for all horizons and models in that
station/fold. Store it as `event_threshold_training_p75`.

An event is `PM10 >= event_threshold_training_p75`. This is a train-derived
diagnostic threshold, not a regulatory, legal, health-alert, or deployment
threshold. Event diagnostics are precision, recall/POD, FAR, CSI, and event bias
(forecast-event count divided by observed-event count). Undefined denominators are
`NA`.

## 11. Ghost-skill and ranking terminology

No universal binary ghost-skill metric or variance-retention cutoff is frozen.
Candidate ghost-skill cases require all three components: positive
persistence-relative skill, material dynamic-fidelity degradation, and interpretable
scientific or operational relevance. “Material” is not converted into an
outcome-optimized universal cutoff in this phase.

The following terms remain distinct:

- **Pairwise rank reversal:** two models change preference between two declared
  metrics for the same station/horizon/condition.
- **Full ranking change:** the complete model ordering differs between declared
  metrics, with ties handled by a deterministic model-name tiebreak.
- **Eligibility disagreement:** the same cell receives different pass/fail labels
  under two explicitly declared diagnostic rule sets. This is not a synonym for
  ranking and no universal acceptance rule is frozen here.

## 12. Row-level and derived outputs

Every future prediction row must contain at least:

- `station_id`
- `condition`
- `model`
- `fold`
- `origin`
- `target_time`
- `horizon`
- `y_true`
- `y_pred`
- `y_persistence`

It should also contain `run_id`, `source_panel_hash`, `protocol_hash`, `train_end`,
`fold_start`, and `event_threshold_training_p75`. Required future artefacts are:

- `predictions_row_level.parquet`
- `metrics_by_station_horizon.parquet`
- `dynamic_fidelity_by_station_horizon.parquet`
- `event_metrics_by_station_horizon.parquet`
- `rank_reversal_table.parquet`
- `run_manifest.json`
- `figure_source_tables/`

Every figure must be generated from a versioned source table. No output listed here
is created by the present freeze task.

## 13. Computational feasibility

The frozen data-support audit implies approximately 1.71 million potential
origin--horizon rows before model-output complete-case filtering. The theoretical
84,440 fits comprise seven XGBoost direct fits and one SARIMA fit per station/fold.
Execution must process one station and one fold at a time, write row-level output
incrementally, and avoid a global feature matrix.

The intended environment is an Apple Silicon M2 with 8 GB RAM, CPU-first, one
thread per fit, and conservative external parallelism. The expected working set is
below approximately 1--2 GB when station/fold data are streamed. The dominant
bottlenecks are repeated XGBoost tree fitting and SARIMA likelihood fitting, not
the frozen Parquet input. A run manifest must record elapsed time, memory-relevant
settings, fit failures, and row counts.

## 14. Freeze boundary

This protocol freezes the `lags_only` condition, station/fold/origin construction,
baseline, SARIMA comparator, primary model, features, tuning, common support,
metrics, training-only event threshold, terminology, and output schema. It does
not freeze meteorological increments, new models, new datasets, outcome-tuned
thresholds, or model-performance conclusions. Those require a separately reviewed
protocol or condition and must reuse the frozen panel and comparable origin keys.

No forecasting results were inspected or generated before this freeze. Model
training remains unauthorized until the protocol and manifest are reviewed and
explicitly released for execution.
