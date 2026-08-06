# P2 Paired Decomposition Contract

Version: 1.0  
Last updated: 2026-08-06  
Status: CANONICAL IMPLEMENTATION CONTRACT  
Parent canon: `P2_PROJECT_CANON.md` v2.0  
Decision ID: `2026-08-06-p2-finite-linear-memory-skill-decomposition`

---

## 1. Unit of evaluation

Atomic prediction key:

```text
station
fold_or_window_id
origin_date
target_date
horizon
model
```

Each atomic row must include:

```text
y_true
y_pred
squared_error
forecast_available_at_origin
source_artifact
source_sha256
producer_repository
producer_commit
```

One-to-one uniqueness is mandatory.

---

## 2. Evaluation scope

Initial scope:

```text
stations: Elche, Valencia Vivers, Zarra EMEP
resolution: daily
horizons: 1..7
p_orders: 7, 14, 21
models: ridge_direct, hgb_direct, sarima_when_comparable
baseline: persistence
```

No new stations, datasets or model families are authorised by this contract.

---

## 3. Rolling-origin and train-only rules

For every fold or origin:

```text
max(train_date) < origin_date < target_date
```

All of the following are fitted or estimated using training observations only:

- mean;
- autocovariances;
- Toeplitz matrix;
- projection coefficients;
- numerical stabilisation parameters, if any;
- model tuning or selection;
- preprocessing and imputation.

Future observations must not enter any reference calculation.

---

## 4. Calendar and missingness

1. Reindex each station to the complete daily calendar.
2. Preserve missing PM10 as `NaN`.
3. Never compress the series by dropping missing dates before lag construction.
4. Compute each autocovariance from calendar-aligned valid pairs at the actual lag.
5. Record pair count by fold, station and lag.
6. Require the origin lag vector needed by AR(p) to be available for that forecast case.
7. Compare alternative missingness treatments only as labelled sensitivity analyses.

Primary autocovariance estimator:

\[
\widehat\gamma(k)=
\frac{1}{n_k}
\sum_{t\in V_k}
(y_t-\widehat\mu)(y_{t-k}-\widehat\mu),
\]

where \(V_k\) contains training dates for which both calendar-aligned observations are present and \(n_k=|V_k|\).

The denominator, pair-count policy and centring convention must be tested and stored in configuration.

---

## 5. Reference forecasts

### Persistence

\[
\widehat y^P_{t+h}=y_t.
\]

### AR(1)

\[
\widehat y_{t+h}^{AR1}=
\widehat\mu+
\frac{\widehat\gamma(h)}{\widehat\gamma(0)}
(y_t-\widehat\mu).
\]

### AR(p)

\[
\Gamma_p[i,j]=\widehat\gamma(|i-j|),
\]

\[
\mathbf c_h=
[\widehat\gamma(h),\ldots,\widehat\gamma(h+p-1)]^\top,
\]

\[
\boldsymbol\beta_h=\operatorname{solve}(\Gamma_p,\mathbf c_h),
\]

\[
\widehat y_{t+h}^{AR(p)}=
\widehat\mu+
\boldsymbol\beta_h^\top
(\mathbf x_t-\widehat\mu\mathbf 1).
\]

Direct projection is mandatory. Recursive iteration is not the primary reference.

---

## 6. Numerical diagnostics

For every fitted \(\Gamma_p\), store:

```text
min_eigenvalue
max_eigenvalue
condition_number
rank
solver_status
regularisation_type
regularisation_value
pair_count_min
pair_count_by_lag
```

Fail closed if the system is singular or numerically invalid.

No silent pseudo-inverse, clipping, nearest-positive-definite repair or diagonal loading is allowed.

If a stabilisation policy is introduced, it must be explicit, deterministic, configured before result interpretation and accompanied by sensitivity analysis.

---

## 7. Paired support

Principal support:

```text
intersection across P, AR1, AR7, AR14, AR21,
ridge_direct, hgb_direct and comparable sarima
for the same station, origin_date, target_date, horizon and y_true
```

Secondary support:

```text
order-specific intersection for each p
```

Both analyses must report case counts. The principal result cannot change support when \(p\) changes.

Validation assertions:

- no duplicate keys;
- identical `y_true` across methods;
- identical target dates;
- identical availability status;
- no missing squared errors inside the selected support;
- no model selected using the evaluated test loss.

---

## 8. Losses and components

For every supported prediction row:

\[
\ell_j=(y_{true}-\widehat y_j)^2.
\]

For every station, horizon, model and \(p\):

\[
L_j=\operatorname{mean}(\ell_j).
\]

Components:

\[
\Delta_{AR1}=L_P-L_{AR1},
\]

\[
\Delta_{mem}=L_{AR1}-L_{ARp},
\]

\[
\Delta_{res}=L_{ARp}-L_M,
\]

\[
\Delta_{total}=L_P-L_M.
\]

Required tolerance check:

```text
abs(Delta_total - (Delta_AR1 + Delta_mem + Delta_res))
<= atol + rtol * abs(Delta_total)
```

Tolerance values must be configured and recorded.

---

## 9. Empirical models and oracle prohibition

Compute separate primary curves for:

```text
ridge_direct
hgb_direct
sarima
```

Include SARIMA only where forecast timing and paired support are genuinely comparable.

Forbidden primary construction:

```text
M(station, horizon) = argmin test MSE across models
```

A selected-model result requires walk-forward, nested or train-only selection and a recorded selection trace.

---

## 10. Moving-block bootstrap

Bootstrap unit: contiguous blocks of origin dates.

The method-loss vector for an origin is resampled jointly across methods, horizons and lag orders.

Requirements:

- common sampled blocks for all methods;
- no independent resampling by model;
- no pooling across stations;
- preserve each sampled origin vector intact;
- record block length, replicate count, seed and interval method;
- report block-length sensitivity;
- recompute MSEs, components, identity and optional fractions inside each replicate.

Until a primary block length is explicitly approved, empirical output must report the configured sensitivity set and must not hide dependence on block length.

---

## 11. Normalised fractions

Secondary fraction:

\[
\pi_{linear}=\frac{L_P-L_{ARp}}{L_P-L_M}.
\]

Report only with status metadata:

```text
DEFINED_STABLE
DEFINED_UNSTABLE
SUPPRESSED_TOTAL_NONPOSITIVE
SUPPRESSED_DENOMINATOR_NEAR_ZERO
SUPPRESSED_INTERVAL_CROSSES_ZERO
```

Do not clip to `[0,1]`.

---

## 12. Synthetic validation

The implementation must include deterministic seeded simulations covering at least:

1. white noise: no systematic persistence-relative gain;
2. AR(1): AR(1) and sufficiently rich AR(p) should be close;
3. finite AR(q), q>1: AR(p) should reproduce additional memory when p is adequate;
4. nonlinear autoregression: residual gain may appear but must not be labelled proof of nonlinearity;
5. incomplete calendar: calendar-preserving estimates must be distinguished from compressed-time estimates;
6. exact additive-identity recovery;
7. bootstrap pairing: identical sampled origins across methods.

Synthetic tests validate implementation behaviour; they do not establish the empirical PM10 claim.

---

## 13. Required tests

```text
test_complete_calendar_preserved
test_missing_dates_not_compressed
test_train_only_autocovariance
test_temporal_order
test_ar1_is_p1_direct_projection
test_arp_direct_projection_formula
test_yule_walker_diagnostics_recorded
test_no_silent_regularisation
test_unique_prediction_keys
test_identical_paired_targets
test_global_common_support_across_p
test_model_curves_are_individual
test_no_oracle_selection
test_mse_identity
test_negative_components_retained
test_fraction_suppression_rules
test_bootstrap_resamples_origin_vectors
test_bootstrap_blocks_shared_across_methods
test_synthetic_white_noise
test_synthetic_ar1
test_synthetic_arp
test_synthetic_missing_calendar
```

---

## 14. Required outputs

```text
config/p2_paired_decomposition.yaml
inputs/P2_INPUT_PROVENANCE.json
outputs/p2_paired_decomposition/losses_paired.parquet
outputs/p2_paired_decomposition/decomposition_by_station_horizon_model_p.csv
outputs/p2_paired_decomposition/support_counts.csv
outputs/p2_paired_decomposition/yule_walker_diagnostics.csv
outputs/p2_paired_decomposition/mse_identity_checks.csv
outputs/p2_paired_decomposition/bootstrap_intervals.csv
outputs/p2_paired_decomposition/bootstrap_block_sensitivity.csv
outputs/p2_paired_decomposition/p_sensitivity.csv
outputs/p2_paired_decomposition/normalised_fractions_secondary.csv
outputs/p2_paired_decomposition/missingness_sensitivity.csv
outputs/p2_paired_decomposition/synthetic_validation_summary.json
outputs/p2_paired_decomposition/paper_go_gate.json
reports/P2_PAIRED_DECOMPOSITION_REPORT.md
```

Every output must include configuration hash, code commit, generation timestamp and input provenance reference.

---

## 15. Gate evaluation

`paper_go_gate.json` contains one Boolean and one evidence path for each condition:

```text
PAIRED_SUPPORT_VALID
TRAIN_ONLY_VALID
NO_ORACLE_SELECTION
MSE_IDENTITY_VERIFIED
MISSINGNESS_TESTS_PASS
P_SENSITIVITY_COMPLETED
BLOCK_BOOTSTRAP_COMPLETED
SYNTHETIC_VALIDATION_COMPLETED
RESULT_REPLICATES_ACROSS_MORE_THAN_ONE_STATION
NON_TRIVIAL_INTERPRETATION_FOUND
```

Codex may set the first nine conditions from mechanical evidence.

`NON_TRIVIAL_INTERPRETATION_FOUND` is a scientific judgement and must remain `PENDING_SCIENTIFIC_REVIEW` unless the user explicitly decides it after inspecting results.

Therefore Codex must not declare `P2_PAPER_GO` unilaterally.
