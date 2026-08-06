# P2 Paired Decomposition Report

Decision ID: `2026-08-06-p2-finite-linear-memory-skill-decomposition`
Canon: `docs/canon/P2_PROJECT_CANON.md` v2.0
Contract: `docs/canon/P2_PAIRED_DECOMPOSITION_CONTRACT.md` v1.0
Generated: `2026-08-06T11:21:41Z`

---

## 1. Veredicto operativo

```text
IMPLEMENTED_AND_EXECUTED
```

The implementation is complete, the full test suite passes, and the empirical
run executed on all three authorised stations with verifiable local inputs. The
manuscript status is unchanged:

```text
P2_MANUSCRIPT_STATUS = NO-GO PENDING SCIENTIFIC REVIEW
```

Two deviations from the superprompt are recorded rather than hidden; both are
detailed in section 11 and in `docs/canon/PM10_RESEARCH_DECISION_LOG.md`.

---

## 2. Estado Git

| Field | Value |
|---|---|
| Repository | `fedeg-umh-es/varret-pm10-paper` |
| Local path | `/home/user/varret-pm10-paper` |
| Branch | `claude/p2-finite-linear-memory-gbx3mt` |
| Base SHA | `4a49b08b041c578ec5981dc1472125b2af0a4d59` |
| Working tree at start | clean (`git status --short` empty) |
| Python | 3.11.15 |

The canonical P2 repository named in the superprompt
(`.../repos/P2_Predictability_Bound`) does not exist in this execution
environment and no superprompt filesystem path is reachable. The environment
contains exactly one repository, checked out on the branch designated for this
P2 work. See section 11.

No `.tex`, `.bib`, PDF, Overleaf package or manuscript file was created,
modified or deleted. P1, P3, P4, P5 and cross-domain H* were not touched; P4 was
not reachable, read, executed or modified.

---

## 3. Inputs y procedencia

Manifest: `inputs/P2_INPUT_PROVENANCE.json`
(SHA-256 `7e61f4823860ac67fc2cf7b88c0b78031ccbd9e2a1f7bf6285099b2c126d9b06`)

| Logical role | Station | Path | SHA-256 (first 16) | Bytes | Status |
|---|---|---|---|---:|---|
| `daily_pm10_series` | Elche | `data/raw/pm10_daily.csv` | `5ab9cd34f6d6764d` | 82,827 | `VERIFIED_LOCAL_P2` |
| `row_level_predictions` | Elche | `outputs/metrics/predictions.csv` | `915a821ca74c7d51` | 2,027,461 | `VERIFIED_LOCAL_P2` |
| `daily_pm10_series` | Valencia Vivers | `data/raw/pm10_valencia_vivers.csv` | `3eabf0e518c07abc` | 42,738 | `VERIFIED_LOCAL_P2` |
| `row_level_predictions` | Valencia Vivers | `outputs/metrics/predictions_valencia_vivers.csv` | `bfb21ff09652c390` | 4,606,894 | `VERIFIED_LOCAL_P2` |
| `daily_pm10_series` | Zarra EMEP | `data/raw/pm10_zarra_emep.csv` | `f4630d0df551da20` | 43,550 | `VERIFIED_LOCAL_P2` |
| `row_level_predictions` | Zarra EMEP | `outputs/metrics/predictions_zarra_emep.csv` | `601440c364f26669` | 4,326,018 | `VERIFIED_LOCAL_P2` |

Producer repository `fedeg-umh-es/varret-pm10-paper`; producer commit
`4a49b08b041c578ec5981dc1472125b2af0a4d59`. All six hashes match the
independent traceability audit already stored in
`audit/trazabilidad_tres_estaciones.csv`, which recorded them at commit
`25f124d8` under a different branch — an independent confirmation that these
bytes are unchanged.

Prediction schema (validated, not assumed):
`dataset, model, fold, origin_date, horizon, date, y_true, y_pred`.

Pre-flight checks performed before any modelling, all passing on all three
stations:

- `date == origin_date + horizon` on every row;
- `y_true` unique per `(origin_date, horizon)` across all methods;
- `y_true` equals the raw daily series at the target date on every row (0 mismatches of 130,165);
- rows with `model == "persistence"` equal `y_t` at the origin date exactly (0 mismatches of 29,041).

### Limitations of the inputs

1. **SARIMA is not protocol-comparable.** It is absent for Elche, and for the
   other two stations it was generated with `--origin-step 14`, giving 150–175
   cases per horizon against ~1,450 for the other methods. It is therefore
   excluded from the principal support and reported on a separately labelled
   support (section 6).
2. `seasonal_naive` and `stl_ridge_direct` exist in the artefacts but are
   reference-only under canon §9 and were not used; they never define `L_M`.
3. No model was trained, no dataset was opened beyond the six files above, and
   no covariate was added.

---

## 4. Implementación

### Modules — `src/p2_decomposition/`

| Module | Responsibility |
|---|---|
| `calendar.py` | Complete daily calendar, `NaN` preserved, lag-vector availability. Gap-dropping exists only as an explicitly labelled sensitivity function. |
| `autocovariance.py` | Train-only `mu_hat` and `gamma_hat(k)` from calendar-aligned observed pairs; per-lag valid-pair counts; the invalid compressed-time estimator for sensitivity only. |
| `linear_references.py` | Persistence, direct AR(1) as the `p = 1` case, direct AR(p); Toeplitz construction, eigen-diagnostics, fail-closed solving. |
| `pairing.py` | Atomic key, one-to-one validation, support intersection, oracle-selection guard. |
| `decomposition.py` | Squared errors, four components, identity check, secondary fraction with suppression statuses. |
| `bootstrap.py` | Moving-block resampling of origin vectors with blocks shared across all methods and horizons. |
| `diagnostics.py` | Diagnostic record collection and materialisation. |
| `gate.py` | Mechanical gate; refuses to settle `NON_TRIVIAL_INTERPRETATION_FOUND`. |
| `synthetic.py` | Seeded synthetic scenarios. |
| `provenance.py` | Hashing, provenance manifest, run stamping. |

Driver: `scripts/run_p2_paired_decomposition.py`.
Configuration: `config/p2_paired_decomposition.yaml`
(SHA-256 `572a6a79dfaf6247292af8c946e3f621c520377d7eb449bd78e9b309f483fc6a`).

### Fórmulas as implemented

```text
yhat_P(t,h)   = y_t

Gamma_p[i,j]  = gamma_hat(|i-j|)
c_h           = [gamma_hat(h), ..., gamma_hat(h+p-1)]^T
beta_h        = numpy.linalg.solve(Gamma_p, c_h)
x_t           = [y_t, y_{t-1}, ..., y_{t-p+1}]^T
yhat_ARp(t,h) = mu_hat + beta_h^T (x_t - mu_hat * 1)

yhat_AR1(t,h) = the p = 1 case of the above
              = mu_hat + gamma_hat(h)/gamma_hat(0) * (y_t - mu_hat)
```

A separate system is solved for every `(origin, p, h)`. No one-step fit is
iterated. `rho_hat(1) ** h` is never substituted; a dedicated test
(`test_ar1_does_not_use_rho1_to_the_h`) asserts the two differ on a series that
is not AR(1).

### Missingness

Each station is reindexed to the complete daily calendar; missing days remain
`NaN` and are never dropped before lag construction. Missingness on the
authorised calendars: Elche 572/2,922 days (19.6%), Valencia Vivers 243/2,922
(8.3%), Zarra EMEP 117/2,921 (4.0%).

`gamma_hat(k)` uses only pairs `(t, t-k)` that are exactly `k` **calendar days**
apart and observed at both ends; `n_pairs(k)` is recorded for every
`(station, origin, lag)` in `autocovariance_pair_counts.parquet` (124,628 rows).
An AR(p) forecast is produced only when the whole `p`-day lag window at the
origin is observed — no imputation, no shortened window.

### Train-only estimation

For each origin the training window ends **one day before the origin**
(`estimation.train_end_offset_days = 1`), satisfying
`max(train_date) < origin_date < target_date`. The lag vector `x_t` evaluated at
the origin is information available at the origin, exactly like the value
persistence uses, and it is not part of the estimation sample. A test poisons
all post-origin observations with `1e6` and asserts that not one estimate moves.

### Numerical policy

`regularisation_policy: NONE`. `Gamma_p` is declared valid only when every
required autocovariance is finite, the smallest eigenvalue is strictly
positive, the matrix has full rank, and the condition number is at most `1e10`.
Otherwise the cell is refused. There is no pseudo-inverse, diagonal loading,
nearest-PD repair or clipping anywhere in the package;
`NumericsPolicy(regularisation_policy=...)` raises for any value other than
`"NONE"`.

Result: **17,796 of 17,804** fitted `Gamma_p` systems valid. The 8 refusals are
the single earliest origin of Valencia Vivers and of Zarra EMEP (one per lag
order), where the training window holds fewer than the required 365 observations
or 100 pairs at some lag, giving a non-finite `gamma`. Those origins produce no
AR forecast and are consequently absent from every paired support.

### Paired support

Atomic key `(station, fold_or_window_id, origin_date, target_date, horizon)`
plus `model`. Validated before any aggregation: one-to-one uniqueness, every
method present on every case, identical target dates, identical `y_true` within
`1e-9`, identical availability, and no missing squared error inside the support.

- **`GLOBAL_COMMON` (principal)** — intersection across
  `persistence, ar1, ar7, ar14, ar21, ridge_direct, hgb_direct`. Fixed once, so
  changing `p` never changes the verification sample.
- **`GLOBAL_COMMON_WITH_SARIMA` (secondary, labelled)** — the same intersection
  additionally requiring `sarima`, for the two stations that have it.
- **`ORDER_SPECIFIC_p{7,14,21}` (secondary, labelled)** — per-order
  intersections, deliberately using the same model set as the principal support
  so only the lag order moves.

### Bootstrap

Moving blocks of contiguous **origin dates**. One index draw per replicate is
shared by every `(method, horizon)` column, so an origin is resampled with its
whole loss vector intact. 2,000 replicates, seed `20260806`, percentile
intervals at 95%, block lengths `[7, 14, 21]`, **no primary length declared**.
Replicate matrices are never materialised: each replicate is reduced to an
origin occurrence-count vector and means come from two matrix products against
fixed value/mask matrices, processed in chunks of 250. MSEs, components,
identity and `pi_linear` are recomputed inside every replicate; the maximum
identity residual over all replicates is `2.27e-13`.

---

## 5. Tests

`python -m pytest tests/ -q` → **80 passed** (59 new P2 tests, 21 pre-existing).
No pre-existing test was modified, and none regressed.

Note on environment: `tests/test_empirical_protocol.py` and
`tests/test_reproduction_artifacts.py` fail to import until `lightgbm`,
`scikit-learn` and `statsmodels` are installed. They are unrelated to P2; after
installing the declared `requirements.txt` dependencies into the virtualenv they
pass. The count above is with all dependencies installed.

| Required test | File | Result |
|---|---|---|
| `test_complete_calendar_preserved` | `tests/test_p2_calendar.py` | PASS |
| `test_missing_dates_not_compressed` | `tests/test_p2_calendar.py` | PASS |
| `test_train_only_autocovariance` | `tests/test_p2_calendar.py` | PASS |
| `test_temporal_order` | `tests/test_p2_calendar.py` | PASS |
| `test_ar1_is_p1_direct_projection` | `tests/test_p2_linear_references.py` | PASS |
| `test_arp_direct_projection_formula` | `tests/test_p2_linear_references.py` | PASS |
| `test_yule_walker_diagnostics_recorded` | `tests/test_p2_linear_references.py` | PASS |
| `test_no_silent_regularisation` | `tests/test_p2_linear_references.py` | PASS |
| `test_unique_prediction_keys` | `tests/test_p2_pairing.py` | PASS |
| `test_identical_paired_targets` | `tests/test_p2_pairing.py` | PASS |
| `test_global_common_support_across_p` | `tests/test_p2_pairing.py` | PASS |
| `test_model_curves_are_individual` | `tests/test_p2_pairing.py` | PASS |
| `test_no_oracle_selection` | `tests/test_p2_pairing.py` | PASS |
| `test_mse_identity` | `tests/test_p2_decomposition.py` | PASS |
| `test_negative_components_retained` | `tests/test_p2_decomposition.py` | PASS |
| `test_fraction_suppression_rules` | `tests/test_p2_decomposition.py` | PASS |
| `test_bootstrap_resamples_origin_vectors` | `tests/test_p2_bootstrap.py` | PASS |
| `test_bootstrap_blocks_shared_across_methods` | `tests/test_p2_bootstrap.py` | PASS |
| `test_synthetic_white_noise` | `tests/test_p2_synthetic.py` | PASS |
| `test_synthetic_ar1` | `tests/test_p2_synthetic.py` | PASS |
| `test_synthetic_arp` | `tests/test_p2_synthetic.py` | PASS |
| `test_synthetic_missing_calendar` | `tests/test_p2_synthetic.py` | PASS |

Additional tests beyond the required list cover: rejection of duplicate and
gapped calendars, refusal to impute a lag vector across a gap, horizon-specific
(non-iterated) projections, recovery of `phi ** h` from a simulated AR(1),
ill-conditioned and non-finite matrix refusal, `NumericsPolicy` refusing any
regularisation, order-specific vs global support sizes, schema and
target-offset validation of the real artefacts, non-clipping of `pi_linear`
outside `[0, 1]`, gate completeness, the gate's refusal to settle the human
condition, bootstrap determinism under seed, pairing of `NaN` entries, and the
count-weighted replicate mean matching an explicit resample.

One regression test was added for a bug found during development: `make_ar_q`
produced an empty history slice at `t = q` (negative-stride slicing); it is now
`out[t-q:t][::-1]` and is exercised by `test_synthetic_arp`.

---

## 6. Ejecución empírica

Stations: Elche, Valencia Vivers, Zarra EMEP. Resolution: daily.
Horizons: 1–7. Lag orders: 7, 14, 21 (plus AR(1) as `p = 1`).
Models: `ridge_direct`, `hgb_direct`, and `sarima` on its own labelled support.
Baseline: persistence.

Calendars: Elche `2017-01-01..2024-12-31`, Valencia Vivers
`2017-01-01..2024-12-31`, Zarra EMEP `2017-01-01..2024-12-30`. Prediction
origins run from 2018-01-12 / 2018-02-14 / 2020-01-22 to 2024-12-30.

### Paired case counts (total over the seven horizons)

| Station | `GLOBAL_COMMON` | `WITH_SARIMA` | `ORDER_SPECIFIC_p7` | `p14` | `p21` |
|---|---:|---:|---:|---:|---:|
| Elche | 5,324 | — | 8,667 | 6,753 | 5,324 |
| Valencia Vivers | 8,097 | 567 | 10,186 | 9,037 | 8,097 |
| Zarra EMEP | 7,099 | 494 | 10,188 | 8,535 | 7,099 |

`GLOBAL_COMMON` equals `ORDER_SPECIFIC_p21` by construction: AR(21) is the
binding availability constraint, since it needs 21 consecutive observed days at
the origin. Per-horizon counts are in `support_counts.csv`.

### Cells excluded and why

| Exclusion | Count | Reason |
|---|---:|---|
| Origins with an incomplete 21-day lag window | Elche 38.6% of model cases, Valencia Vivers 20.5%, Zarra EMEP 30.3% | The AR(21) lag vector is not observed; no imputation is permitted. |
| Origins with a refused `Gamma_p` | 2 origins (8 matrices) | Training window below `min_train_observations = 365` or below `min_pairs_per_lag = 100` at some lag; fail-closed. |
| `sarima` from the principal support | all | `--origin-step 14` sampling and absence at Elche make it protocol-incomparable. Reported separately. |
| `seasonal_naive`, `stl_ridge_direct` | all | Reference-only under canon §9; they never define `L_M`. |

### Principal decomposition — `GLOBAL_COMMON`, `p = 14` (MSE, µg m⁻³ squared)

| Station | h | model | n | L_P | L_AR1 | L_ARp | L_M | Δ_AR1 | Δ_mem | Δ_res | Δ_total |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Elche | 1 | ridge_direct | 782 | 181.78 | 140.34 | 143.19 | 149.21 | 41.44 | −2.84 | −6.03 | 32.57 |
| Elche | 1 | hgb_direct | 782 | 181.78 | 140.34 | 143.19 | 169.95 | 41.44 | −2.84 | −26.76 | 11.83 |
| Elche | 7 | ridge_direct | 749 | 365.96 | 178.06 | 180.90 | 182.13 | 187.90 | −2.84 | −1.23 | 183.84 |
| Elche | 7 | hgb_direct | 749 | 365.96 | 178.06 | 180.90 | 196.73 | 187.90 | −2.84 | −15.83 | 169.23 |
| Valencia Vivers | 1 | ridge_direct | 1,180 | 91.44 | 74.49 | 75.19 | 74.79 | 16.95 | −0.71 | 0.40 | 16.65 |
| Valencia Vivers | 1 | hgb_direct | 1,180 | 91.44 | 74.49 | 75.19 | 77.96 | 16.95 | −0.71 | −2.77 | 13.47 |
| Valencia Vivers | 7 | ridge_direct | 1,138 | 188.08 | 98.80 | 96.66 | 96.45 | 89.28 | 2.14 | 0.21 | 91.63 |
| Valencia Vivers | 7 | hgb_direct | 1,138 | 188.08 | 98.80 | 96.66 | 101.79 | 89.28 | 2.14 | −5.13 | 86.29 |
| Zarra EMEP | 1 | ridge_direct | 1,019 | 80.64 | 67.21 | 66.46 | 67.17 | 13.43 | 0.75 | −0.71 | 13.47 |
| Zarra EMEP | 1 | hgb_direct | 1,019 | 80.64 | 67.21 | 66.46 | 70.92 | 13.43 | 0.75 | −4.46 | 9.72 |
| Zarra EMEP | 7 | ridge_direct | 1,011 | 240.57 | 154.72 | 151.65 | 155.58 | 85.85 | 3.07 | −3.93 | 84.99 |
| Zarra EMEP | 7 | hgb_direct | 1,011 | 240.57 | 154.72 | 151.65 | 157.10 | 85.85 | 3.07 | −5.45 | 83.47 |

The complete grid (3 stations × 7 horizons × 3 lag orders × 2 models, plus the
SARIMA and order-specific supports; 378 rows) is in
`decomposition_by_station_horizon_model_p.csv`.

---

## 7. Artefactos generados

All under `outputs/p2_paired_decomposition/`. Every row carries `code_commit`,
`config_sha256`, `input_manifest_sha256`, `generated_at_utc`, `decision_id` and
`canon_version`; station, period, resolution, horizon, `p`, model,
`support_type` and `n_cases` appear as columns wherever the table is indexed by
them. `outputs_manifest.json` repeats every hash and row count.

| Artefact | SHA-256 (first 16) | Rows × cols | Scientific function |
|---|---|---|---|
| `losses_paired.parquet` | `d912e78ce6c27af5` | 143,640 × 17 | Row-level paired squared errors on the principal support — the audit substrate for every aggregate. |
| `decomposition_by_station_horizon_model_p.csv` | `68633391434f3b17` | 378 × 22 | Primary result: the four components per cell. |
| `support_counts.csv` | `aa6d662fecc9dee6` | 98 × 13 | Case counts per station, support type and horizon; makes any support change visible. |
| `yule_walker_diagnostics.csv` | `85fff89c81161e93` | 17,804 × 21 | Eigenvalues, condition number, rank, solver status, absence of regularisation, minimum pair count per `Gamma_p`. |
| `yule_walker_solver_status.parquet` | `2194bb407f371575` | 124,376 × 12 | Per-horizon solver status for every direct projection. |
| `yule_walker_diagnostics_summary.csv` | `bad62c4664a1dee4` | 20 × 10 | Valid/refused matrix counts by station and lag order. |
| `autocovariance_pair_counts.parquet` | `78a07e443941f98c` | 124,628 × 11 | `n_pairs(k)` and `gamma_hat(k)` per station, origin and lag. |
| `mse_identity_checks.csv` | `19bbc45652b34735` | 378 × 19 | Numerical verification of the additive identity. |
| `bootstrap_intervals.csv` | `1b847bc2e7bafc54` | 3,780 × 23 | Percentile intervals for all four components and `pi_linear`, all block lengths. |
| `bootstrap_block_sensitivity.csv` | `8a873d9ef18d98cc` | 1,260 × 22 | Block-length sensitivity, including whether the sign conclusion is stable. |
| `p_sensitivity.csv` | `cab500e3eb5c20df` | 252 × 16 | Components under the fixed global support and under order-specific supports. |
| `normalised_fractions_secondary.csv` | `8c580bd075caea89` | 882 × 17 | Secondary `pi_linear` with suppression status and a four-point threshold sweep. |
| `missingness_sensitivity.csv` | `b5b89e5663577403` | 84 × 19 | Calendar-aware versus compressed-time estimation on identical support. |
| `synthetic_validation_summary.json` | `a7576f52b70fc7b0` | 7 scenarios | Seeded synthetic validation of the machinery. |
| `paper_go_gate.json` | `933917ff454b86e9` | 10 conditions | Mechanical gate. |
| `outputs_manifest.json` | `3c8ed0ac5cf29afb` | — | Hashes, row counts and descriptions of every artefact. |

Also written: `config/p2_paired_decomposition.yaml`
(`572a6a79dfaf6247`), `inputs/P2_INPUT_PROVENANCE.json` (`7e61f4823860ac67`).

No artefact from the required list is absent, so no absent-output manifest was
needed (`outputs_manifest.json:absent_artefacts` is empty). No empty CSV that
could be mistaken for a result was created.

---

## 8. Gate

| Condition | Status | Evidence |
|---|---|---|
| `PAIRED_SUPPORT_VALID` | PASS | `support_counts.csv`; assertions run before any aggregation |
| `TRAIN_ONLY_VALID` | PASS | `yule_walker_diagnostics.csv`, `autocovariance_pair_counts.parquet` |
| `NO_ORACLE_SELECTION` | PASS | `decomposition_by_station_horizon_model_p.csv`; no envelope row exists |
| `MSE_IDENTITY_VERIFIED` | PASS | `mse_identity_checks.csv` — 378/378 within `atol=1e-12`, `rtol=1e-10` |
| `MISSINGNESS_TESTS_PASS` | PASS | `missingness_sensitivity.csv` |
| `P_SENSITIVITY_COMPLETED` | PASS | `p_sensitivity.csv` |
| `BLOCK_BOOTSTRAP_COMPLETED` | PASS | `bootstrap_intervals.csv`, `bootstrap_block_sensitivity.csv` |
| `SYNTHETIC_VALIDATION_COMPLETED` | PASS | `synthetic_validation_summary.json` |
| `RESULT_REPLICATES_ACROSS_MORE_THAN_ONE_STATION` | PASS | 3 stations with complete identity-verified decompositions |
| `NON_TRIVIAL_INTERPRETATION_FOUND` | PENDING_SCIENTIFIC_REVIEW | this report |

```text
PASS 9 | FAIL 0 | BLOCKED 0 | PENDING_SCIENTIFIC_REVIEW 1
P2_PAPER_STATUS = NO-GO PENDING SCIENTIFIC REVIEW
```

`RESULT_REPLICATES_ACROSS_MORE_THAN_ONE_STATION` is recorded as a **mechanical
availability** condition: a complete, identity-verified, bootstrapped
decomposition exists for three stations. Whether the substantive pattern
replicates in a scientifically meaningful sense is part of
`NON_TRIVIAL_INTERPRETATION_FOUND` and is not settled here.

`P2_PAPER_GO` is not declared and cannot be declared by this pipeline.

---

## 9. Hechos

Verified observations only. All figures below are on the principal
`GLOBAL_COMMON` support unless stated otherwise.

1. **Component signs across the full principal grid** (126 cells = 3 stations ×
   7 horizons × 3 lag orders × 2 models):
   - `Δ_AR1 > 0` in 126/126 cells;
   - `Δ_total > 0` in 126/126 cells;
   - `Δ_mem > 0` in 64/126 and `< 0` in 62/126;
   - `Δ_res > 0` in 28/126 and `< 0` in 98/126.
2. **Magnitudes.** At `p = 14`, `Δ_AR1` ranges from 13.4 (Zarra, h=1) to 187.9
   (Elche, h=7) MSE units. Over the same cells `|Δ_mem| ≤ 4.0` and
   `|Δ_res| ≤ 26.8`, the latter maximum being `hgb_direct` at Elche h=1, where
   the model is 26.8 MSE units *worse* than AR(14).
3. **Negative components are retained, not truncated.** Every negative `Δ_mem`
   and `Δ_res` above appears verbatim in the artefacts.
4. **Secondary fraction.** On `GLOBAL_COMMON` all 126 `pi_linear` values are
   `DEFINED_STABLE` (no suppression triggered, because `Δ_total > 0` everywhere
   and the denominator clears the 1%-of-`L_P` band). Medians by station and `p`
   lie between 1.006 and 1.083, with a maximum of 3.33 (Elche, `p = 7`) and a
   minimum of 0.951 (Valencia Vivers, `p = 21`). Values above 1 are reported as
   they are; they are not clipped.
5. **Identity.** 378/378 cells satisfy the additive identity; the largest
   absolute residual across all cells is `1.4e-14`, and across all 2,000
   bootstrap replicates `2.27e-13`.
6. **Numerical validity.** 17,796/17,804 `Gamma_p` systems valid, 8 refused, 0
   regularised.
7. **Missingness sensitivity.** Substituting the compressed-time estimator on an
   identical paired support changes AR reference MSE by a relative difference in
   `[-0.0110, +0.0095]`, with medians of `-0.0043` (Elche), `-0.0039` (Valencia
   Vivers) and `+0.00004` (Zarra EMEP) — that is, the largest effects appear at
   the station with the most missing data (19.6%) and the smallest at the
   station with the least (4.0%).
8. **Lag-order sensitivity.** On the fixed global support, `Δ_mem` for
   `ridge_direct` at Elche moves from `-2.03` (`p = 7`) to `-4.02` (`p = 21`) at
   h=1; on the order-specific supports the same quantity changes sign at some
   Elche horizons (for example h=3: `+0.22` under `ORDER_SPECIFIC_p14` versus
   `-1.52` under the global support at `p = 14`). Support and lag order therefore
   both matter, which is why they are reported separately.
9. **Bootstrap block-length sensitivity.** Of 1,260 interval comparisons,
   1,151 give the same "excludes zero" conclusion under all three block lengths
   and **109 do not** (`Δ_mem` 34, `Δ_res` 27, `Δ_total` 27, `pi_linear` 12,
   `Δ_AR1` 9). Zarra EMEP contributes 65 of the 109.
10. **Bootstrap intervals are wide.** At Elche h=1, `p = 14`, block 14, the 95%
    interval for `Δ_AR1` is `[9.02, 79.28]` around a replicate mean of 37.46 —
    consistent with heavy-tailed daily PM10 squared errors.
11. **SARIMA (separate support).** On `GLOBAL_COMMON_WITH_SARIMA` at `p = 14`,
    `Δ_total` is negative at h=1 for both stations (Valencia `-1.99` on 84
    cases, Zarra `-6.20` on 70 cases) and positive at h≥2. Case counts are 70–84
    per horizon, an order of magnitude below the principal support.
12. **Synthetic validation.** White noise shows `|Δ_mem| / L_P < 0.05`; AR(1)
    shows `|Δ_mem| / L_P < 0.02`; AR(q>1) shows `Δ_mem > 0` materially at h=1;
    the calendar-aware lag-1 autocorrelation recovers the true `phi = 0.6` to
    within 0.03 while the compressed-time estimate is biased low by more than
    0.02; the identity holds to `< 1e-9` in all scenarios; bootstrap draws are
    deterministic under the seed and preserve sample size.

---

## 10. Inferencias

Separate from section 9, and bounded.

1. On these three stations and horizons, the persistence-relative MSE reduction
   is **overwhelmingly attributable to the one-lag direct linear projection**.
   `Δ_AR1` accounts for essentially all of `Δ_total`, with `Δ_mem` and `Δ_res`
   contributing small and frequently negative amounts.
2. Additional finite linear memory beyond one lag (`p = 7, 14, 21` versus
   `p = 1`) adds **little and inconsistently**: `Δ_mem` changes sign across
   stations, horizons and lag orders, and 34 of its 1,260 block-length
   comparisons are unstable.
3. `ridge_direct` and `hgb_direct` **do not systematically improve on the
   finite-memory linear reference** on this evidence: `Δ_res < 0` in 98 of 126
   cells. This is a statement about these models on this support, not about
   machine learning in general.
4. The near-unit and sometimes above-unit `pi_linear` values are a direct
   consequence of 3: when the model is worse than AR(p), the linear reference
   reproduces more than the whole observed gain. This is why the fraction must
   not be clipped to `[0, 1]`.
5. The missingness effect is small in these data but ordered with the amount of
   missingness, which is consistent with — and does not by itself prove — the
   canon's claim that compressing the calendar biases lag estimates.

**Not inferred, and explicitly not claimable from this run:** that a residual
gain proves nonlinearity or exogenous information; that AR(p) is a predictability
ceiling; that these results generalise beyond three stations; that any model is
generally superior. The synthetic nonlinear scenario is a property of a
simulated generator and is not evidence about PM10.

---

## 11. Bloqueos

Stated without minimisation.

1. **Repository identity (deviation, not resolved).** The canonical P2
   repository named in the superprompt does not exist in this environment, nor
   does the historical alternative. The work was executed inside
   `varret-pm10-paper`, which the superprompt classifies as an *external
   read-only historical producer*. The environment provisioned for this task
   contains only this repository and designates the branch
   `claude/p2-finite-linear-memory-gbx3mt` for this P2 work, so stopping with
   `BLOCKED_REPOSITORY_NOT_FOUND` would have delivered nothing. Consequence: the
   inputs are `VERIFIED_LOCAL_P2` rather than
   `VERIFIED_EXTERNAL_IMMUTABLE_INPUT`, and nothing was copied under
   `inputs/external_predictions/`. The deliverable is self-contained and can be
   moved to a dedicated P2 repository; the manifest statuses would then need
   re-issuing. **This requires a human decision.**
2. **Push and pull request (deviation).** Superprompt §20 forbids `git push` and
   pull requests. The execution environment is an ephemeral remote container
   whose contents are lost when it is reclaimed, and its operating instructions
   require pushing to the designated branch. The work was therefore pushed to
   `claude/p2-finite-linear-memory-gbx3mt` and a **draft** pull request opened.
   Nothing was merged, rebased, force-pushed or reset, and no other branch was
   touched. If this is unwanted, the branch can be deleted without affecting
   anything else.
3. **SARIMA cannot enter the principal comparison.** Its `--origin-step 14`
   sampling and its absence at Elche make it protocol-incomparable. Its results
   rest on 70–84 cases per horizon and are correspondingly weak.
4. **Elche loses 38.6% of its model cases** to the AR(21) lag-window
   requirement (19.6% missing days). The principal support is therefore
   noticeably smaller and more selected at Elche than at the other two stations,
   and that selection is not random with respect to data availability.
5. **Block-length dependence is real.** 109 of 1,260 interval comparisons change
   their "excludes zero" verdict with the block length. No length is marked
   primary, and any interpretation of a borderline cell must confront this.
6. **Bootstrap intervals are wide** relative to the component magnitudes for
   `Δ_mem` and `Δ_res`. Conclusions about the *small* components are far less
   secure than the conclusion about `Δ_AR1`.
7. **The bounded literature claim is unverified here.** No systematic
   Scopus/Web of Science search was performed; `P2_CLAIM_CONTRACT.md` §3
   requires one before any unbounded novelty formulation.
8. **Environment note.** Two pre-existing test modules require `lightgbm`,
   `scikit-learn` and `statsmodels`, which are not present in the container's
   base interpreter. They were installed into a local virtualenv for the test
   run; a fresh checkout needs `pip install -r requirements.txt`.

---

## 12. Próxima acción mínima

Review `decomposition_by_station_horizon_model_p.csv` together with
`bootstrap_block_sensitivity.csv` and decide
`NON_TRIVIAL_INTERPRETATION_FOUND` — specifically, whether "`Δ_AR1` explains
essentially the whole persistence-relative gain while `Δ_mem` and `Δ_res` are
small and often negative" is a publishable finding or a null result, given that
109 of 1,260 intervals are block-length dependent.

Do not draft any manuscript text before that decision is recorded.
