# P2 — Codex Superprompt: Paired AR(p) Skill Decomposition

**Version:** `1.0` · **Date:** 2026-08-06
**Authority:** `docs/canon/P2_PAIRED_DECOMPOSITION_CONTRACT.md` v1.0
**Emitted by:** decision `2026-08-06-p2-portfolio-realignment`

Single execution unit. Evidence generation only. **Do not touch any manuscript.**

---

## PROMPT — copy everything below this line into Codex

You are implementing the P2 paired rolling-origin skill decomposition in the
repository `varret-pm10-paper`. This is evidence generation only.

### HARD PROHIBITIONS

Violating any of these fails the task regardless of code quality:

- Do **not** modify `paper_a.tex`, `paper_a_ems.tex`, `supplementary_material.tex`,
  or any `.tex` file.
- Do **not** modify `references.bib`.
- Do **not** modify anything under `outputs/p3_*`, `audit/`, `evidence/`,
  `submission_package/`, or any existing figure.
- Do **not** modify existing scripts under `scripts/` — add new ones.
- Do **not** write interpretive prose about what the results mean.
- Do **not** impute missing values, forward-fill, interpolate, or drop dates to
  compress the calendar, anywhere, for any reason.

### CONTEXT YOU MUST READ FIRST

1. `docs/canon/P2_PAIRED_DECOMPOSITION_CONTRACT.md` — the authoritative spec.
   Where this prompt and that contract disagree, the contract wins.
2. `docs/canon/P2_PROJECT_CANON.md` §4 — binding specification changes.
3. `scripts/01_generate_e1_rr_lags_only_predictions.py` — the existing
   rolling-origin generator. Your AR references must reproduce its origin
   selection, full-calendar reindexing, and train-only discipline **exactly**,
   so that predictions are pairable row-for-row.

### INPUTS

Daily series (full calendar has gaps; `date`, `pm10`):

| Station key | Raw file | Existing predictions |
|---|---|---|
| `e1_rr_daily` | `data/raw/pm10_daily.csv` | `outputs/metrics/predictions.csv` |
| `e1_rr_valencia_vivers` | `data/raw/pm10_valencia_vivers.csv` | `outputs/metrics/predictions_valencia_vivers.csv` |
| `e1_rr_zarra_emep` | `data/raw/pm10_zarra_emep.csv` | `outputs/metrics/predictions_zarra_emep.csv` |

Existing prediction schema:
`dataset, model, fold, origin_date, horizon, date, y_true, y_pred`
(`date` is the target date). Horizons `h = 1…7`.

Calendar missingness is 19.6 % / 8.3 % / 4.0 % respectively — this is the reason
the estimator must be missingness-aware, not a detail to smooth over.

### DELIVERABLE 1 — `src/models/ar_yule_walker.py`

Missingness-aware autocovariance estimation and direct linear projection.

```python
def estimate_autocovariance(y, max_lag, estimator="amplitude_modulated") -> np.ndarray
```

`y` is the full-calendar training window with `np.nan` for gaps.

- `"amplitude_modulated"` (PRIMARY): `μ̂` = mean of observed; `z_t = y_t − μ̂` if
  observed else `0`; `γ̂(k) = (1/n_T) Σ_{t=1}^{n_T−k} z_t z_{t+k}` where `n_T` is
  the **full calendar length** of the window, not the observed count. Dividing by
  the full length is what guarantees a PSD Toeplitz matrix — do not "fix" it to
  divide by `n_T − k` or by the observed count.
- `"pairwise"` (SECONDARY): `γ̂(k) = (1/N_k) Σ (y_t − μ̂)(y_{t+k} − μ̂)` over pairs
  with both endpoints observed. Return `np.nan` for any `k` with `N_k = 0`.

```python
def direct_projection_coefficients(gamma, p, h) -> np.ndarray
```

Builds `Γ_p = [γ(|i−j|)]` and `c_h = [γ(h), …, γ(h+p−1)]ᵀ`, returns
`β_h = Γ_p⁻¹ c_h`. Solve via `scipy.linalg.solve_toeplitz` or Cholesky — never
`np.linalg.inv`. Check `λ_min(Γ_p)`; if `< -1e-10`, apply Tikhonov ridge on the
diagonal, **and return a flag recording that the fallback fired**. Silent
fallback is a failure.

```python
def predict_ar_p(y_window, gamma, p, h, mu) -> float
```

`ŷ_{t+h} = μ̂ + Σ_{j=1}^{p} β_{h,j} (y_{t−j+1} − μ̂)`.

Returns `np.nan` unless `y_t … y_{t−p+1}` are **all observed**. Do not fill a gap
in the lag window — an unavailable origin is the correct, intended outcome.

**AR(1) is `p = 1` of this same function.** It must yield
`μ̂ + [γ̂(h)/γ̂(0)](y_t − μ̂)`. Do **not** implement it as `ρ̂(1)^h` — that is a
different reference. Do **not** implement AR(p) as a one-step model iterated
forward — the canonical object is the direct per-horizon projection.

### DELIVERABLE 2 — `scripts/50_generate_ar_reference_predictions.py`

For each station × horizon `h ∈ 1…7` × order `p ∈ {1, 7, 14, 21}` × estimator
`∈ {amplitude_modulated, pairwise}`:

1. Load the raw daily file, reindex to the **full daily calendar**, gaps as
   `NaN`. Never drop or compress.
2. Iterate the **same origins** the existing generator used, so results pair
   row-for-row. Reuse its `MIN_TRAIN_ROWS = 365` and origin-selection logic.
3. At each origin: training window = full calendar up to and including the
   origin. Estimate `μ̂` and `γ̂(k)` for `k = 0 … h+p−1` on that window **only**.
4. Predict, applying the strict availability rule.
5. Emit rows into `outputs/p2_paired_decomposition/ar_reference_predictions.csv`:
   `dataset, model, origin_date, horizon, target_date, y_true, y_pred, p,
   gamma_estimator, n_train_obs, n_train_calendar, psd_fallback`
   with `model ∈ {"ar1", "ar_p"}`.

Run the three stations in parallel with `joblib` as the existing generator does.

### DELIVERABLE 3 — `src/evaluation/paired_decomposition.py`

**Support construction.** Core method set:
`persistence, ar1, ar_p(7), ar_p(14), ar_p(21), ridge_direct, hgb_direct`.

Primary support per `(station, horizon)` = the set of `(origin_date,
target_date)` where every core method has a finite prediction. Assert `y_true` is
**exactly** equal across methods for each pair — mismatch aborts, never silently
drops. Emit `paired_support_manifest.csv` with `n_paired`, `n_dropped`,
`drop_reason`.

Secondary supports, each separately labelled with its own `n_paired`:
`sarima_secondary` (core ∪ sarima) and `per_p` (core with a single `p`).
SARIMA exists at 0 / 1 098 / 1 196 origins for the three stations — it is
excluded from the primary support by design, not by accident. Never compare a
number from one support against a number from another without both `n_paired`
shown.

**Decomposition.** With `ℓ_i^m = (y_i − ŷ_i^m)²` and `L_m = mean(ℓ^m)` over the
support:

```
delta_ar1   = L_P    − L_AR1
delta_mem   = L_AR1  − L_ARp
delta_res   = L_ARp  − L_M
delta_total = L_P    − L_M
```

Assert `|delta_total − (delta_ar1 + delta_mem + delta_res)| ≤ 1e-9 · max(1,
|delta_total|)`. Failure aborts the run. Write the max residual to
`identity_check.json`.

MSE throughout. Do not compute RMSE-scale skill in this module.

`pi_linear = (L_P − L_ARp) / (L_P − L_M)`, emitted **only** where
`delta_total > 0` and the bootstrap CI for `delta_total` excludes zero;
otherwise `NaN` plus a `pi_linear_reason` code. Do not truncate negative
components. Do not clip `pi_linear` to `[0, 1]`.

`L_M` is per model — `ridge_direct` and `hgb_direct` separately, `sarima` on its
own support. **Never** define `L_M` by picking the lowest-loss model per station
and horizon on the same test sample.

**Bootstrap.** Moving-block over `origin_date`, not over individual errors:

- One origin resampling per station per replicate, applied to **every method and
  every horizon** — this is what makes the intervals paired and preserves
  cross-horizon coherence.
- Within a replicate, the joint loss vector for a resampled origin stays
  together.
- Block length default `ceil(n_origins**(1/3))`; also run `ℓ ∈ {7, 14, 30}`.
- `B = 2000`, seed recorded. 95 % percentile CI on `delta_ar1`, `delta_mem`,
  `delta_res`, `delta_total`, and conditionally `pi_linear`.

### DELIVERABLE 4 — `scripts/51_build_paired_decomposition.py`

Driver producing, under `outputs/p2_paired_decomposition/`:
`paired_support_manifest.csv`, `decomposition_by_station_horizon.csv`,
`decomposition_bootstrap_ci.csv`, `p_sensitivity.csv`,
`gamma_estimator_sensitivity.csv` (AM vs pairwise, with `psd_fallback_count`),
`identity_check.json`, `run_manifest.json` (seeds, library versions, input
SHA-256, timestamps), and `README.md` describing reproduction and each file.

Every numeric row carries `n_paired` and `support`. A number without both is not
a result.

### DELIVERABLE 5 — `scripts/52_synthetic_validation.py`

Known-DGP falsification suite. Each case states its expectation in advance;
failure to recover it is a pipeline defect, not a finding.

| Case | DGP | Expected |
|---|---|---|
| S1 | AR(1), φ=0.6 | `delta_mem ≈ 0`, `delta_res ≈ 0`, `pi_linear ≈ 1` |
| S2 | AR(3), non-trivial higher lags | `delta_mem > 0`, recovered at `p ≥ 3` |
| S3 | AR(1) + weekly seasonal | `delta_mem > 0` concentrated at `p ≥ 7` |
| S4 | SETAR / threshold AR | `delta_res > 0` for HGB, `≈ 0` for Ridge |
| S5 | White noise | all components `≈ 0`, CIs cover zero |

Each case at missingness `∈ {0 %, 5 %, 20 %}` under MCAR **and** block-wise
gap-clustered patterns (20 % matches the observed `e1_rr_daily` rate). Also
report, per case: `γ̂_AM` and `γ̂_PW` bias against the analytic `γ(k)` of the DGP
as a function of `k` and missingness; and availability loss per `p` and pattern.

Output `synthetic_validation_results.csv`:
`case, dgp, missingness, pattern, quantity, expected, observed, tolerance, pass`.

### DELIVERABLE 6 — `tests/test_p2_paired_decomposition.py`

Hard-failure assertions:

1. Full calendar reindexed before any lag or autocovariance construction.
2. No imputation: gap counts identical before and after feature construction.
3. No calendar compression: `origin_date` differences are real calendar days.
4. Train-only: for every origin, `max(train date) ≤ origin_date`, and no train
   row's target date exceeds the origin.
5. `λ_min(Γ̂_p) ≥ −tol` for the amplitude-modulated estimator on real windows.
6. Strict availability: no origin with a `NaN` in its `p`-lag window appears in
   an `ar_p` prediction.
7. `y_true` identical across all methods for every pair in the support.
8. Additive identity holds to `1e-9` relative on every row.
9. **AR(1) ≠ ρ̂(1)^h**: on a real window with `h ≥ 2`, assert the implemented
   AR(1) differs from the `ρ̂(1)^h` construction — proving the correct reference
   was implemented.
10. **No oracle**: assert no output row's `L_M` equals the per-cell minimum
    across models unless a `selection_rule` column is present.

Write the §9 assertion results to `missingness_test_report.json`.

### ACCEPTANCE

The task is complete when all files in Deliverables 1–6 exist, `pytest
tests/test_p2_paired_decomposition.py` passes, `identity_check.json` reports
pass, `missingness_test_report.json` shows all seven assertions passing, and
`synthetic_validation_results.csv` shows all cases passing at every missingness
level.

Report at the end: which of these `P2_PAPER_GO` conditions your run satisfies,
with the artefact path evidencing each —

```
PAIRED_SUPPORT_VALID          TRAIN_ONLY_VALID
NO_ORACLE_SELECTION           MSE_IDENTITY_VERIFIED
MISSINGNESS_TESTS_PASS        P_SENSITIVITY_COMPLETED
BLOCK_BOOTSTRAP_COMPLETED     SYNTHETIC_VALIDATION_COMPLETED
RESULT_REPLICATES_ACROSS_MORE_THAN_ONE_STATION
NON_TRIVIAL_INTERPRETATION_FOUND
```

The last two are **assessed, not asserted** — report what the numbers show and
state plainly if replication across stations does not hold or if the
interpretation is trivial. A negative finding reported honestly is a valid
outcome; a negative finding concealed is a task failure.

Do not write the paper. Do not interpret. Produce the tables.
