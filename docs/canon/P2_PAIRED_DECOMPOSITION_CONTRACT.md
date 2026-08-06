# P2_PAIRED_DECOMPOSITION_CONTRACT

**Version:** `1.0`
**Date:** 2026-08-06
**Binds:** `docs/canon/P2_PROJECT_CANON.md` v2.0
**Scope:** the executable specification of the AR(1)/AR(p) references, the paired
support, the additive decomposition, uncertainty, and the falsification suite.

This contract is the specification. Implementation must conform to it; where
implementation and contract disagree, the contract is authoritative.

---

## 1. Notation

Per station, on the **full daily calendar** `t = 1 … n` (reindexed, gaps as
`NaN`, never imputed, never compressed):

- `y_t` — daily PM10, possibly missing.
- `O = { t : y_t observed }`.
- `μ̂`, `γ̂(k)` — train-only mean and autocovariances, estimated on the window
  ending at the forecast origin.

---

## 2. Missingness-aware autocovariance estimation

Estimated **only** on the training window `T = [t_start, t_origin]`. No
observation at or after the target date may enter.

`μ̂ = mean{ y_t : t ∈ T ∩ O }`

### 2.1 Primary estimator — amplitude-modulated (PSD-guaranteed)

Demean, then zero-fill:

```
z_t = y_t − μ̂     if t ∈ O
z_t = 0           otherwise
```

```
γ̂_AM(k) = (1/n_T) · Σ_{t=1}^{n_T − k} z_t · z_{t+k},     k = 0 … K
```

where `n_T = |T|` is the **full calendar length** of the training window, not
the observed count. Dividing by the full length is what makes the resulting
Toeplitz matrix positive semi-definite, so `Γ_p` is solvable. This is the
primary estimator for all headline results.

### 2.2 Secondary estimator — pairwise / equal-spacing

```
γ̂_PW(k) = (1/N_k) · Σ_{t ∈ P_k} (y_t − μ̂)(y_{t+k} − μ̂)
P_k = { t : t ∈ O and t+k ∈ O and t, t+k ∈ T },   N_k = |P_k|
```

Less biased toward zero at large `k`, but **not** guaranteed PSD. Required
handling:

- Check `λ_min(Γ̂_p)`. If `λ_min < tol`, record the failure and apply a
  documented fallback (Tikhonov ridge on the diagonal, or PSD projection).
  Fallback usage must be counted and reported, never silent.
- If `N_k = 0` for any required `k`, the origin is unavailable for that `p`.

Reported as an estimator-sensitivity result, never as a headline.

---

## 3. Reference definitions

### 3.1 AR(p) — direct per-horizon linear projection (canonical object)

For each horizon `h` and order `p`:

```
Γ_p  = [ γ̂(|i − j|) ]_{i,j = 1…p}
c_h  = [ γ̂(h), γ̂(h+1), …, γ̂(h+p−1) ]ᵀ
β_h  = Γ_p⁻¹ c_h
```

```
ŷ_{t+h}^{AR(p)} = μ̂ + Σ_{j=1}^{p} β_{h,j} · ( y_{t−j+1} − μ̂ )
```

Solve via Cholesky or Levinson–Durbin on the Toeplitz system; never by explicit
matrix inversion. Requires `γ̂(k)` for `k = 0 … h + p − 1`.

**Prohibited:** estimating a one-step AR(p) and iterating it recursively to
horizon `h`. That is a different reference and is not the canonical object.

### 3.2 AR(1) — the `p = 1` case of §3.1

```
ŷ_{t+h}^{AR(1)} = μ̂ + [ γ̂(h) / γ̂(0) ] · ( y_t − μ̂ )
```

**Prohibited:** substituting `ρ̂(1)^h` for `γ̂(h)/γ̂(0)` without an explicit
declaration that a parametric AR(1) restriction is being imposed. Under a pure
population AR(1) the two coincide; on a real series they do not.

The historical hybrid AR(1) construction is `DIAGNOSTIC ONLY` and excluded from
primary results.

### 3.3 Persistence

```
ŷ_{t+h}^{P} = y_t          (requires t ∈ O)
```

### 3.4 Availability rule

An origin `t` is available for `AR(p)` at horizon `h` iff
`y_t, y_{t−1}, …, y_{t−p+1}` are **all observed**, the training window has at
least `MIN_TRAIN_ROWS` observed values, and `y_{t+h}` is observed.

No gap in the lag window may be filled. A gap makes the origin unavailable —
that is the intended behaviour, and the resulting availability loss must be
reported per `p`.

**Optional secondary reference (`ar_p_adaptive`, diagnostic only):** projection
onto the observed subset of the lag window, solving the sub-system restricted to
observed lags. This recovers origins lost to the strict rule but changes the
reference definition per origin. It may be reported as a sensitivity result and
may never enter a primary decomposition.

---

## 4. Paired support

### 4.1 Core method set (defines the primary support)

```
persistence, AR(1), AR(7), AR(14), AR(21), ridge_direct, hgb_direct
```

### 4.2 Primary support

Per `(station, horizon)`, the set `S` of `(origin_date, target_date)` pairs for
which **every** core method has a finite prediction, with identical `y_true`.

Rationale for excluding SARIMA from the primary support: SARIMA exists at 0 of
8 667 origins for `e1_rr_daily`, 1 098 of 10 186 for `valencia_vivers`, and
1 196 of 10 188 for `zarra_emep` — it was generated on a subsampled origin grid.
A single all-method intersection would collapse the support by ~90 % and
annihilate it for `e1_rr_daily`.

### 4.3 Secondary supports

- **SARIMA:** core method set ∪ `{sarima}`, on its own intersection, own
  `n_paired`, labelled `support = "sarima_secondary"`.
- **Per-`p` sensitivity:** intersection over the core set with a single `p`,
  labelled `support = "per_p"` with its own `n_paired`.
- `seasonal_naive`, `stl_ridge_direct` (Valencia / Zarra only): optional, same
  labelling rule.

Figures and tables may **never** compare a number computed on one support against
a number computed on another without both `n_paired` values shown.

### 4.4 Support invariants (must be asserted, not assumed)

```
same origin_date
same target_date
same y_true          (exact match, not tolerance-based)
same forecast availability
same squared-error cases
```

Any violation is a hard failure — abort, do not drop rows silently.

---

## 5. Decomposition

Per `(station, horizon, model M, support)`, with `ℓ_i^m = (y_i − ŷ_i^m)²`:

```
L_m = (1/|S|) · Σ_{i ∈ S} ℓ_i^m
```

```
Δ_AR1   = L_P    − L_AR1
Δ_mem   = L_AR1  − L_ARp
Δ_res   = L_ARp  − L_M
Δ_total = L_P    − L_M
```

**Identity check (`MSE_IDENTITY_VERIFIED`):**

```
| Δ_total − (Δ_AR1 + Δ_mem + Δ_res) |  ≤  1e−9 · max(1, |Δ_total|)
```

Failure aborts the run.

**Scale:** MSE throughout. If an RMSE-scale skill is also reported it must be
derived by explicit transformation and labelled as such. Never mix an
MSE-derived reference with an RMSE skill.

### 5.1 Secondary proportion

```
π_linear = (L_P − L_ARp) / (L_P − L_M)
```

Emitted only where `Δ_total > 0` **and** the bootstrap interval for `Δ_total`
excludes zero. Otherwise `NaN` with a reason code. Negative components are not
truncated; `π_linear` is not clipped to `[0, 1]`.

---

## 6. No oracle selection

`L_M` is computed **per model**. Primary curves are individual:
`ridge_direct`, `hgb_direct`, and `sarima` on its own support.

A "best model" envelope is admissible **only** if the selection used information
strictly prior to the evaluated origin — walk-forward selection, or tuning and
selection nested inside train. Any such envelope carries `selection_rule` and
`selection_information_cutoff` columns. Selecting per station and horizon the
model with the lowest loss on the same test sample is prohibited and fails the
gate.

---

## 7. Uncertainty — moving-block bootstrap over origins

- **Resampling unit:** the `origin_date`, not the individual error.
- **Joint preservation:** within a replicate, for each resampled origin the full
  vector `(ℓ_P, ℓ_AR1, ℓ_AR7, ℓ_AR14, ℓ_AR21, ℓ_Ridge, ℓ_HGB, ℓ_SARIMA)` is kept
  together, for the same origin and target.
- **Shared blocks:** one origin resampling per station per replicate, applied to
  every method **and every horizon**, so intervals are paired and cross-horizon
  coherence is preserved.
- **Block length:** default `ℓ = ceil(n_origins^{1/3})`; sensitivity over
  `ℓ ∈ {7, 14, 30}`.
- **Replicates:** `B = 2000`. Seed recorded in the run manifest.
- **Intervals:** 95 % percentile, on every one of `Δ_AR1`, `Δ_mem`, `Δ_res`,
  `Δ_total`, and (conditionally) `π_linear`.

Pooling errors across horizons, or resampling individual errors, is prohibited —
it destroys the pairing and understates serial dependence.

---

## 8. Falsification suite (`SYNTHETIC_VALIDATION_COMPLETED`)

Synthetic series with a known data-generating process, run through the identical
pipeline. Each case states the expected outcome in advance; failure to recover it
is a pipeline defect, not a finding.

| Case | DGP | Expected |
|---|---|---|
| S1 | AR(1), `φ = 0.6` | `Δ_mem ≈ 0`, `Δ_res ≈ 0`; `π_linear ≈ 1` |
| S2 | AR(3) with non-trivial higher lags | `Δ_mem > 0`, recovered by `p ≥ 3` |
| S3 | AR(1) + weekly seasonal component | `Δ_mem > 0` concentrated where `p ≥ 7` |
| S4 | SETAR / threshold AR (non-linear) | `Δ_res > 0` for `hgb_direct`, `≈ 0` for `ridge_direct` |
| S5 | White noise | all components `≈ 0`; intervals cover zero |

**Missingness arm.** Each case is run at missingness `∈ {0 %, 5 %, 20 %}` under
both MCAR and block-wise (gap-clustered) patterns. The 20 % level is chosen to
match the observed `e1_rr_daily` rate (19.6 %). Required checks:

- `γ̂_AM` and `γ̂_PW` compared against the analytic `γ(k)` of the DGP; bias
  reported as a function of `k` and missingness rate.
- Availability loss reported per `p` and missingness pattern.
- Verification that decomposition sign and ordering survive at 20 % missingness
  in cases S1–S4.

---

## 9. Missingness tests (`MISSINGNESS_TESTS_PASS`)

Assertions, each a hard failure:

1. Full calendar reindexed before any lag or autocovariance construction.
2. No imputation anywhere in the path: gap counts before and after feature
   construction are identical.
3. No calendar compression: `origin_date` differences reflect real calendar days.
4. Train-only: for every origin, `max(train date) ≤ origin_date` and no train
   row's `target_date` exceeds `origin_date`.
5. `Γ̂_p` from the primary estimator satisfies `λ_min ≥ −tol`.
6. Availability rule applied strictly: no origin with a `NaN` in its `p`-lag
   window appears in an `AR(p)` prediction.
7. `y_true` identical across all methods for every pair in the support.

---

## 10. Required outputs

```
outputs/p2_paired_decomposition/
├── ar_reference_predictions.csv       # station, model, origin_date, horizon,
│                                      # target_date, y_true, y_pred, p,
│                                      # gamma_estimator, n_train_obs, n_train_calendar
├── paired_support_manifest.csv        # station, horizon, support, method_set,
│                                      # n_paired, n_dropped, drop_reason
├── decomposition_by_station_horizon.csv
│                                      # station, horizon, model, p, support,
│                                      # L_P, L_AR1, L_ARp, L_M,
│                                      # delta_ar1, delta_mem, delta_res, delta_total,
│                                      # pi_linear, pi_linear_reason, n_paired
├── decomposition_bootstrap_ci.csv     # + ci_low, ci_high, B, block_length, seed
├── p_sensitivity.csv                  # p ∈ {7,14,21}, primary and per_p supports
├── gamma_estimator_sensitivity.csv    # AM vs PW, psd_fallback_count
├── synthetic_validation_results.csv   # case, missingness, pattern, expected, observed, pass
├── missingness_test_report.json       # the 7 assertions of §9
├── identity_check.json                # max identity residual, pass/fail
├── run_manifest.json                  # seeds, versions, input hashes, timestamps
└── README.md                          # how to reproduce; what each file means
```

Every numeric output carries `n_paired` and `support`. A number without both is
not a result.

---

## 11. Scope boundary

This contract covers **evidence generation only**.

Prohibited within any execution under it:

- Modifying `paper_a.tex`, `paper_a_ems.tex`, or any manuscript.
- Modifying `references.bib`.
- Modifying existing P3 artefacts under `outputs/p3_*` or `audit/`.
- Writing interpretive prose that implements a claim not authorised by
  `docs/canon/P2_CLAIM_CONTRACT.md`.

The deliverable is tables, tests and diagnostics. Nothing else.
