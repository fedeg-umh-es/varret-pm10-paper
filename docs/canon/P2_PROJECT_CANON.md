# P2_PROJECT_CANON

**Version:** `2.0`
**Date:** 2026-08-06
**Status:** `EXPERIMENT GO / PAPER NO-GO PENDING EMPIRICAL GATE`
**Supersedes:** P2 canon v1.x (`INCUBATION / NO-GO AS STANDALONE PAPER`)
**Authority:** decision entry `docs/decision_log/2026-08-06-p2-portfolio-realignment.md`

---

## 0. Provenance note (read first)

The v1.x canon documents that this file supersedes (`P2_PROJECT_CANON.md`,
`PM10_RESEARCH_PROGRAMME_CANON.md`) were **not versioned in this repository**.
A full history search (`git log --all --diff-filter=A`) returned no such files on
any branch or commit. They existed only outside the repository.

Consequently this file does not *edit* a prior canonical state — it **installs
the canonical state in the repository for the first time**, at v2.0, and records
the v1.x state it replaces as reported by the decision that authorised this
change. Any assertion below attributed to v1.x is carried on that authority and
is not independently verifiable from this repository.

From this version onward, `docs/canon/` is the single source of truth. Documents
outside it that describe P2 scope are **superseded nomenclature** (§9).

---

## 1. Identity

**P2 — Finite Linear Memory Skill Decomposition.**

P2 is the **only primary front** of the PM10 research programme as of v2.0.

The v1.x identity of P2 — distinguishing skill explained by persistence, by
richer linear memory, and by additional information — is **not broken** by this
version. It is made operationally precise: the "richer linear memory" term is
now a named, estimable reference (direct linear projection of order `p`,
Yule–Walker autocovariances, missingness-aware, train-only), and the
decomposition is now additive on a paired MSE scale.

---

## 2. Dominant question

> What fraction of the paired MSE reduction relative to persistence is
> reproducible by one-lag dependence and by finite linear memory of order `p`,
> and what fraction remains as model-specific residual gain, **by station and by
> horizon**?

---

## 3. Reference ladder

```
P  →  AR(1)  →  AR(p)  →  M
```

Paired losses on a common support: `L_P`, `L_AR1`, `L_ARp`, `L_M`.

```
Δ_AR1   = L_P    − L_AR1      (immediate dependence)
Δ_mem   = L_AR1  − L_ARp      (additional finite linear memory)
Δ_res   = L_ARp  − L_M        (model-specific residual gain)
Δ_total = L_P    − L_M    =   Δ_AR1 + Δ_mem + Δ_res
```

The attribution scale is **MSE**, never RMSE. A reference derived in MSE must
not be mixed with an RMSE skill score without the explicit transformation. This
requirement is carried over unchanged from v1.x.

Full mathematical specification: `docs/canon/P2_PAIRED_DECOMPOSITION_CONTRACT.md`.

---

## 4. Binding specification changes vs. v1.x

### 4.1 AR(1) is the `p = 1` case of the same direct projection

```
ŷ_{t+h}^{AR1} = μ̂ + [ γ̂(h) / γ̂(0) ] · (y_t − μ̂)
```

Under a pure population AR(1) this coincides with `φ^h`. On a real series,
`γ̂(h)/γ̂(0)` **must not** be silently replaced by `ρ̂(1)^h`. Doing so is a
different reference and requires an explicit declaration.

The historical hybrid AR(1) construction remains `DIAGNOSTIC ONLY` and may not
appear in a primary result.

### 4.2 AR(p) is a direct per-horizon linear projection

For each `h`, solve for `β_h` from the train-only Yule–Walker system. Estimating
a single one-step AR(p) and iterating it recursively is a **different reference**
and is not the canonical object. The canonical object is the direct linear
projection of `y_{t+h}` onto the last `p` values.

### 4.3 Primary results are per-model; no oracle envelope

The decomposition is produced separately for `ridge_direct`, `hgb_direct`, and
`sarima` (where comparable). `L_M` may **not** be defined by selecting, per
station and horizon, whichever model achieved the lowest loss on that same test
sample.

This **replaces** the v1.x rule that authorised selecting the best among HGB,
Ridge and SARIMA. A "best model" envelope may appear only under walk-forward or
nested-in-train selection using information strictly prior to the evaluated
origin, and must be labelled as such.

### 4.4 Paired support and `p` sensitivity

`p ∈ {7, 14, 21}`.

Per station, horizon and model, every compared method must share: `origin_date`,
`target_date`, `y_true`, forecast availability, and squared-error cases.

- **Primary support:** the global intersection across the three orders of `p`
  and across the **core method set** (§4.5).
- **Secondary sensitivity:** per-`p` paired support, always reported with its
  case count.

`p` and the verification set must never change simultaneously without both being
shown.

### 4.5 Core method set — correction adopted at v2.0

The v1.x wording "intersection across all methods" is **not implementable** on
the existing evidence and is corrected here.

Measured on the current prediction tables:

| Station | Calendar missingness | SARIMA origins |
|---|---|---|
| `e1_rr_daily` | 19.6 % | 0 of 8 667 |
| `e1_rr_valencia_vivers` | 8.3 % | 1 098 of 10 186 (10.8 %) |
| `e1_rr_zarra_emep` | 4.0 % | 1 196 of 10 188 (11.7 %) |

SARIMA was generated on a subsampled origin grid. Including it in a single
all-method intersection collapses the primary support by roughly 90 % and
annihilates it entirely for `e1_rr_daily`.

Therefore:

- **Core method set** (defines the primary support):
  `persistence`, `AR(1)`, `AR(p)` for all `p ∈ {7,14,21}`, `ridge_direct`,
  `hgb_direct`.
- **SARIMA decomposition:** produced on its own common support, reported as a
  clearly labelled secondary result with its own `n_paired`. It may never be
  compared numerically against a primary-support figure.
- `seasonal_naive` and `stl_ridge_direct` are available for `valencia_vivers`
  and `zarra_emep` only; they are optional secondary references under the same
  labelling rule.

### 4.6 Moving-block bootstrap over origin vectors

Blocks are resampled over **temporal origins**, not over individual errors
pooled across horizons. Within a replicate the joint vector

```
( ℓ_P, ℓ_AR1, ℓ_AR7, ℓ_AR14, ℓ_AR21, ℓ_Ridge, ℓ_HGB, ℓ_SARIMA )
```

is preserved for the same origin and target. The same blocks apply to every
method, so intervals are genuinely paired.

### 4.7 Proportions are secondary

```
π_linear = (L_P − L_ARp) / (L_P − L_M)
```

- **Primary result:** absolute MSE differences.
- **Secondary result:** normalised fractions, reported only where `Δ_total > 0`
  and the estimate is stable.
- Negative components are **not truncated**.
- Fractions are **not forced** into `[0, 1]`.
- `π_linear` is suppressed where `Δ_total ≈ 0`, `Δ_total < 0`, or the interval
  covers zero.

---

## 5. Missingness treatment

Non-negotiable, carried from v1.x and reinforced:

- Full daily calendar reindexed before any lag or autocovariance construction.
- Gaps preserved as `NaN`. **No imputation. No calendar compression.**
- Autocovariances estimated on the training window only, ending at the origin.
- Primary estimator: amplitude-modulated (zero-filled after demeaning), which
  guarantees a positive semi-definite `Γ_p`.
- Secondary estimator: pairwise / equal-spacing, with an explicit PSD check and
  documented fallback.

The missingness levels above (4.0 %–19.6 %) are the reason this is a
specification requirement and not an implementation detail.

---

## 6. Claims

Binding text: `docs/canon/P2_CLAIM_CONTRACT.md`. Summary:

- **Principal (allowed):** a paired rolling-origin evaluation decomposing the
  loss reduction relative to persistence into one-lag dependence, additional
  finite linear memory, and model-specific residual gain.
- **Bibliographic (allowed, scoped):** within the reviewed 20-work corpus, no
  evaluation was identified that integrates persistence, AR(1),
  AR(p)/Yule–Walker, paired rolling-origin comparison and explicit
  incomplete-calendar treatment for this attribution.
- **Empirical (pending):** the relative importance of the components varies by
  station and horizon.
- **Prohibited:** that residual gain demonstrates non-linearity or exogenous
  information.

The reference remains conditional on stationarity, loss, order `p`, available
information and autocovariance estimation. It is **not a universal bound**.

---

## 7. Status

```text
P2_EXPERIMENT_STATUS:   GO
P2_MANUSCRIPT_STATUS:   NO-GO PENDING EMPIRICAL GATE
P2_ROLE:                PRIMARY FRONT (sole)
```

No scientific execution may begin before this canon entry and the programme
canon are committed. Canonisation precedes execution.

---

## 8. `P2_PAPER_GO` gate

All conditions jointly:

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

v1.x carried these as promotion thresholds. v2.0 makes them an explicit gate.
Three stations are available, so `RESULT_REPLICATES_ACROSS_MORE_THAN_ONE_STATION`
is achievable without new data acquisition.

---

## 9. Superseded nomenclature

Documents assigning **Predictability Bound** to P1 and **Operational
Meteorology** to P2 are **superseded nomenclature**. They must not be used as a
source of truth. Under v2.0, P2 is *Finite Linear Memory Skill Decomposition*.

Where such labels appear in this repository's historical artefacts
(`outputs/p3_*`, `docs/e1_rr_post_evaluation_contract.md`, `AUDIT_SUMMARY.md`),
they are historical record and are not edited.

---

## 10. Working title and editorial promise

**Title:** *A Paired Rolling-Origin Decomposition of Persistence-Relative Skill
in Daily PM10 Forecasting*

**Promise:** beating persistence does not by itself identify additional
predictive information, because the improvement decomposes into immediate
dependence, finite linear memory, and model-specific residual gain.

**Maximum contributions:**

1. An additive decomposition of the paired MSE reduction relative to persistence.
2. A train-only AR(1)/AR(p) implementation compatible with an incomplete calendar.
3. Station–horizon evidence on how much skill finite linear memory reproduces.
4. An interpretive frame that blocks automatic attribution of the residual to
   non-linearity or exogenous covariates.
