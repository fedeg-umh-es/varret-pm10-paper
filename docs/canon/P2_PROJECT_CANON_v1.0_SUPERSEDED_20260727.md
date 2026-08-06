# SUPERSEDED P2 CANON NOTICE

Status: HISTORICAL / NOT A SOURCE OF TRUTH  
Superseded on: 2026-08-06  
Superseded by: `P2_PROJECT_CANON.md` v2.0

This v1.0 document is retained for provenance. Its `INCUBATION / NO-GO AS STANDALONE PAPER`, test-set best-model-envelope rule and instruction not to interrupt P3 are obsolete.

---

# P2 — Predictability Bound

Version: 1.0  
Last updated: 2026-07-27  
Status: INCUBATION / NO-GO AS STANDALONE PAPER  
Canonical file: P2_PROJECT_CANON.md

---

## 1. Source of truth

This file is the only persistent source of truth for P2.

Chat history is not canonical.

No formula, interpretation, result, or claim may be changed silently.

Any modification requires an explicit scientific decision and a recorded update to this file.

---

## 2. Canonical identity

### Name

P2 — Predictability Bound

### Central research question

> How much of the observed skill relative to persistence can be explained by the finite-memory linear autocorrelation structure of PM10?

### Scientific object

The object is the relationship between:

- persistence;
- autocorrelation;
- optimal finite-memory linear prediction;
- empirical model skill;
- and externally supplied predictive information.

### Core thesis

Persistence is not merely a baseline.

The autocorrelation structure of the series defines an interpretable linear predictability reference against which empirical model skill can be decomposed.

The purpose is to distinguish:

1. skill explained by simple persistence;
2. skill explained by richer linear memory;
3. skill potentially attributable to exogenous information or nonlinear structure.

---

## 3. Canonical terminology

Forbidden term unless mathematically justified:

```text
universal linear predictability ceiling
```

Preferred term:

```text
finite-memory optimal linear reference
```

or:

```text
AR(p)/Yule–Walker linear predictability reference
```

The reference is conditional on:

- stationarity assumptions;
- squared-error loss;
- the chosen lag order $p$;
- the selected information set;
- valid autocovariance estimation;
- the exact definition of skill.

It is not a universal upper bound for all forecasting models.

---

## 4. Correct ACF treatment

The daily series must be reindexed to a complete calendar.

Missing dates must remain as `NaN`.

Forbidden:

```python
y = y[~np.isnan(y)]
```

before lag construction.

Removing missing values compresses calendar time and changes the meaning of the lag.

Canonical procedure:

1. reindex to full daily calendar;
2. preserve missing observations;
3. compute lagged covariance using calendar-aligned valid pairs;
4. report pair counts by lag;
5. verify the covariance matrix.

Confirmed corrected lag-1 autocorrelations:

| Series | Previous biased value | Calendar-corrected value |
|---|---:|---:|
| Elche | 0.501 | 0.511 |
| Valencia Vivers | 0.592 | 0.614 |
| Zarra EMEP | 0.623 | 0.643 |

These corrected values replace the compacted-series values.

---

## 5. Rejected explanation

The negative Valencia aggregate skill was not caused by positional pairing with persistence.

The pairing was previously verified using explicit keys:

```text
dataset
fold
origin_date
horizon
date
```

with one-to-one validation and no duplicates.

The negative aggregate arose from averaging models that included a very weak `stl_ridge_direct` result.

Canonical decision:

> Do not describe the Valencia anomaly as a baseline-pairing bug.

---

## 6. Reference hierarchy

The canonical comparison should distinguish three quantities.

### AR(1) reference

A simple one-lag benchmark derived from $\phi=\rho(1)$.

It represents a minimal linear-memory reference.

### Hybrid construction

The previously used hybrid quantity combines:

- an AR(1)-based numerator;
- an empirical persistence denominator based on $\rho(h)$.

It is not automatically a rigorous linear ceiling.

Canonical status:

```text
diagnostic only
```

If it differs materially from the rigorous Yule–Walker reference, remove it from the main scientific narrative.

### Finite-memory Yule–Walker reference

Use an AR($p$) representation derived from autocovariances.

Initial daily specification:

```text
p = 14
```

Rationale:

- two weeks of daily memory;
- ability to represent weekly structure;
- manageable finite-memory reference.

This is not a universal bound.

Sensitivity to $p$ must eventually be assessed.

---

## 7. Required mathematical consistency

Before interpreting the reference, verify:

- whether skill is defined using MSE or RMSE;
- whether the formula uses the correct square root;
- whether the covariance matrix is positive semidefinite;
- eigenvalues of the Toeplitz/autocovariance matrix;
- numerical conditioning;
- any regularisation;
- pair counts for each lag;
- sensitivity to missingness;
- sensitivity to lag order.

No paper may mix:

- an MSE-derived bound;
- with an RMSE-defined empirical skill;

without the correct transformation.

---

## 8. Canonical empirical comparison

Daily series:

- Elche;
- Valencia Vivers;
- Zarra EMEP.

Horizons:

```text
h = 1,...,7
```

Main empirical models:

- `hgb_direct`;
- `ridge_direct`;
- `sarima`.

Reference-only models:

- `seasonal_naive`;
- `stl_ridge_direct`.

The best empirical model must be selected from:

```text
hgb_direct
ridge_direct
sarima
```

The weak reference models may be shown but must not determine the “best model” envelope.

---

## 9. Required definitive output

For every series and horizon:

```text
h
rho_h
AR1_reference
hybrid_reference
YW_linear_reference
best_model_skill
best_model_name
valid_pair_count
```

Required scientific questions:

1. Does the best empirical model remain below the finite-memory linear reference?
2. At which series and horizons does it exceed the AR(1) reference?
3. How far does the hybrid construction deviate from the Yule–Walker reference?
4. Does the hierarchy remain stable under reasonable values of $p$?
5. Does missing-data treatment materially alter the conclusion?

---

## 10. Novelty threshold

Weak contribution:

> We calculate an AR model and compare it with machine learning.

Potentially strong contribution:

> We construct a missingness-aware finite-memory linear reference that separates skill attributable to local autocorrelation from skill requiring additional information, and show that this distinction changes the interpretation of empirical PM10 forecasts.

The line is not publishable as a standalone paper unless it provides:

- mathematical clarity;
- non-arbitrary lag-order treatment;
- synthetic validation;
- multiple stations;
- uncertainty;
- a consequential scientific interpretation.

---

## 11. Boundaries with other lines

P2 may support P3 by estimating how much lags-only predictability is attributable to linear memory.

P2 may support P4 by distinguishing variance collapse from legitimate linear predictability.

P2 must not absorb:

- operational meteorology availability;
- Aurora evaluation;
- variance retention as the central object;
- H* cross-domain claims.

---

## 12. Current scientific state

Completed conceptually:

- identification of the compacted-calendar ACF bug;
- corrected lag-1 autocorrelations;
- rejection of the false pairing-bug narrative;
- recognition that the hybrid expression is not a rigorous linear ceiling.

Pending:

- canonical Yule–Walker implementation in `main`;
- complete h=1..7 tables;
- covariance-matrix diagnostics;
- MSE/RMSE consistency;
- lag-order sensitivity;
- synthetic validation;
- claim-level interpretation.

Current status:

```text
NO-GO AS STANDALONE PAPER
```

---

## 13. Role allocation

### ChatGPT / Claude

- mathematical interpretation;
- novelty assessment;
- claim boundaries;
- terminology control;
- analysis of whether results support a standalone paper.

### Codex

- implement the Yule–Walker calculation;
- add numerical tests;
- compute full tables;
- test lag-order sensitivity;
- store outputs and manifests.

### Claude Code

- verify implementation and outputs in the canonical repository;
- audit branch and commit state;
- confirm that the manuscript does not precede the evidence.

### Overleaf

No substantial drafting until the definitive tables exist.

---

## 14. Current priority

P2 remains in controlled incubation.

It must not interrupt the active P3 manuscript.

---

## 15. Promotion criteria

Promote P2 from incubation only if:

- the Yule–Walker reference is fully reproducible;
- mathematical assumptions are explicit;
- results are stable enough under $p$;
- at least one non-trivial interpretation emerges;
- the contribution is more than applying classical AR theory to three series.

---

## 16. Next minimum action

Run one definitive, read-only experiment producing the three complete h=1..7 tables with:

- AR(1);
- hybrid;
- finite-memory Yule–Walker;
- best empirical model.

Do not draft the paper before reviewing those tables.
