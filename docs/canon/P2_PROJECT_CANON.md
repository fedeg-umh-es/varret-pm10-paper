# P2 — Finite Linear Memory Skill Decomposition

Version: 2.0  
Last updated: 2026-08-06  
Status: PRIMARY FRONT — EXPERIMENT GO / MANUSCRIPT NO-GO PENDING EMPIRICAL GATE  
Canonical file: `P2_PROJECT_CANON.md`  
Decision ID: `2026-08-06-p2-finite-linear-memory-skill-decomposition`

---

## 1. Source of truth

This file is the persistent scientific source of truth for P2.

Chat history, historical repository names, exploratory reports and manuscript prose are not canonical unless their decisions are incorporated here.

Any change to the scientific question, decomposition, reference definitions, support rules, uncertainty design, claims or promotion gate requires an explicit decision-log entry and a versioned update to this file.

No new P2 scientific execution may begin from a contradictory or older canon.

---

## 2. Portfolio decision

```text
PORTFOLIO_REALIGNED

PRIMARY_FRONT:
P2 — Finite Linear Memory Skill Decomposition

P2_EXPERIMENT_STATUS:
GO

P2_MANUSCRIPT_STATUS:
NO-GO PENDING EMPIRICAL GATE

SECONDARY_FRONT:
NONE

P3:
HOLD — NEXT IN QUEUE

P4:
NO-GO / ARCHIVED

CROSS-DOMAIN H*:
WAIT PROTECTED
```

P2 is the only active scientific front. This decision replaces the former `INCUBATION / NO-GO AS STANDALONE PAPER` allocation and the instruction that P2 must not interrupt P3.

---

## 3. Canonical identity

### Name

P2 — Finite Linear Memory Skill Decomposition

### Dominant research question

> What part of the MSE reduction relative to persistence can be reproduced by AR(1) dependence and finite linear memory AR(p), and what part remains as model-specific residual gain, by station and forecast horizon?

### Scientific object

P2 studies attribution of persistence-relative predictive improvement, not a universal predictability ceiling.

The canonical reference ladder is:

\[
P \longrightarrow AR(1) \longrightarrow AR(p) \longrightarrow M.
\]

The scientific objective is to separate:

1. improvement associated with a one-lag linear projection;
2. additional improvement associated with finite linear memory;
3. residual improvement specific to each evaluated model under the same paired support.

The residual component must not be interpreted automatically as proof of nonlinearity, exogenous information or causal mechanism.

---

## 4. Loss scale and additive decomposition

All attribution is performed using squared error and mean squared error.

For paired losses:

\[
L_P,\quad L_{AR1},\quad L_{ARp},\quad L_M,
\]

define:

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

The required identity is:

\[
\Delta_{total}=\Delta_{AR1}+\Delta_{mem}+\Delta_{res}.
\]

The identity must be verified numerically for every station, horizon, model, lag order and bootstrap replicate where applicable.

MSE-derived quantities must not be mixed with RMSE-defined skill without an explicit and correct transformation.

---

## 5. Canonical references

### 5.1 Persistence

For origin \(t\) and horizon \(h\):

\[
\widehat y^{P}_{t+h}=y_t.
\]

Persistence is evaluated on exactly the same valid cases as every reference and model in the paired comparison.

### 5.2 AR(1) direct one-lag projection

AR(1) is the \(p=1\) case of the same direct linear projection used for AR(p):

\[
\widehat y_{t+h}^{AR1}
=
\widehat\mu+
\frac{\widehat\gamma(h)}{\widehat\gamma(0)}
(y_t-\widehat\mu).
\]

Under a strict population AR(1) process this coefficient equals \(\phi^h\). For an empirical PM10 series, \(\widehat\gamma(h)/\widehat\gamma(0)\) must not be replaced silently by \(\widehat\rho(1)^h\).

A strict parametric AR(1) forecast may be reported only as a separately labelled sensitivity analysis.

### 5.3 AR(p) direct finite-memory projection

For each horizon \(h\), with lag vector ordered as \([y_t,y_{t-1},\ldots,y_{t-p+1}]^\top\):

\[
\mathbf c_h=
[\gamma(h),\gamma(h+1),\ldots,\gamma(h+p-1)]^\top,
\]

\[
\boldsymbol\beta_h=\Gamma_p^{-1}\mathbf c_h,
\]

where \(\Gamma_p\) is the Toeplitz autocovariance matrix with elements \(\gamma(|i-j|)\).

The forecast is:

\[
\widehat y_{t+h}^{AR(p)}=
\widehat\mu+
\boldsymbol\beta_h^\top
(\mathbf x_t-\widehat\mu\mathbf 1).
\]

The canonical object is a direct horizon-specific projection. A one-step AR(p) fitted once and iterated recursively is a different reference and is not the primary P2 definition.

### 5.4 Historical hybrid construction

The former hybrid expression remains:

```text
diagnostic only
```

It must not determine the main decomposition, paper title, central tables or claims.

---

## 6. Lag-order specification

The mandatory sensitivity set is:

```text
p in {7, 14, 21}
```

No single order may be presented as universally optimal.

The principal paired analysis uses the common verification support available to all three lag orders and all compared methods. A secondary sensitivity analysis may use order-specific paired support, but must report the number and identity of valid cases.

Conclusions that change materially with \(p\) fail the stability requirement of the manuscript gate.

---

## 7. Missingness and calendar contract

Daily PM10 series must be reindexed to the complete daily calendar.

Missing dates remain `NaN`. Removing missing observations before lag construction is forbidden because it compresses calendar time and changes lag meaning.

All means, autocovariances and projection coefficients are estimated using training data only.

For every fold or origin, the implementation must:

1. preserve the full calendar index;
2. estimate \(\widehat\mu\) from observed training values only;
3. estimate each \(\widehat\gamma(k)\) from calendar-aligned observed pairs at the true lag \(k\);
4. report the valid-pair count for every lag;
5. report eigenvalues and condition number of \(\Gamma_p\);
6. never regularise or repair \(\Gamma_p\) silently;
7. record any explicit numerical stabilisation and its sensitivity.

The historical compacted-calendar calculation is invalid as a primary estimator.

---

## 8. Paired-support contract

For every station, horizon, lag order and empirical model, all compared losses must share:

```text
same origin_date
same target_date
same y_true
same forecast availability
same squared-error cases
```

The required join must be one-to-one and validated by explicit keys.

The principal support is the global intersection across:

- persistence;
- AR(1);
- AR(7);
- AR(14);
- AR(21);
- `ridge_direct`;
- `hgb_direct`;
- `sarima`, when genuinely comparable.

Support counts must be reported by station, horizon, model and lag order. Changing \(p\) and the verification sample simultaneously without disclosure is forbidden.

---

## 9. Empirical-model contract

Primary decompositions are produced separately for:

- `ridge_direct`;
- `hgb_direct`;
- `sarima`, when its origins, targets and forecast availability are comparable.

The primary result is never an oracle envelope that selects the lowest test loss at each station and horizon.

A model envelope may appear only if model selection was performed using information prior to the evaluated period through a documented walk-forward, nested or train-only selection protocol.

Historical rules authorising a test-set “best model” envelope are superseded.

Reference-only methods such as `seasonal_naive` or `stl_ridge_direct` may be shown as diagnostics but do not define \(L_M\).

---

## 10. Uncertainty contract

Uncertainty is estimated with a moving-block bootstrap over temporal origin vectors.

A resampled unit is an origin vector that preserves jointly all available method and horizon losses for that origin, including:

\[
(\ell_P,\ell_{AR1},\ell_{AR7},\ell_{AR14},\ell_{AR21},
\ell_{Ridge},\ell_{HGB},\ell_{SARIMA}).
\]

The same sampled blocks must be applied to all methods. Errors must not be resampled independently by method or pooled across horizons.

Bootstrap block length, number of replicates, seed and interval method must be declared in configuration before interpreting results. Block-length sensitivity must be reported. No silent default is canonical.

---

## 11. Primary and secondary outputs

### Primary

Absolute paired MSE differences:

- \(\Delta_{AR1}\);
- \(\Delta_{mem}\);
- \(\Delta_{res}\);
- \(\Delta_{total}\).

Negative components are retained and interpreted; they are not truncated.

### Secondary

When \(\Delta_{total}>0\) and the denominator is sufficiently separated from zero, the linear-memory fraction may be reported as:

\[
\pi_{linear}=\frac{L_P-L_{ARp}}{L_P-L_M}.
\]

Normalised fractions are suppressed or flagged when:

- \(\Delta_{total}\approx0\);
- \(\Delta_{total}<0\);
- its interval includes zero;
- numerical instability is material.

Fractions are not forced into \([0,1]\), and negative components are not hidden.

---

## 12. Claims

The detailed claim rules are stored in `P2_CLAIM_CONTRACT.md`.

Canonical summary:

### Permitted methodological claim

> We propose a paired rolling-origin evaluation that decomposes persistence-relative MSE reduction into a contribution associated with one-lag dependence, an additional contribution associated with finite linear memory, and a model-specific residual gain.

### Empirical claim pending the gate

> The relative importance of the components varies by station and forecast horizon.

This sentence is a hypothesis until the required results pass the empirical gate.

### Prohibited claim

> The residual gain demonstrates nonlinearity or exogenous predictive information.

The references are conditional on loss, lag order, information set, stationarity assumptions, missingness treatment and train-only estimation. They are not universal limits.

---

## 13. Literature positioning

Existing work supports the general principle that a stronger naive reference can reduce apparent forecast skill, the standard AR(p)/Yule–Walker mathematics, and the need to respect temporal spacing when observations are missing.

The current novelty claim is bounded:

- the internal corpus of 20 reviewed works did not identify the complete P2 combination;
- open-web checking is consistent with a plausible gap;
- this is not proof of absence from the full literature;
- a systematic Scopus/Web of Science search is required before any broad “not previously proposed” statement is used in the manuscript.

The 2026 rolling-origin PM10 preprint from this research programme is an internal antecedent and must be cited as such, not presented as unrelated external evidence.

This literature-verification task does not block implementation of the experiment, but it blocks an unbounded novelty claim.

---

## 14. Empirical scope

Initial daily PM10 scope retained from P2 v1.0:

- Elche;
- Valencia Vivers;
- Zarra EMEP;
- horizons \(h=1,\ldots,7\).

No new stations, datasets, models or domains are opened by this decision.

The paper requires replication across more than one station. Three-station evidence remains bounded and must not be generalised to PM10 forecasting as a whole.

---

## 15. PAPER_GO gate

`P2_PAPER_GO` requires all conditions below:

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

Until every item is supported by a traceable artefact:

```text
P2_MANUSCRIPT_STATUS = NO-GO PENDING EMPIRICAL GATE
```

An experiment may run under `P2_EXPERIMENT_STATUS = GO`, but manuscript drafting must not precede the evidence.

---

## 16. Required artefacts

At minimum:

```text
P2_CLAIM_CONTRACT.md
P2_PAIRED_DECOMPOSITION_CONTRACT.md
config/p2_paired_decomposition.yaml
outputs/p2_paired_decomposition/losses_paired.parquet
outputs/p2_paired_decomposition/decomposition_by_station_horizon_model_p.csv
outputs/p2_paired_decomposition/support_counts.csv
outputs/p2_paired_decomposition/yule_walker_diagnostics.csv
outputs/p2_paired_decomposition/mse_identity_checks.csv
outputs/p2_paired_decomposition/bootstrap_intervals.csv
outputs/p2_paired_decomposition/p_sensitivity.csv
outputs/p2_paired_decomposition/missingness_sensitivity.csv
outputs/p2_paired_decomposition/synthetic_validation_summary.json
outputs/p2_paired_decomposition/paper_go_gate.json
reports/P2_PAIRED_DECOMPOSITION_REPORT.md
```

All row-level model predictions used as inputs require explicit provenance, immutable hashes and a declared producer. P4 remains archived and must not be modified or silently imported as a code dependency.

---

## 17. Boundaries with other projects

- P1 remains the common rigorous-evaluation and H* methodological backbone.
- P2 does not absorb operational meteorology, Ghost Skill, variance retention, Aurora evaluation or cross-domain H*.
- P3 is held next in queue and is not modified by the P2 experiment.
- P4 is archived and read-only.
- Cross-domain H* work remains protected in WAIT.

Historical documents assigning Predictability Bound to P1 or Operational Meteorology to P2 are `SUPERSEDED NOMENCLATURE` and are not sources of truth.

---

## 18. Tool allocation

### ChatGPT / Claude

- scientific specification;
- claim discipline;
- interpretation after results;
- gate decision.

### Codex

- implement and test the paired decomposition;
- audit input provenance;
- execute only the authorised P2 experiment;
- generate traceable artefacts.

### Claude Code

- independently audit code, outputs, provenance and gate status after Codex completes.

### Overleaf

No manuscript changes are authorised before `P2_PAPER_GO`.

---

## 19. Next minimum action

Execute the single canonical Codex superprompt for the paired P2 implementation.

Do not touch the manuscript, P3, P4 or cross-domain H*.
