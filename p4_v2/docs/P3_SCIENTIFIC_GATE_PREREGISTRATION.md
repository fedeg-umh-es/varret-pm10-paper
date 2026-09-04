# P3 Scientific Gate Preregistration

`PROJECT = P3`  
`CONDITION = lags_only`  
`PRIMARY_SUPPORT = conservative_intersection`

## 1. Phase-2C reconciliation

The historical Phase-2C preregistration was blocked because a recoverable canonical
CASE A/B/C/D taxonomy was not found in the available project documentation,
protocol code, manifests, or explicitly inspected PM10/P3 legacy repositories.

The taxonomy is therefore recorded as:

`CASE_ABCD_STATUS = NOT_RECOVERED`  
`CASE_ABCD_USED_IN_P3 = NO`

P3 does not invent, retroactively canonize, or apply CASE A/B/C/D labels. The prior
blocked state is preserved as governance history; it is not treated as a scientific
result. The unrecovered taxonomy is removed as a prerequisite for the P3 analysis.

## 2. Frozen inputs and scientific unit

Primary predictions:

`p4_v2/results/predictions_row_level_lags_only_primary_support.parquet`

SHA-256:

`3ece14b77cac0262c643355ac7689d101de4728c49d565ab2616c010fcb1a296`

Primary support is the conservative intersection of the historically documented
and executed supports. It is not retrospectively labelled the correct, true, or
original universe.

The primary scientific cell is:

`station_id × model × horizon`

Models are `xgboost_direct` and `SARIMA`; horizons are `1,...,7`; maximum cells are
`324 × 2 × 7 = 4,536`. Stations are not pooled before cell-level quantities are
computed.

## 3. Baseline governance

`PRIMARY_BASELINE = persistence`

The train-only mean artefact remains preserved at:

`p4_v2/results/train_only_mean_baseline_by_fold.parquet`

It is classified as:

`TRAIN_ONLY_MEAN_BASELINE_STATUS = DERIVED_NOT_ACTIVE`

It is a deterministic, training-only reference derived from nonmissing canonical
daily PM10 values through `train_end = fold_start - 1 calendar day`. It does not
replace persistence and is not used by the primary error-skill layer. It may only
be activated by a later, separately justified scientific question.

## 4. P3 analysis architecture

P3 proceeds through five explicit layers:

`LAYER_1 = ERROR_SKILL`  
`LAYER_2 = DYNAMIC_FIDELITY`  
`LAYER_3 = EVENT_REPRESENTATION`  
`LAYER_4 = RANK_DISCORDANCE`  
`LAYER_5 = GHOST_SKILL_ADJUDICATION`

No A/B/C/D labels are used.

## 5. Layer 1 — error skill

For each station-model-horizon cell, later calculations use:

`MAE = mean(abs(y_pred - y_true))`  
`RMSE = sqrt(mean((y_pred - y_true)^2))`  
`bias = mean(y_pred - y_true)`

`Skill_MAE = 1 - MAE_model / MAE_persistence`  
`Skill_RMSE = 1 - RMSE_model / RMSE_persistence`

Positive error skill means strictly `Skill > 0`. A positive point estimate alone
is not interpreted as statistically established superiority.

## 6. Paired inference

Comparisons use origin-level paired loss differences on conservative common support.

For absolute error:

`d_t = |y_true - y_pred_model| - |y_true - y_persistence|`

For squared error:

`d_t = (y_true - y_pred_model)^2 - (y_true - y_persistence)^2`

The primary dependence-aware procedure is a moving block bootstrap with:

`BOOTSTRAP_TYPE = moving block bootstrap`  
`PRIMARY_BLOCK_LENGTH = 28 calendar days`  
`BOOTSTRAP_RESAMPLES = 10,000`  
`RANDOM_SEED = 20260816`  
`CONFIDENCE_LEVEL = 0.95`

The 28-day length is fixed before global execution because the design has daily
origins, horizons up to seven days, serial dependence in daily PM10 errors, and
28-day refit blocks. It is not selected from resulting significance. Sensitivity
lengths remain 7 and 56 days. Bootstrap resampling is applied to origin-level
paired differences, never to already-aggregated MAE/RMSE values.

## 7. Multiplicity governance

`MULTIPLICITY_FAMILY = GLOBAL_ALL_CELLS`

The primary hypothesis family is the 4,536 station-model-horizon cells. The primary
correction is:

`MULTIPLICITY_CORRECTION = Benjamini-Hochberg FDR q = 0.05`

`MULTIPLICITY_FAMILY_STATUS = FROZEN`

The family defines the set of cell-level paired hypotheses; it does not assert that
the cells are independent observations. Temporal and cross-cell dependence remain
part of interpretation and are not converted into additional independent sample
counts. Station-level prevalence is descriptive and is not substituted for the
cell-level hypothesis family. This family was fixed without inspecting discovery
counts.

## 8. Layer 2 — dynamic fidelity

The following are continuous diagnostics; no universal binary fidelity cutoff is
imposed:

`variance_retention = Var(y_pred) / Var(y_true)`  
`std_ratio = SD(y_pred) / SD(y_true)`  
`correlation = Pearson(y_pred, y_true)`

Amplitude is defined by the within-cell 5th-to-95th percentile range:

`amplitude_ratio = [Q95(y_pred)-Q05(y_pred)] / [Q95(y_true)-Q05(y_true)]`

Variance and standard deviation use population denominators (`ddof=0`). A zero
denominator is `NA`. These diagnostics complement and do not replace MAE or RMSE.
`Skill_VP`, if retained, is auxiliary only. No additional temporal-variability
metric is introduced by this reconciliation.

## 9. Layer 3 — event representation

The existing fold-specific `event_threshold_p75` is derived from training-only
nonmissing daily PM10 values. The primary event indicator is:

`event = 1[y_true > event_threshold_p75]`

Later event diagnostics are precision, recall/POD, FAR, CSI, and event bias. The
threshold is a **train-derived high-concentration threshold**, not a regulatory,
legal, health-alert, or deployment threshold.

## 10. Layer 4 — rank discordance

The following concepts remain separate:

- **PAIRWISE_RANK_REVERSAL:** XGBoost and SARIMA have opposite ordering under
  error skill and a declared fidelity/event metric for the same station and
  horizon.
- **ELIGIBILITY_DISAGREEMENT:** a forecast satisfies an explicit error-skill
  criterion but fails an explicit fidelity/event criterion.
- **FULL_RANKING_CHANGE:** not a primary concept with only two forecast models;
  with two models it reduces to the pairwise preference comparison.

Kendall tau is not used as a two-model ranking statistic.

## 11. Layer 5 — ghost-skill adjudication

A ghost-skill diagnosis requires all three components:

1. positive error-based skill relative to persistence;
2. material degradation of dynamic fidelity;
3. that degradation changes a scientific or operational interpretation.

This is not defined as smoothing alone, variance reduction alone, low correlation
alone, event failure alone, positive `Skill_VP`, or a threshold crossing. No
universal numerical cutoff for “material degradation” is frozen. Materiality is
adjudicated from the joint diagnostic pattern and the explicit scientific
consequence at the cell level.

## 12. Ghost-skill existence gate

The unrecovered `CASE_B_EXISTENCE_GATE` is removed. It is replaced by:

`GHOST_SKILL_EXISTENCE_GATE = DEFINED_NOT_EXECUTED`

The question is:

> Does at least one station-model-horizon cell satisfy the frozen three-component
> ghost-skill definition?

This gate is defined but not applied in this phase. It does not authorize global
classification or prevalence counting.

## 13. Diagnostic controls

The former A/B/C/D controls are replaced by three logic controls:

| Control | Diagnostic pattern | Expected result |
| --- | --- | --- |
| CONTROL_1 | positive skill + preserved fidelity + preserved event representation | `NOT_GHOST_SKILL` |
| CONTROL_2 | positive skill + severe fidelity degradation + scientifically relevant event/dynamic loss | `GHOST_SKILL_CANDIDATE / DIAGNOSIS_SUPPORTED` |
| CONTROL_3 | non-positive skill + severe fidelity degradation | `NOT_GHOST_SKILL` |

The control specification is stored in
`p4_v2/manifests/p3_scientific_gate_control_cases.json`. It is a logical validation
of the frozen adjudication rule, not a result from the 324-station benchmark.
The controls were evaluated without consulting real benchmark distributions.

## 14. Authorization boundary

The scientific gate is now reconciled, but no global application is performed here.
The next phase may compute cell-level metrics, paired bootstrap inference, event
diagnostics, rank discordance, and the ghost-skill existence gate using only the
verified primary predictions and the frozen definitions above.

No model was retrained, no prediction was regenerated, and no manuscript was
modified in this reconciliation.
