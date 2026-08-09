# Maximal Grade-A Subsets

Audit date: 2026-08-10

## Interpretation

The Grade-A unit for this audit is one `station × model × horizon` result.
An included unit must have preserved `y_true`, `y_pred`, matched persistence,
origin, target date, horizon, and an exact fold/origin key. The audit then
requires deterministic reconstruction of pooled RMSE, persistence-relative
skill, DM/HLN with the existing within-dataset BH procedure, variance
retention, train-derived P75 threshold and recall, Rule A, Rule B, and formal
discordance.

The current 595-row aggregate was not used to select the subsets or validate
their values. It remains a historical aggregate artifact with incomplete
Grade-A provenance.

## Evidence boundary

The complete local row-level evidence is:

- Elche: HGB direct and Ridge direct, horizons 1–7;
- Valencia-Vivers: HGB direct, Ridge direct, SARIMA, seasonal naive and
  STL+Ridge, horizons 1–7;
- Zarra-EMEP: the same five model families, horizons 1–7.

All three stations have preserved raw PM10 series covering the training
origins. The prediction files contain exact date-valued `fold` fields equal
to `origin_date`; the audit uses that preserved key directly. This supports
exact row matching, although it does not recover a separate named five-fold
manifest for the historical files.

## Maximal rectangular candidates

| Candidate | Stations | Models | Horizons | Cells | Row-level rows | Grade-A chain | Why maximal |
|---|---:|---|---:|---:|---:|---|---|
| C1 | 2 | HGB direct, Ridge direct, SARIMA, seasonal naive, STL+Ridge | 1–7 | 70 | 82,922 | YES | Both stations have all five model families; adding Elche would lose three families. |
| C2 | 3 | HGB direct, Ridge direct | 1–7 | 42 | 58,082 | YES | All three stations have these two families; adding any other family would exclude Elche. |

Single-station rectangles and smaller model subsets are complete but are
strictly contained in C1 or C2 and therefore are not maximal. No individual
horizon was removed: both candidates retain h=1,…,7.

## Candidate C1 — two stations, five families

Candidate C1 preserves the broadest model-family comparison that can be
supported by complete local row-level evidence. Its exact audit diagnostics
are in `GRADE_A_SUBSET_DIAGNOSTICS.csv`.

- Positive persistence-relative skill: 43/70 units.
- Positive-skill units with alpha < 0.50: 41/43.
- Rule A: 28/70.
- Rule B: 0/70 under alpha >= 0.50 and P75 recall >= 0.20.
- Rule-A units changing status: 28/28.
- Formal discordant units: 17.
- P75 recall support: available for all 70 units.

Median diagnostics across units are MAE 7.087, RMSE 10.982, bias -0.072,
skill 0.088, alpha 0.216, standard-deviation ratio 0.465, correlation
0.165, and P75 recall 0.331. These are audit summaries, not population
estimates.

Scientific advantages:

- two independent monitoring stations;
- all five model families and all seven horizons;
- exact event-threshold support from the preserved station series;
- a reproducible eligibility consequence within a multi-family benchmark.

Scientific limitations:

- only two stations;
- historical SARIMA model-generation configuration remains unrecoverable;
- evidence is local and has not been packaged as a new public release;
- the 28/28 change result is specific to this subset and its operational
  criteria;
- no rank-reversal rule was defined or evaluated in this audit.

Story strength: **MODERATE_P4_STORY**. The error–fidelity contrast is
reproducible across two stations, five families and seven horizons, and it
changes eligibility. It is materially narrower than the current 17-station
claim and should not be presented as its empirical replacement without a
separate preregistered subset analysis.

## Candidate C2 — three stations, two families

Candidate C2 maximizes station coverage while retaining the two model
families available at every station.

- Positive persistence-relative skill: 42/42 units.
- Positive-skill units with alpha < 0.50: 42/42.
- Rule A: 39/42.
- Rule B: 0/42 under alpha >= 0.50 and P75 recall >= 0.20.
- Rule-A units changing status: 39/39.
- Formal discordant units: 21.
- P75 recall support: available for all 42 units.

Median diagnostics across units are MAE 6.929, RMSE 10.471, bias -0.153,
skill 0.222, alpha 0.114, standard-deviation ratio 0.337, correlation
0.180, and P75 recall 0.233.

Scientific advantages:

- all three stations with exact raw-series support;
- all seven horizons;
- a repeated error–fidelity contrast across multiple sites;
- deterministic P75 event support and eligibility calculations.

Scientific limitations:

- only two model families;
- no seasonal-naive or STL+Ridge contrast;
- the result is less informative about model-family trade-offs;
- the 39/39 change result is subset-specific and not comparable to 277/8.

Story strength: **MODERATE_P4_STORY**, but weaker than C1 for the intended
multi-family evaluation question. It is stronger on station coverage and
weaker on model contrast.

## Preferred candidate

**C1** is the preferred candidate for a future subset reanalysis because it
retains all five model families, all seven horizons, exact event support and
more of the intended evaluation contrast while still spanning more than one
station. C2 is the principal sensitivity alternative because it maximizes
station coverage.

Neither candidate supports the current 17-station Paper-A claims. The exact
current numerical claims 277, 8, 269, 97.1%, 101 and -0.863 are not reusable
as subset results.

## Audit-only diagnostics

The deterministic wrapper
`audit/final_submission/run_maximum_reproducible_subset_audit.py` uses the
preserved row-level files, preserved raw series, and the existing
`scripts/05_dm_significance.py` DM/HLN/BH implementation. It performs no
model fitting, tuning, imputation of prediction values, or aggregate-table
matching. The resulting unit matrix is
`audit/final_submission/GRADE_A_UNIT_MATRIX.csv`; candidate summaries are in
`GRADE_A_SUBSET_DIAGNOSTICS.csv`.

P75 thresholds use the preserved PM10 observations at dates no later than the
forecast origin. BH adjustment is applied within each station/dataset over
the available model–horizon comparisons, matching the existing code's
within-dataset family definition. Rank reversal was not computed because no
rank-reversal decision rule is part of this subset audit.
