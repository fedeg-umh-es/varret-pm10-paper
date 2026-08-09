# P4 — Maximum Reproducible Subset Audit

Audit date: 2026-08-10

## Historical artifact status

The 595-row canonical table was preserved unchanged and remains classified
as:

`AGGREGATE_EMPIRICAL_EVIDENCE_WITH_INCOMPLETE_GRADE_A_PROVENANCE`

It was not used as a target for subset selection or numerical reconciliation.
The current manuscript and publication package were not modified.

## Grade-A unit matrix

`GRADE_A_UNIT_MATRIX.csv` inventories all 595 station–model–horizon units in
the 17-station aggregate scope. It contains 84 exact Grade-A candidates and
511 failures.

| Failure cause | Units |
|---|---:|
| Missing full observed training series for exact train-derived P75 support | 392 |
| SARIMA-aligned persistence and full training support unavailable in the public release | 98 |
| Preserved row-level `y_pred` absent for the model at that station | 21 |
| **Total failed units** | **511** |

The 84 passing units are exactly the local historical units from three
stations: 14 Elche HGB/Ridge units, 35 Valencia-Vivers units, and 35
Zarra-EMEP units.

## Complete rectangular subsets

Two maximal rectangular Grade-A candidates were identified:

1. C1: Valencia-Vivers and Zarra-EMEP × five model families × seven
   horizons = 70 cells.
2. C2: Elche, Valencia-Vivers and Zarra-EMEP × HGB direct and Ridge direct ×
   seven horizons = 42 cells.

Both retain every horizon h=1,…,7. Their exact audit summaries are stored in
`GRADE_A_SUBSET_DIAGNOSTICS.csv` and described in
`MAXIMAL_GRADE_A_SUBSETS.md`.

## Required-chain verification

For every passing unit:

- `y_true`, `y_pred` and persistence rows were matched one-to-one by the
  preserved fold/origin key and target date;
- target date equalled `origin_date + horizon` exactly;
- the date-valued `fold` field equalled `origin_date` exactly;
- persistence-relative RMSE skill was calculated from matched rows;
- DM/HLN statistics and within-dataset BH adjustment used the existing
  implementation;
- alpha and standard-deviation ratio were calculated from matched rows;
- P75 thresholds were calculated from preserved raw PM10 observations at or
  before each origin;
- P75 recall, Rule A, Rule B and formal discordance were calculated from the
  same exact support.

This is an evaluation-chain Grade-A result for the subset evidence. It is not
a claim that the historical SARIMA model-generation configuration can be
rerun, and it is not a new public evidence release.

## Core P4 phenomenon test

### C1

C1 has positive skill in 43/70 units, low alpha among 41 of those 43 units,
28 Rule-A units, 28 Rule-A-to-Rule-B changes, and 17 formal discordant units.
The phenomenon spans two stations, five model families and all seven
horizons. P75 recall is available for all units and participates in the
eligibility rule.

Classification: **MODERATE_P4_STORY**.

### C2

C2 has positive skill in all 42 units, low alpha in all 42, 39 Rule-A units,
39 Rule-A-to-Rule-B changes, and 21 formal discordant units. The phenomenon
spans three stations, two model families and all seven horizons.

Classification: **MODERATE_P4_STORY**, with weaker model-family contrast than
C1.

Rank reversal was not evaluated: the subset audit has no pre-specified rank
reversal rule, and the current Paper-A question is eligibility change.

## Current-paper comparison

| Dimension | Current paper | Preferred C1 |
|---|---:|---:|
| Stations | 17 | 2 |
| Model families | 5 | 5 |
| Horizons | 7 | 7 |
| Cells | 595 | 70 |
| Multi-station | yes | yes |
| Multi-family | yes | yes |
| Multi-horizon | yes | yes |
| Error–fidelity discordance | aggregate provenance incomplete | exact subset evidence |
| Eligibility consequence | 277 → 8 historically reported | 28 → 0 in audit subset |
| Event dimension | aggregate chain incomplete | exact P75 recall support |
| Rank reversal | not the current claim | not evaluated |

The dominant question is retained only in narrowed form. The current
numerical claims are not reusable.

## Decision

The existing evidence contains a genuine, reproducible Grade-A evaluation
subset, but no subset preserves the current 17-station scope. The strongest
scientifically coherent option is therefore a narrower paper based on C1,
subject to a separate preregistered reanalysis and author governance review.

`EXISTING_GRADE_A_SUBSET_SUPPORTS_NARROWER_PAPER`

`NEXT_ACTION = PREREGISTER_SUBSET_REANALYSIS`

No subset reanalysis or manuscript revision was performed in this audit.
