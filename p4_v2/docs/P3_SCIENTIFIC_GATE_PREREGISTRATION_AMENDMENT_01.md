# P3 Scientific Gate Preregistration Amendment 01

`PROJECT = P3`  
`CONDITION = lags_only`  
`AMENDMENT_TYPE = pre-analysis integrity correction`

## Link to the original preregistration

Original preregistration, preserved unchanged:

`p4_v2/docs/P3_SCIENTIFIC_GATE_PREREGISTRATION.md`

Original SHA-256 at amendment time:

`eb83863ce375e4a07f03b344779a854d90de01d0bde67c9d55de122f403c9146`

The original declared the maximum family as `324 × 2 × 7 = 4,536`. This amendment
does not rewrite that historical declaration; it corrects the executable primary
analysis family after the frozen common-support reconciliation was verified.

## Amendment decision

`PREVIOUS_STATIONS_EXPECTED = 324`  
`CORRECTED_STATIONS_EXPECTED = 323`  
`PREVIOUS_CELLS_EXPECTED = 4536`  
`CORRECTED_CELLS_EXPECTED = 4522`

`EXCLUDED_STATION = 50297039_10_49`  
`EXCLUSION_REASON = ZERO_PRIMARY_COMMON_SUPPORT`

The station is present in the frozen station panel, but it has zero
`keys_intersection` in the conservative primary support. It is present in the
executed-only sensitivity support with 5,091 support keys (10,182 model rows),
but those keys are not part of the primary analysis family.

The station contributes 14 expected cells (`2 models × 7 horizons`) and zero
reconstructable primary cells. It is therefore excluded from the primary Phase-3
family for data-support integrity, not because of forecast performance.

## Reconciled family

The corrected primary family is:

`323 stations × 2 models × 7 horizons = 4,522 cells`

The primary parquet contains exactly these 4,522 unique station-model-horizon
combinations. Every remaining station has exactly one `xgboost_direct` and one
`sarima` cell for every horizon 1--7. There are no missing or extra combinations
and no duplicate prediction keys.

`323 × 14 = 4522` is verified deterministically from the frozen model and horizon
sets; no forecast value, error, metric, or scientific result was used.

## Evidence and timing

Evidence used for this amendment:

- `p4_v2/data/processed/station_panel_frozen.csv`
- `p4_v2/manifests/station_panel_manifest.json`
- `p4_v2/results/predictions_row_level_lags_only_primary_support.parquet`
- `p4_v2/results/predictions_row_level_lags_only_sensitivity.parquet`
- `p4_v2/results/support_reconciliation_by_station.csv`
- `p4_v2/manifests/p3_support_reconciliation_manifest.json`

`SCIENTIFIC_RESULTS_INSPECTED_BEFORE_AMENDMENT = NO`

No MAE, RMSE, skill, bootstrap, BH correction, dynamic-fidelity metric, event
metric, rank-discordance result, or ghost-skill result was inspected or computed
before this amendment. The earlier failed Phase-3 run stopped during preflight at
the family-count check.

This is a pre-analysis integrity correction, not a post-hoc scientific exclusion.
The frozen station panel, prediction values, support reconciliation, protocol, and
original preregistration remain unchanged.

## Corrected preflight authorization

The corrected family may proceed to a new Phase-3 scientific execution only after
the amended preflight artifact reports:

- `SOURCE_ROWS = 3412754`;
- `STATIONS_EXPECTED = 323`;
- `STATIONS_RECONCILED = 323`;
- `CELLS_EXPECTED = 4522`;
- `CELLS_RECONCILED = 4522`;
- `DUPLICATE_CELLS = 0`;
- `UNRECONCILED_CELLS = 0`.

This amendment authorizes no scientific calculation by itself.
