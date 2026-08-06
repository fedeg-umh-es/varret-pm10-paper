# P2 Input Provenance

Repository: fedeg-umh-es/varret-pm10-paper
Branch: claude/p2-predictability-bound-ldvcbi
Commit: 4a49b08b041c578ec5981dc1472125b2af0a4d59

## Data Inputs

### Elche
- Raw: `data/raw/pm10_daily.csv`
- Predictions: `outputs/metrics/predictions.csv`
- Dataset keyword: `e1_rr_daily`

### Valencia Vivers
- Raw: `data/raw/pm10_valencia_vivers.csv`
- Predictions: `outputs/metrics/predictions_valencia_vivers.csv`
- Dataset keyword: `e1_rr_valencia_vivers`

### Zarra EMEP
- Raw: `data/raw/pm10_zarra_emep.csv`
- Predictions: `outputs/metrics/predictions_zarra_emep.csv`
- Dataset keyword: `e1_rr_zarra_emep`

## Multi-station skill source
- `evidence/paper_a/aggregates/master_diagnostic_table.csv`

## Skill definition
Skill_RMSE = 1 - RMSE_model / RMSE_persistence

## Persistence baseline
Simple lag-1 persistence: y_pred = y_t (last observed) for all h.

## P4 references status
Protected P4 commits (fdb73b2, a181f7d, 7731570, ab5eb4d, f1e993a) are present on origin/claude/p4-lightgbm-ems-audit-rjs3ch and not modified by P2.
