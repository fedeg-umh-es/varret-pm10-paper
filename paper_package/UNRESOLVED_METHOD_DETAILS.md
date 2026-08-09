# UNRESOLVED METHOD DETAILS

Generated: 2026-08-09

## Status codes
- VERIFIED: confirmed from code or data
- PARTIALLY_VERIFIED: confirmed partially, some detail unclear
- MISSING: not found in any source
- CONFLICTING: found conflicting information

| Detail | Status | Notes |
|---|---|---|
| Station list (17 stations) | VERIFIED | evidence/paper_a/metadata/station_metadata.csv |
| Period 2017-2024 | VERIFIED | stated in manuscript, consistent with data |
| Rolling-origin folds (number) | PARTIALLY_VERIFIED | rolling-origin confirmed; exact fold count not uniquely documented in audited scripts |
| HGB hyperparameters | VERIFIED | scripts/01_generate_e1_rr_lags_only_predictions.py |
| Ridge hyperparameters | VERIFIED | scripts/01_generate_e1_rr_lags_only_predictions.py |
| SARIMA order | PARTIALLY_VERIFIED | auto-ARIMA or fixed order not confirmed in main scripts |
| STL+Ridge decomposition parameters | PARTIALLY_VERIFIED | STL window parameters not fully documented in audited files |
| Bootstrap CI for alpha (n_bootstrap) | PARTIALLY_VERIFIED | CI columns present in master_diagnostic_table.csv; bootstrap n not confirmed |
| DM test: one-sided vs two-sided | VERIFIED | scripts/05_dm_significance.py (two-sided with BH correction) |
| BH correction level (alpha=0.05) | VERIFIED | scripts/05_dm_significance.py |
| HLN correction applied | VERIFIED | stated in manuscript, consistent with dm_stat column |
| Exceedance threshold P75 | VERIFIED | recall_p75 column in master_diagnostic_table.csv |
| Exceedance threshold P90 | VERIFIED | stated in Results section |
| Missing value treatment (listwise deletion) | VERIFIED | stated in Methods section |
| No interpolation/imputation | VERIFIED | stated in Methods section |
| Lags used (Lag 0..6) | VERIFIED | scripts/01_generate_e1_rr_lags_only_predictions.py |
| Zenodo DOI 10.5281/zenodo.18675211 | PARTIALLY_VERIFIED | referenced in manuscript; must confirm actual deposition |

## Material blockers

None identified that prevent writing a defensible manuscript.
