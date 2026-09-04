# P4-v2 Data Freeze

**Status: FROZEN_BEFORE_FORECASTING**  
**Created:** `2026-08-16T08:28:43Z`  
**Git HEAD:** `2f1d2552c25bc6c1aab8b9cf72bf1cbe0ab29560`

## Freeze statement

The station inclusion contract and station panel were frozen before any forecasting model was trained or evaluated. Future station exclusions are forbidden unless caused by a documented data-integrity failure unrelated to model performance. Any such change requires a new panel version and explicit audit trail.

## Frozen artefacts

- Canonical station identifier: `PUNTO_MUESTREO`.
- Frozen daily table: `p4_v2/data/processed/pm10_daily_canonical.parquet`.
- Frozen station panel: `p4_v2/data/processed/station_panel_frozen.csv`.
- Exclusion ledger: `p4_v2/results/station_exclusion_ledger.csv`.
- Identity audit: `p4_v2/results/station_identity_audit.csv`.
- Inclusion contract: `p4_v2/docs/P4_V2_STATION_INCLUSION_CONTRACT.md`.
- Frozen station count: **324** of **415** PM10 sampling-point series.

## Scientific boundary

This freeze contains only source-data identity, temporal support, daily completeness, and missingness decisions. No model, forecast, error metric, skill, variance-retention result, event result, rank, or eligibility outcome was used. Model training remains unauthorized until the Phase-1 protocol is separately frozen.

## Hashes

```text
raw_data_manifest_sha256       = 0e1d30ac728e04b01688c4d1b8485d1562345342e8c66bb6a0151f5713a2e555
canonical_daily_parquet_sha256 = 1d4fbf031ecece900803303944d0d15e64e5bbe22260823ea57e3f6bcb2f8a34
frozen_panel_sha256             = bc2a921e610dc60976f0f9609de09a7d5733e367102f6fe14a2c11361ebb0d27
exclusion_ledger_sha256         = 3c4c1823cb8666cad3b74f460612a799ad2dfc7f1d2fccf25fbc9c0e788dda02
inclusion_contract_sha256       = 813e09009131e48d7ae4a2304597c2485211583a07f19c6b66fc2f15c5ab01f0
```
