# Producer Audit Report: rolling_origin Protocol

**Repository**: `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`  
**Execution Timestamp**: 2026-08-07T15:21:35Z  
**Producer Script**: `scripts/run_paper_a_empirical.py`  
**Producer Manifest**: `outputs/reproduction/run_manifest_rolling_origin.json`  
**Producer Output Parquet**: `outputs/reproduction/predictions_rolling_origin.parquet`  
**Producer Commit**: `4909e048e0b9f516031b9e217be0b806fa9dfb8b`  
**Assigned Evidence Grade**: `B_HIGH_PENDING_PRODUCER_AUDIT`

---

## 1. Audit Checkpoints & Classification

| Checkpoint | Description / Audit Evidence | Classification |
| :--- | :--- | :---: |
| **1. Rolling-Origin Protocol** | 5 expanding folds (`expanding_folds(8760)`: initial train 50%, 5 disjoint 10% test windows of 876h each). LightGBM retrained per fold/horizon; SARIMA fitted once per fold training window and updated sequentially via `.append(..., refit=False)` per hourly origin without parameter refitting. | **VERIFIED** |
| **2. Preprocessing Train-Only** | `causal_inputs(observed)` uses causal `.ffill()`. Threshold `p75_train` is computed strictly using `observed.iloc[:train_end].dropna().quantile(0.75)` on fold training data. | **VERIFIED** |
| **3. Feature Construction & No Leakage** | Lag features (`0, 1, 6, 24, 48, 168`) and rolling statistics (`6, 24, 48, 168`) rely strictly on causal inputs at origin $t$. Target hour/day/month sine/cosine features rely on $t + h$ calendar projections known deterministically in advance. Tested in `test_empirical_protocol.py::test_features_at_origin_do_not_change_when_future_changes`. | **VERIFIED** |
| **4. Temporal Coherence (`origin_time` / `target_time` / `horizon`)** | `target_time` equals `origin_time + pd.to_timedelta(horizon, unit='h')`. `horizon` unit is explicitly hours. | **VERIFIED** |
| **5. Fold Logic & Disjointness** | Expanding train windows ($[0, \text{train\_end})$) with non-overlapping sequential test windows ($[\text{train\_end}, \text{test\_end})$). | **VERIFIED** |
| **6. Baseline Persistence** | `y_persistence = float(causal.iloc[origin])`, strictly using the latest causally validated observation available at `origin_time`. | **VERIFIED** |
| **7. Producer Commit Tracking** | Producer commit tracked (`4909e048e0b9f516031b9e217be0b806fa9dfb8b`). Producer code and environment fully reproducible. | **VERIFIED** |
| **8. Source Series / Station Provenance** | Source dataset input is `data/processed/casa_de_campo_pm10_2023.csv`. However, `station` metadata column is absent in the source Parquet file (`station_status = "MISSING_FROM_SOURCE"`). | **NOT_VERIFIED** |

---

## 2. Evidence Grade Justification

While producer execution checkpoints 1 through 7 are fully **VERIFIED**, checkpoint 8 is classified as **NOT_VERIFIED** because the source Parquet file (`predictions_rolling_origin.parquet`) does not contain a `station` column, and station provenance is not traceable directly from the Parquet schema.

Following the strict **EVIDENCE-GRADE GUARD** directive:
* Creating `station = "UNSPECIFIED_SINGLE_SERIES"` in derived outputs is a schema convenience and does NOT constitute verified station/source provenance.
* Therefore, `EVIDENCE_GRADE` MUST remain **`B_HIGH_PENDING_PRODUCER_AUDIT`**.
