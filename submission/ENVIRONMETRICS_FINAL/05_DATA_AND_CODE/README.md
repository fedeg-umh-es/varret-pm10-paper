# 05_DATA_AND_CODE — Reproduction Package & Data Access

This directory contains the core data tables, execution manifests, and analysis scripts supporting the findings in *Model-Preference Reversal in Multi-Horizon PM$_{10}$ Forecasting: When RMSE Skill and Dynamic Fidelity Diverge*.

Per *Environmetrics* data and code policy, this package is structured so that it can be uploaded as `Data Files` or deposited to a public scientific repository (e.g., Zenodo / GitHub).

---

## 1. Data & Prediction Artefact

- **Evaluation Dataset**: Hourly $\text{PM}_{10}$ concentration series evaluated under a 5-fold expanding rolling-origin protocol with common support across models.
- **Primary Input Prediction Artefact**: `predictions_rolling_origin.parquet`
- **Canonical SHA-256 Checksum**:
  `e7073712ba1ab9f3de29621dfa9c96eec634b86ad7bf66ae37a9c098d15b58c4`
- **Rows**: 32,730 row-level out-of-sample predictions (16,365 matched evaluation pairs per model across horizons $h \in \{1, 6, 24, 48\}$ h and 5 expanding folds).

---

## 2. Scripts & Pipeline Execution Order

The packaged scripts deterministically process the canonical parquet file into all source tables, publication tables, and publication figures:

```bash
# 1. Event detection integration & contingency tables
python3 scripts/41_run_exceedance_integration.py

# 2. Dynamic fidelity integration (variance retention, temporal variability, amplitude)
python3 scripts/42_run_dynamic_fidelity_integration.py

# 3. Multi-fold stability audit across expanding folds
python3 scripts/43_run_fold_stability_audit.py

# 4. Generate publication source tables (Tables 1-4)
python3 scripts/44_generate_publication_source_tables.py

# 5. Generate canonical publication figures (Figures 1-2)
python3 scripts/46_generate_canonical_figures.py
```

---

## 3. Packaged Outputs & Tables

- `tables/pub_table_1_error_metrics.csv`: Continuous RMSE and persistence-relative skill scores (Table 1).
- `tables/pub_table_2_dynamic_fidelity.csv`: Variance retention ($\alpha$), temporal variability ($\tau$), amplitude ratio ($\beta$), and event amplitude retention ($\gamma$) (Table 2).
- `tables/pub_table_3_event_metrics.csv`: TP, FP, FN, TN, POD, CSI, and event bias at train-derived $p_{75}$ threshold (Table 3).
- `tables/pub_table_4_ghost_skill_structure.csv`: Multi-criteria diagnostic summary, rank reversal flags, and fold stability (Table 4).
- `tables/fig1_horizon_evaluation_divergence.csv`: Underlying data plotting Figure 1.
- `tables/fig2_sarima48_fold_stability.csv`: Underlying data plotting Figure 2.

---

## 4. Software & Runtime Dependencies

- **Language**: Python 3.10+ (tested on Python 3.11)
- **Key Dependencies**: `numpy`, `pandas`, `pyarrow`, `scipy`, `matplotlib`, `seaborn` (see `requirements.txt`).

---

## 5. Repository & Code Availability

- **Public Repository**: `https://github.com/fedeg-umh-es/varret-pm10-paper`
- **Experimental Source Commit**: `95c9cbdc8c582f5657523c404afa58e61f5e1137`
- **Publication Packaging Commit**: `f233a20ae7eb7411e84eaaa326c3aff87601f628`
- **Classification**: `PUBLIC_UPLOAD_READY` / `REPOSITORY_ONLY`

---

## 6. Licensing & Redistribution Constraints

- The analysis code and processed summary tables are provided under the MIT open-source license.
- All observational time-series data derive from public ambient air quality monitoring archives.
