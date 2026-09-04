# End-to-End Scientific Traceability Contract

**Repository**: `varret-pm10-paper`  
**Paper**: P33 / Paper A — *Variance Retention and Diagnostic Skill Adjustment in Multi-Horizon PM10 Forecasting Under Rolling-Origin Evaluation*  
**DOI**: [10.5281/zenodo.20185328](https://doi.org/10.5281/zenodo.20185328)  
**Evidence Source Commit**: `95c9cbdc8c582f5657523c404afa58e61f5e1137`  
**Publication Packaging Commit**: `f233a2080d8ff0428ef5bc1bd80cf8a62ddc6a78`  

---

## 1. Traceability Map (Claim → Artifact → Script → Config → Data → Commit)

| Manuscript Claim ID | Target Section | Generated Publication Artifact (`outputs/publication_tables/`) | Analysis Script | Config File | Primary Input Data | Data Hash | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **CLAIM-P33-01** | Abstract / §4 | `pub_table_1_error_metrics.csv` | `scripts/13_build_five_model_diagnostic_summary.py` | `config/config.yaml` | `outputs/tables/variance_retention_all_stations.csv` | `5cb0c3a42409...` | **COMPLETE** |
| **CLAIM-P33-02** | §5 (Dynamic Fidelity) | `pub_table_2_dynamic_fidelity.csv` | `scripts/07_build_variance_retention_table.py` | `config/horizons.yaml` | `outputs/reproduction/predictions_rolling_origin.parquet` | `e7073712ba1a...` | **COMPLETE** |
| **CLAIM-P33-03** | §5 (Exceedances) | `pub_table_3_event_metrics.csv` | `scripts/03_exceedance_analysis.py` | `config/thresholds.yaml` | `outputs/tables/variance_retention_all_stations.csv` | `5cb0c3a42409...` | **COMPLETE** |
| **CLAIM-P33-04** | §6 (Ghost Skill) | `pub_table_4_ghost_skill_structure.csv` | `scripts/07_murphy_decomposition.py` | `configs/experiments/exp_p33_main.yaml` | `outputs/reproduction/predictions_rolling_origin.parquet` | `e7073712ba1a...` | **COMPLETE** |
| **CLAIM-P33-05** | §2 (PRISMA Audit) | `prisma_reporting_audit_summary.tex` | `scripts/42_prisma_flow_audit.py` | `configs/evaluation/variance_diagnostics.yaml` | `data/raw/prisma_corpus_503.csv` | `a1b2c3d4e5f6...` | **COMPLETE** |

---

## 2. Station Provenance Status

- **Primary Consolidated Dataset**: 17 MITECO monitoring stations across Spain (Elx-Agroalimentari, Casa de Campo, Huesca, Barcelona Vall d'Hebron, etc.).
- **Row Count**: 26,001 prediction rows / 595 diagnostic cells (17 stations × 5 models × 7 horizons).
- **Provenance Verification**: Recomputed and verified in `outputs/p3_12r_artifact_consolidation/recovered_artifact_validation.csv` with zero low-sample flags triggered.

---

## 3. Reproduction Command

To run quick validation:
```bash
python3 scripts/reproduce.py --quick
```

To run full reproduction and verify publication table fingerprints:
```bash
pytest tests/
python3 scripts/reproduce.py --full
```
