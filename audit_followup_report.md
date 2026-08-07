# Audit Follow-up Report: Existing Functionality & Reuse Plan

**Repository**: `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`  
**Execution Timestamp**: 2026-08-07T15:21:30Z  
**Commit**: `4909e048e0b9f516031b9e217be0b806fa9dfb8b`  
**Status**: `READY_FOR_MODULE_INTEGRATION`  
**Evidence Grade**: `B_HIGH_PENDING_PRODUCER_AUDIT`

---

## 1. Existing Functionality Audit Table

| Requirement | Existing Implementation | Reuse | Modify | New Code Needed |
| :--- | :--- | :---: | :---: | :---: |
| **Parquet Reader & Canonical Schema Adapter** | `scripts/03_exceedance_analysis.py` accepts CSV with `origin_date`/`date`; `run_paper_a_empirical.py` exports Parquet with `origin_time`/`target_time`. | PARTIAL | NO | **YES** (`src/evaluation/exceedance_adapter.py` schema adapter with canonical `origin_time`/`target_time` and aliases). |
| **Case Alignment Verification** | None. Scripts group models independently without checking exact case identity across models. | NO | NO | **YES** (Case alignment audit module checking exact match on `(fold, origin_time, target_time, horizon, y_true)`). |
| **Event Metrics (POD, FAR, POFD, CSI)** | `scripts/03_exceedance_analysis.py` computes `recall`, `precision`, `f1`, `flag_rate`, `base_rate`. Missing `FAR`, `POFD`, `CSI`, `event_bias`, `exceedance_intensity_error`. | PARTIAL | NO | **YES** (Comprehensive event metric calculator in `src/evaluation/exceedance_adapter.py`). |
| **Murphy MSE Decomposition** | `scripts/07_murphy_decomposition.py` fully implements Murphy MSE decomposition (`bias_sq`, `cond_bias_sq`, `irreducible_sq`, `alpha`, `rho`). | **YES** | NO | NO (Reused as-is for variance retention and Murphy analysis). |
| **Rank Comparison & Kendall Tau-b** | `scripts/39_rank_comparison_kge_vs_phi.py` computes Spearman rank correlation between metrics across horizons. | PARTIAL | NO | **YES** (Kendall tau-b rank correlation with explicit handling of ties and pairwise rank reversal classification). |
| **Source Tables Export with SHA-256** | `scripts/08_build_run_summary.py` exports summaries without standardized metadata headers (`input_sha256`, commit hashes). | PARTIAL | NO | **YES** (Standardized CSV exporter with manifest metadata headers in `outputs/source_tables/`). |

---

## 2. Reuse Strategy

1. **Keep Existing Scripts Untouched**: `scripts/03_exceedance_analysis.py`, `scripts/06_build_skill_tables.py`, `scripts/07_murphy_decomposition.py`, and `scripts/39_rank_comparison_kge_vs_phi.py` are left intact to ensure complete backward compatibility.
2. **Build Modular Adapter Layer**: `src/evaluation/exceedance_adapter.py` encapsulates schema normalization, case alignment, event metric calculations, rank reversal tests, and metadata tracking.
3. **Integration Script**: `scripts/41_run_exceedance_integration.py` imports `src/evaluation/exceedance_adapter.py` to process `outputs/reproduction/predictions_rolling_origin.parquet` and produce all required source tables under `outputs/source_tables/`.
