# Environmetrics Evidence and Packaging Audit: Model-Preference Reversal & Dynamic Fidelity

**Project**: P3 — Ghost Skill & Dynamic Fidelity  
**Target Journal**: *Environmetrics* (Wiley)  
**Editorial Status**: Scientific Result Set FROZEN  
**Audit Scope**: Repository Provenance, Evidence Traceability, Methodological Precision, and Submission Packaging  
**Date**: 2026-08-17  

---

## 1. Executive Verdict

- **Evidence Base Integrity**: **PASS** (100% verified against frozen canonical parquet `e7073712ba1a...`).
- **48-h Core Empirical Divergence**: **FULLY SUPPORTED** (SARIMA retains positive RMSE skill $+0.1325$ with zero event detections $\text{POD}=0.0$, $\text{CSI}=0.0$; LightGBM retains event detection $\text{CSI}=0.2144$, $\text{POD}=0.3226$ despite negative RMSE skill $-0.0793$).
- **Model-Preference Reversal**: **FULLY SUPPORTED (PAIRWISE)** (Present across all evaluated horizons $h \in \{1, 6, 24, 48\}$ on CSI, and $h \in \{6, 24, 48\}$ on POD vs continuous RMSE skill).
- **Ghost-Skill Diagnostic Gate**: **FULLY SUPPORTED (BOUNDED DIAGNOSTIC)** (Satisfied in pooled analysis and replicated in 3/5 expanding folds; dynamic collapse and complete event failure persist in 5/5 folds).
- **Environmetrics Packaging Status**: **ACTIONABLE PACKAGING REFACTOR REQUIRED** (The empirical evidence is sound and frozen, but the manuscript framing must transition from a "PM10 / ghost-skill-first" narrative to a "statistical methodology / model-preference reversal" narrative suited for *Environmetrics*).

---

## 2. Repository & Source-of-Truth Artifact Map

### Canonical Primary Data & Predictions
- **Row-Level Input Parquet**: `outputs/reproduction/predictions_rolling_origin.parquet`
  - **SHA-256**: `e7073712ba1ab9f3de29621dfa9c96eec634b86ad7bf66ae37a9c098d15b58c4`
  - **Rows**: 32,730 total prediction rows (16,365 matched evaluation pairs per model across 5 folds and 4 horizons)
  - **Source Commit**: `95c9cbdc8c582f5657523c404afa58e61f5e1137`
  - **Producer Script**: `scripts/run_paper_a_empirical.py --protocol rolling_origin`

### Publication Tables (`outputs/publication_tables/`)
All tables produced deterministically by `scripts/44_generate_publication_source_tables.py` (Packaging commit: `f233a20ae7eb7411e84eaaa326c3aff87601f628`):
1. `pub_table_1_error_metrics.csv` (8 rows): Continuous RMSE, Persistence RMSE, $\text{Skill}_{\text{RMSE}}$
2. `pub_table_2_dynamic_fidelity.csv` (8 rows): `variance_retention`, `std_ratio`, `alpha_kge`, `correlation`, `amplitude_ratio`, `temporal_variability`, `event_amplitude_retention`
3. `pub_table_3_event_metrics.csv` (8 rows): `tp`, `fp`, `fn`, `tn`, `pod`, `far`, `pofd`, `csi`, `precision`, `event_bias`, `exceedance_intensity_error`
4. `pub_table_4_ghost_skill_structure.csv` (8 rows): Comprehensive diagnostic synthesis, fold replication counts, and Kendall $\tau_b$
5. `manuscript_architecture_registry.csv` & `manuscript_evidence_map_registry.csv`: Formal wording and traceability contracts

### Core Source & Diagnostic Tables (`outputs/source_tables/`)
- `event_metrics_by_model_horizon.csv` (Producer: `scripts/41_run_exceedance_integration.py`)
- `rank_reversal_table.csv` (Producer: `scripts/41_run_exceedance_integration.py`)
- `dynamic_fidelity_by_model_horizon.csv` & `ghost_skill_audit_table.csv` (Producer: `scripts/42_run_dynamic_fidelity_integration.py`)
- `dynamic_fidelity_definition_registry.csv` (Mathematical definitions & zero-variance handling)
- `fold_stability_by_model_horizon_fold.csv` (40 fold-level evaluation cells) & `fold_stability_summary_sarima.csv` (Producer: `scripts/43_run_fold_stability_audit.py`)
- `case_alignment_report.csv` & `duplicate_report.csv` (Zero missing, zero duplicates, exact timestamp matching)

### Canonical Figures (`figures/` and `outputs/figure_source_tables/`)
Produced deterministically by `scripts/46_generate_canonical_figures.py`:
- `figures/fig1_horizon_evaluation_divergence.pdf` (Source: `outputs/figure_source_tables/fig1_horizon_evaluation_divergence.csv`)
- `figures/fig2_sarima48_fold_stability.pdf` (Source: `outputs/figure_source_tables/fig2_sarima48_fold_stability.csv`)

---

## 3. Claim-to-Evidence Matrix

| Major Manuscript Claim | Status | Source File & Identifier | Producer Script | Target Table/Figure | Maximum Defensible Wording |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **1. Positive RMSE skill vs persistence** | `VERIFIED` | `pub_table_1_error_metrics.csv` (`model=sarima, horizon=48, skill_rmse=+0.132462`) | `scripts/44_generate_publication_source_tables.py` | Table 1, Fig 1a | SARIMA achieves positive pooled RMSE skill relative to persistence at all evaluated horizons (+0.031 to +0.132). |
| **2. Dynamic-fidelity degradation** | `VERIFIED` | `pub_table_2_dynamic_fidelity.csv` (`model=sarima, horizon=48, variance_retention=0.003651, temporal_variability=0.022280`) | `scripts/42_run_dynamic_fidelity_integration.py` | Table 2, Fig 1b | At 48 h, SARIMA predictions undergo near-total dynamic attenuation, retaining 0.37% of observed variance and 2.2% of temporal variability. |
| **3. Event-representation failure** | `VERIFIED` | `pub_table_3_event_metrics.csv` (`model=sarima, horizon=48, tp=0, fn=1153, pod=0.0, csi=0.0`) | `scripts/41_run_exceedance_integration.py` | Table 3, Fig 1c | SARIMA at 48 h fails to identify any of the 1,153 observed training-defined $p_{75}$ exceedances ($\text{POD}=0.000$, $\text{CSI}=0.000$). |
| **4. LightGBM vs SARIMA 48-h divergence** | `VERIFIED` | `pub_table_1`, `pub_table_2`, `pub_table_3` at `horizon=48` | `scripts/44_generate_publication_source_tables.py` | Table 4, Fig 1 | At 48 h, SARIMA outperforms persistence on RMSE but captures no events; LightGBM has negative RMSE skill but captures 372 events ($\text{CSI}=0.2144$). |
| **5. Model-preference reversal** | `VERIFIED` | `rank_reversal_table.csv` (`horizon=48, rank_reversal_csi=True, rank_reversal_pod=True`) | `scripts/41_run_exceedance_integration.py` | Table 4, Fig 1 | Model selection based solely on RMSE skill selects SARIMA, whereas selection based on threshold event detection selects LightGBM. |
| **6. Dynamic collapse replication across folds** | `VERIFIED` | `fold_stability_summary_sarima.csv` (`dynamic_collapse_all_folds=True, complete_event_failure_all_folds=True`) | `scripts/43_run_fold_stability_audit.py` | Table 4, Fig 2 | Variance collapse ($<0.12\%$) and zero event detection ($\text{POD}=0.0$) occur in all 5 of 5 expanding folds for 48-h SARIMA. |
| **7. Full 3-condition diagnostic replication** | `VERIFIED` | `fold_stability_summary_sarima.csv` (`folds_with_concordant_degradation=3`, `stability_pattern=GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS`) | `scripts/43_run_fold_stability_audit.py` | Table 4, Fig 2 | The complete 3-condition diagnostic pattern (positive skill + dynamic collapse + event failure) replicates in 3 of 5 folds (Folds 0, 1, 4). |
| **8. Series-to-series rank concordance ($\tau_b$)** | `VERIFIED` | `rank_reversal_table.csv` (`horizon=48, kendall_taub_prediction_series=0.235634`) | `scripts/41_run_exceedance_integration.py` | §3.4 text | Kendall's $\tau_b$ between the continuous LightGBM and SARIMA prediction series is 0.236 at 48 h (down from 0.858 at 1 h). |

---

## 4. 48-Hour Evidence Audit

### Target Statement Under Audit:
> *"At 48 h, SARIMA retains positive RMSE-based skill relative to persistence while failing to detect the train-defined exceedance events, whereas LightGBM retains event-detection capability despite inferior RMSE-based skill."*

### Numerical Evidence Verification:
- **Sample Support**: $N = 3,952$ matched forecast–observation pairs ($1,153$ true exceedance events, $2,799$ non-events).
- **Persistence Benchmark**: $\text{RMSE} = 14.1099\,\mu\text{g/m}^3$.
- **SARIMA ($h=48$)**:
  - $\text{RMSE} = 12.2409\,\mu\text{g/m}^3 \implies \mathbf{\text{Skill}_{\text{RMSE}} = +0.1325}$ ($+13.25\%$ improvement over persistence).
  - $\text{Variance Retention} = \mathbf{0.0037}$ ($0.37\%$ of observed variance).
  - $\text{Temporal Variability} = \mathbf{0.0223}$ ($2.23\%$ of observed step-to-step volatility).
  - $\text{TP} = 0, \text{FP} = 0, \text{FN} = 1,153, \text{TN} = 2,799 \implies \mathbf{\text{POD} = 0.0000}, \mathbf{\text{CSI} = 0.0000}$.
- **LightGBM ($h=48$)**:
  - $\text{RMSE} = 15.2283\,\mu\text{g/m}^3 \implies \mathbf{\text{Skill}_{\text{RMSE}} = -0.0793}$ ($-7.93\%$ degradation relative to persistence).
  - $\text{Variance Retention} = \mathbf{0.8659}$ ($86.59\%$ of observed variance).
  - $\text{Temporal Variability} = \mathbf{0.3856}$ ($38.56\%$ of observed step-to-step volatility).
  - $\text{TP} = 372, \text{FP} = 582, \text{FN} = 781, \text{TN} = 2,217 \implies \mathbf{\text{POD} = 0.3226}, \mathbf{\text{CSI} = 0.2144}$.

### Verdict:
**FULLY SUPPORTED (PASS)**. Every quantity in the target statement is mathematically verified at machine precision from `pub_table_1`, `pub_table_2`, and `pub_table_3`.

---

## 5. Rank-Reversal Audit

### Horizon-by-Horizon Model Preference Matrix

| Horizon ($h$) | Continuous RMSE Skill Winner | CSI ($p_{75}$) Operational Winner | POD ($p_{75}$) Operational Winner | Reversal Present? |
| :---: | :---: | :---: | :---: | :---: |
| **1 h** | **SARIMA** ($+0.0314$ vs $-0.0104$) | **LightGBM** ($0.6576$ vs $0.6450$) | **SARIMA** ($0.7854$ vs $0.7734$) | **YES (on CSI)** |
| **6 h** | **SARIMA** ($+0.0827$ vs $+0.0729$) | **LightGBM** ($0.4499$ vs $0.4247$) | **LightGBM** ($0.5853$ vs $0.5338$) | **YES (on CSI & POD)** |
| **24 h** | **SARIMA** ($+0.0450$ vs $-0.0343$) | **LightGBM** ($0.3278$ vs $0.0955$) | **LightGBM** ($0.4719$ vs $0.0992$) | **YES (on CSI & POD)** |
| **48 h** | **SARIMA** ($+0.1325$ vs $-0.0793$) | **LightGBM** ($0.2144$ vs $0.0000$) | **LightGBM** ($0.3226$ vs $0.0000$) | **YES (on CSI & POD)** |

### Nature of Rank Reversal & Kendall $\tau_b$:
1. **Pairwise Nature**: The reversal is strictly pairwise between the statistical comparator (`sarima`) and the machine-learning comparator (`lightgbm`), benchmarked against `persistence`.
2. **Prediction-Series Concordance ($\tau_b$)**: The reported Kendall $\tau_b = 0.236$ at 48 h is computed between the $N = 3,952$ paired continuous point predictions ($\hat{y}_{\text{LightGBM}}$ vs $\hat{y}_{\text{SARIMA}}$). It is **not** a model-ranking coefficient across a model portfolio. It demonstrates that the temporal orderings of the two forecasts decouple as horizon increases ($\tau_b = 0.858$ at 1 h $\to 0.236$ at 48 h).
3. **Defensible Terminology**: Use **"Pairwise Model-Preference Reversal"** or **"Evaluative Model Ordering Inversion"**. Avoid "structural rank reversal across model classes" (which implies a multi-model tournament).

---

## 6. Ghost-Skill Three-Condition Gate

### Diagnostic Criteria (Evaluated for SARIMA at $h = 48\text{ h}$):
1. **Condition 1: Positive Error Skill Relative to Persistence**
   - Result: $\text{Skill}_{\text{RMSE}} = +0.1325 > 0$ $\implies$ **PASS**
2. **Condition 2: Material Degradation Across Dynamic Fidelity**
   - Result: Variance retention $= 0.0037$ ($99.63\%$ collapse), Temporal variability $= 0.0223$ ($97.77\%$ attenuation), Amplitude ratio $= 0.0626$ ($93.74\%$ contraction), Correlation $= -0.0906$ $\implies$ **PASS**
3. **Condition 3: Changed Evaluative / Operational Conclusion**
   - Result: Continuous RMSE prefers SARIMA, but event detection yields $\text{TP}=0, \text{POD}=0.0, \text{CSI}=0.0$ vs LightGBM ($\text{TP}=372, \text{CSI}=0.2144$), completely inverting the preferred operational model $\implies$ **PASS**

### Fold-Level Replication Summary (5 Expanding Folds):
- Dynamic collapse ($\text{VarRet} \le 0.12\%$): **5 of 5 folds** ($100\%$)
- Complete event detection failure ($\text{POD}=0.0, \text{CSI}=0.0, \text{TP}=0$): **5 of 5 folds** ($100\%$)
- Positive continuous RMSE skill ($\text{Skill}_{\text{RMSE}} > 0$): **3 of 5 folds** ($60\%$; Folds 0, 1, 4)
- **Joint Three-Condition Satisfaction**: **3 of 5 folds** ($60\%$)

### Gate Verdict:
**GHOST_SKILL_STATUS = FULLY_SUPPORTED (BOUNDED DIAGNOSTIC)**.

---

## 7. *Environmetrics* Packaging Risks

### Risk 1: Inverted Narrative (Term-First vs Phenomenon-First)
- *Vulnerability*: Leading with the term "Ghost Skill" in the title, opening abstract, and opening introduction creates an impression of neologism marketing before showing the statistical anomaly.
- *Remedy*: Lead with **Model-Preference Reversal in Multi-Horizon Forecast Evaluation**. Present the observable divergence between $L_2$ error skill and threshold/dynamic fidelity first; define "ghost skill" strictly as a secondary bounded diagnostic label for the failure mode.

### Risk 2: Reviewer Objection — "This is Just Conditional Mean Shrinkage / Smoothing Relabeled"
- *Vulnerability*: Lines 623 and 654–657 in the current manuscript state that the study *"does not isolate or test the model-internal mechanism"*. An *Environmetrics* statistical reviewer knows that for any stationary process under $L_2$ loss, $\mathbb{E}[Y_{t+h}|\mathcal{F}_t] \to \mu$ as $h \to \infty$. When autocorrelation decays ($\rho_h \to 0$), the constant mean forecast $\hat{y}=\mu$ has $\text{Var}(\hat{y})=0$, yet achieves an automatic positive persistence-relative skill of $1 - \sqrt{\sigma^2 / (2\sigma^2)} = 1 - 1/\sqrt{2} \approx +0.293$.
- *Remedy*: Do not describe the attenuation as an "untested mystery". Explicitly state the statistical baseline in Section 2 and Discussion: explain that under squared-error loss, conditional mean shrinkage naturally attenuates variance as predictability decays, which is precisely why RMSE-based skill scores relative to persistence are structurally deceptive for threshold-sensitive environmental applications.

### Risk 3: Excessive Atmospheric Domain Specificity vs Methodological Contribution
- *Vulnerability*: The manuscript currently opens like an applied air-pollution case study (WHO guidelines, episode alerts), which obscures the general statistical verification contribution.
- *Remedy*: Frame the paper as a methodological contribution to **point forecast verification under rolling-origin cross-validation for environmental time series**, with hourly PM$_{10}$ serving as the empirical demonstration.

### Risk 4: Conflation in Discussion
- *Vulnerability*: Section 4.1 currently weaves raw numbers, diagnostic labeling, and interpretation into single composite paragraphs.
- *Remedy*: Enforce a strict three-tier structure:
  1. `RESULT`: The empirical divergence and model-preference reversal.
  2. `DIAGNOSIS`: Multi-metric evaluation identifying dynamic collapse and event blind spots.
  3. `INTERPRETATION`: Why aggregate error scores fail to penalize variance attenuation and recommendations for multi-criteria reporting in environmental forecasting.

---

## 8. Required Manuscript Repackaging Changes

1. **Title**:
   - *Current*: `Ghost Skill in Multi-Horizon PM10 Forecasting: Positive Error-Based Skill with Collapsed Dynamic Fidelity and Failed Event Representation`
   - *Proposed*: `Model-Preference Reversal in Multi-Horizon Environmental Forecasting: When RMSE Skill and Dynamic Fidelity Diverge`
2. **Abstract**:
   - Reorder flow: (1) Standard verification relies on persistence-relative RMSE skill; (2) Multi-horizon evaluation reveals that models favored by RMSE can experience complete dynamic collapse and event failure, reversing model preferences; (3) Empirical demonstration on LightGBM vs SARIMA ($h=1\dots 48$ h); (4) Introduce "ghost skill" as the diagnostic label for this 3-condition pattern (replicated in 3/5 folds; collapse in 5/5 folds); (5) Methodological takeaway: verification must include dynamic and event dimensions.
3. **Introduction**:
   - Paragraph 1: Methodological challenge of multi-horizon verification in environmental time series under rolling-origin evaluation.
   - Paragraph 2: $L_2$ loss, conditional expectation shrinkage, and the known limitation that RMSE does not constrain trajectory dynamics.
   - Paragraph 3: The observable phenomenon: model-preference reversal when comparing statistical and ML models.
   - Paragraph 4: Diagnostic formalization ("ghost skill") and single-series experimental setup.
   - Paragraph 5: Methodological scope and boundaries.
4. **Discussion (§4.1)**:
   - Separate cleanly into:
     - *Empirical Divergence* (Report values: $+0.1325$ vs $-0.0793$ RMSE skill, $0.2144$ vs $0.0000$ CSI).
     - *Diagnostic Assessment* (Variance retention $0.0037$, fold stability $3/5$ full pattern, $5/5$ collapse).
     - *Statistical Interpretation* (Connection to conditional mean shrinkage under decaying autocorrelation; why persistence skill alone misleads model selection).
5. **Cover Letter (*Environmetrics*)**:
   - Cite **Bonas et al. (*Environmetrics* 36(1), e2864)** as a relevant methodological precedent in the journal addressing environmental time-series predictability and persistence benchmarking.
   - Clarify that the present work advances the discussion by demonstrating how joint evaluation across error, dynamic fidelity, and event representation exposes model-preference reversals under rolling-origin splits with common support.

---

## 9. Evidence Gaps

- **Station Metadata**: Station-level geographic metadata are not present in the canonical row-level parquet. Handled appropriately by strictly scoping the manuscript as an audited single-series benchmark (`station_status = MISSING_FROM_SOURCE`, Grade B provenance).
- **Multi-Model Tournament**: Only two model classes (`lightgbm` and `sarima`) are evaluated against `persistence`. Handled appropriately by defining all rank reversals as pairwise model-preference inversions.
- **Scientific Evidence**: **ZERO EVIDENCE GAPS** for the single-series model-preference reversal claim.

---

## 10. Completed Implementation Status

The literal editorial refactoring of `manuscript.tex` has been applied directly to the working manuscript.

---

## 11. Post-Refactor Manuscript Audit

### Title Before / After
- **Before**: `Ghost Skill in Multi-Horizon PM$_{10}$ Forecasting: Positive Error-Based Skill with Collapsed Dynamic Fidelity and Failed Event Representation`
- **After**: `Model-Preference Reversal in Multi-Horizon PM$_{10}$ Forecasting: When RMSE Skill and Dynamic Fidelity Diverge`

### Sections Materially Rewritten
1. **Title & Abstract**: Fully refactored to lead with the observable model-preference reversal phenomenon and exact 48-h evidence (+0.1325 vs -0.0793 RMSE skill; CSI = 0.0000 vs 0.2144), followed by the formalization of the bounded "ghost skill" diagnostic label and multi-criteria evaluation takeaway.
2. **Introduction (§1)**: Rebuilt the introductory funnel around the statistical tension between $L_2$ squared-error minimization (conditional mean shrinkage under decaying autocorrelation) and dynamic/event fidelity preservation in environmental time series. Contextualized persistence benchmarking with Bonas et al. (*Environmetrics* 2025). Reduced contributions to three tightly bounded claims.
3. **Data & Evaluation Setting (§2)**: Preserved rolling-origin protocol, common support, and causal train-only preprocessing. Defined all dynamic-fidelity metrics and threshold exceedances prior to stating the ghost-skill diagnostic criteria.
4. **Results (§3)**: Preserved exact frozen numbers in all 4 tables and 2 figures. Re-anchored narratives around pairwise model-preference reversal and fold stability.
5. **Discussion (§4)**: Enforced a strict three-tier progression in §4.1 (`Empirical result` $\to$ `Diagnostic assessment` $\to$ `Statistical interpretation`). Added §4.2 (*Contextualization: Shrinkage and Variance Attenuation in Forecast Verification*) to explicitly preempt the "smoothing relabeled" critique by deriving the asymptotic persistence skill of the unconditional mean ($1 - 1/\sqrt{2} \approx +0.293$) under zero variance retention. Updated §4.4 (*Implications*) to connect with *Environmetrics* forecast evaluation literature.
6. **Limitations (§5) & Conclusion (§6)**: Explicitly bounded the empirical scope to the single-series pairwise contrast, framing dynamic fidelity as a diagnostic complement to standard verification.

### Claims Weakened / Bounded
- Scoped all rank reversal statements as strictly **pairwise model-preference reversals** between LightGBM and SARIMA (benchmark: persistence).
- Formalized "ghost skill" strictly as an operational diagnostic label for the empirical 3-condition pattern, disclaiming any universal forecasting law or causal mechanism.
- Retained Grade B provenance bounding for station metadata.

### Claims Removed
- Removed any implication of multi-model tournament rankings.
- Removed claims implying that the paper discovers variance attenuation or shrinkage.

### Claims Confirmed
- SARIMA 48-h RMSE skill $+0.1325$, variance retention $0.0037$ ($0.37\%$), temporal variability $0.0223$ ($2.2\%$), $\text{POD} = 0.0000$, $\text{CSI} = 0.0000$.
- LightGBM 48-h RMSE skill $-0.0793$, variance retention $0.8659$, temporal variability $0.3856$, $\text{POD} = 0.3226$, $\text{CSI} = 0.2144$.
- Dynamic collapse and complete event failure in 5 of 5 expanding folds; full three-condition diagnostic replication in 3 of 5 folds.
- Pairwise model-preference reversal across all evaluated horizons on CSI ($h \in \{1, 6, 24, 48\}$ h) and on POD ($h \in \{6, 24, 48\}$ h).

### Numerical Corrections & Invariance
- 100% of numerical values in `manuscript.tex` match frozen publication tables (`pub_table_1`, `pub_table_2`, `pub_table_3`, `pub_table_4`) and source tables at machine precision.

### Unresolved Blockers
- **None**. (Standard author action pending before final submission: insert grant/funding numbers at line 487).

### LaTeX Compilation Status
- **Compiler**: `pdflatex` (TeX Live 2026) + `bibtex`
- **Output**: `manuscript.pdf` (15 pages, 408,277 bytes)
- **Fatal Errors**: 0
- **Undefined Citations**: 0
- **Undefined References**: 0
- **Pytest Suite**: 61/61 tests passed (100%)

---

## 12. Final Author Review

### Author-Level Submission Checklist Audit:
1. **Core Story**: Title, Abstract, Introduction funnel, Results headings, Discussion paragraph openers, and Conclusion reconstruct the exact unified narrative: persistence-relative error skill can favor a dynamically collapsed model that misses all threshold events, leading to a pairwise model-preference reversal against an ML model that retains them. Ghost skill is applied strictly as a bounded diagnostic for this 3-condition pattern.
2. **48-h Empirical Anchor**: Verified 100% consistent everywhere:
   - SARIMA ($h=48$): $\text{Skill}_{\text{RMSE}} = +0.1325$, $\text{VarRet} = 0.0037$ ($0.37\%$), $\tau = 0.0223$ ($2.2\%$), $\text{POD} = 0.0000$, $\text{CSI} = 0.0000$ ($0/1,153$ events detected).
   - LightGBM ($h=48$): $\text{Skill}_{\text{RMSE}} = -0.0793$, $\text{VarRet} = 0.8659$, $\tau = 0.3856$, $\text{POD} = 0.3226$, $\text{CSI} = 0.2144$ ($372/1,153$ events detected).
3. **Rank-Reversal Wording**: Strictly designated as "pairwise model-preference reversal". Verified CSI reversal at $h \in \{1, 6, 24, 48\}$ h and POD reversal at $h \in \{6, 24, 48\}$ h. Zero claims of global tournaments.
4. **Ghost-Skill Boundary**: Fully respects the 3-condition definition; accurately reports 3/5 fold replication of the complete pattern and 5/5 fold persistence of dynamic collapse and event failure.
5. **Shrinkage & Variance-Attenuation Derivation (§4.2)**: Mathematical derivation of asymptotic persistence skill ($1 - 1/\sqrt{2} \approx +0.293$) under zero variance retention is mathematically sound, assumptions explicit (stationary process, $\rho_h \to 0$), properly contextualized as literature-grounded statistical interpretation (Murphy 1988; Gneiting 2011; Taylor 2001), and explicitly disclaims discovering shrinkage.
6. **Novelty Control**: Zero instances of "novel", "first", "unprecedented", or "discovery of smoothing".
7. **Single-Case Boundary**: Strictly framed as an audited single-series PM$_{10}$ empirical demonstration (Grade B provenance).
8. **Event Threshold**: Consistently labeled as train-derived $p_{75}$ threshold per fold; never confused with statutory or regulatory standards.
9. **Bonas et al. Precedent**: Accurately cited as directly relevant methodological precedent in *Environmetrics* (36(1), e2864).
10. **Skip Tests**: Both Introduction and Discussion pass paragraph-by-paragraph logical skip tests.
11. **Submission Package Readiness**: All core components present (LaTeX, PDF, BibTeX, Vector Figures, Figure Sources, Declarations).

---

```
PROJECT = P3
TARGET = ENVIRONMETRICS

SCIENTIFIC_CONTENT_STATUS = PASS_FROZEN_VALIDATED
CORE_STORY_STATUS = PASS_UNIFIED_STORY
NUMERICAL_TRACEABILITY = PASS_100_PERCENT
48H_ANCHOR_STATUS = PASS_EXACT_MATCH
RANK_REVERSAL_WORDING = PASS_PAIRWISE_BOUNDED
GHOST_SKILL_BOUNDARY = PASS_3_CONDITION_DIAGNOSTIC
SHRINKAGE_DERIVATION_STATUS = KEEP
SINGLE_CASE_BOUNDARY = PASS_STRICTLY_BOUNDED
EVENT_THRESHOLD_WORDING = PASS_TRAIN_DERIVED_P75
REFERENCE_AUDIT = PASS_ALL_RESOLVED
LATEX_COMPILE_STATUS = PASS_15_PAGES_ZERO_ERRORS
SUBMISSION_PACKAGE_STATUS = READY_FOR_SUBMISSION

BLOCKING_SCIENTIFIC_ISSUES = NONE
BLOCKING_EDITORIAL_ISSUES = NONE
MISSING_SUBMISSION_ITEMS = AUTHOR_GRANT_NUMBER_AT_LINE_500_ONLY

FINAL_RECOMMENDATION = SUBMIT
```


