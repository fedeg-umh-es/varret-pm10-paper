# Section 3: Empirical Results

**Repository**: `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`  
**Execution Timestamp**: 2026-08-07T16:53:45Z  
**Experimental Source Commit (`evidence_source_commit`)**: `95c9cbdc8c582f5657523c404afa58e61f5e1137` (Frozen experimental state)  
**Publication Packaging Commit (`publication_packaging_commit`)**: `f233a2080d8ff0428ef5bc1bd80cf8a62ddc6a78` (Versioned publication source tables)  
**Evidence Map Commit (`evidence_map_commit`)**: `ed19b71`  
**Evidence Status**: `B_HIGH_SOURCE_PROVENANCE_PENDING`  
**Station Metadata Status**: `MISSING_FROM_SOURCE`

---

## 3.1 Error-Based Forecast Skill

Evaluating models strictly by root-mean-square error relative to the causal persistence baseline ($Skill_{\text{RMSE}} = 1 - \text{RMSE}_{\text{model}} / \text{RMSE}_{\text{persistence}}$) indicates that forecast skill is strongly dependent on lead time (**Table 1**, `pub_table_1_error_metrics.csv`).

At short horizons ($h=1\text{ h}$), LightGBM ($\text{RMSE} = 5.6668\text{ }\mu\text{g/m}^3$) yields negative skill ($Skill_{\text{RMSE}} = -0.0104$) relative to persistence ($\text{RMSE} = 5.6086\text{ }\mu\text{g/m}^3$), whereas SARIMA ($\text{RMSE} = 5.4328\text{ }\mu\text{g/m}^3$) achieves positive skill ($Skill_{\text{RMSE}} = +0.0314$). At $h=6\text{ h}$, LightGBM reaches its maximum relative error skill ($Skill_{\text{RMSE}} = +0.0729$, $\text{RMSE} = 9.1465\text{ }\mu\text{g/m}^3$) outperforming persistence ($\text{RMSE} = 9.8654\text{ }\mu\text{g/m}^3$), while SARIMA achieves $Skill_{\text{RMSE}} = +0.0827$ ($\text{RMSE} = 9.0499\text{ }\mu\text{g/m}^3$).

At extended horizons ($h=24\text{ h}$ and $h=48\text{ h}$), LightGBM continuous error skill degrades below persistence ($Skill_{\text{RMSE}} = -0.0343$ at 24 h, $\text{RMSE} = 12.8018\text{ }\mu\text{g/m}^3$; $Skill_{\text{RMSE}} = -0.0793$ at 48 h, $\text{RMSE} = 15.2283\text{ }\mu\text{g/m}^3$). Conversely, SARIMA maintains positive continuous skill across extended lead times, attaining $Skill_{\text{RMSE}} = +0.0450$ at 24 h ($\text{RMSE} = 11.8194\text{ }\mu\text{g/m}^3$ vs persistence $\text{RMSE} = 12.3770\text{ }\mu\text{g/m}^3$) and $Skill_{\text{RMSE}} = +0.1325$ at 48 h ($\text{RMSE} = 12.2409\text{ }\mu\text{g/m}^3$ vs persistence $\text{RMSE} = 14.1099\text{ }\mu\text{g/m}^3$). Evaluated exclusively by continuous root-mean-square error, SARIMA appears superior at extended horizons.

---

## 3.2 Dynamic Fidelity and Variance Collapse

Dynamic fidelity metrics reveal that positive continuous error skill can coexist with severe structural attenuation of predicted time-series variability (**Table 2**, `pub_table_2_dynamic_fidelity.csv`). We track four non-redundant dynamic-fidelity indicators across lead times: variance retention ($\text{Var}(y_{\text{pred}})/\text{Var}(y_{\text{true}})$), temporal variability ($\text{mean}(|\Delta y_{\text{pred}}|)/\text{mean}(|\Delta y_{\text{true}}|)$ calculated intra-fold between contiguous 1-h steps), amplitude ratio ($\text{IQR}_{95-5}(y_{\text{pred}})/\text{IQR}_{95-5}(y_{\text{true}})$), and event amplitude retention ($\text{mean}(y_{\text{pred}}[y_{\text{true}} > p_{75}])/\text{mean}(y_{\text{true}}[y_{\text{true}} > p_{75}])$).

While LightGBM retains substantial dynamic variability at $h=48\text{ h}$ (variance retention = 0.8659, temporal variability = 0.3856), SARIMA undergoes progressive structural attenuation. At $h=24\text{ h}$, SARIMA retains 0.0286 (2.86%) of observed variance, with temporal variability dropping to 0.1530 and amplitude ratio falling to 0.1734.

At $h=48\text{ h}$, SARIMA experiences near-total dynamic collapse: pooled variance retention drops to **0.0037 (0.37%)**, temporal variability falls to **0.0223 (2.23%)**, and amplitude ratio falls to **0.0626 (6.26%)**. Rather than capturing high-frequency concentration dynamics, 48-h SARIMA predictions collapse toward an uninformative mean-like trajectory despite outperforming persistence in continuous RMSE. Note that pooled variance retention, standard deviation ratio ($\text{std\_ratio} = 0.0604$), and KGE variability ($\alpha_{\text{KGE}} = 0.0604$) are algebraically related expressions of dispersion attenuation rather than independent lines of evidence.

---

## 3.3 Exceedance Event Representation

Forecast capability to represent PM10 exceedances degrades rapidly for models exhibiting dynamic variance collapse (**Table 3**, `pub_table_3_event_metrics.csv`). Using the train-derived 75th percentile threshold ($p_{75} = 22.0\text{ }\mu\text{g/m}^3$), we evaluate early-warning utility through Probability of Detection (POD) and Critical Success Index (CSI).

At $h=6\text{ h}$, LightGBM retains active event detection capabilities ($\text{POD} = 0.5853$, $\text{CSI} = 0.4499$, $\text{event\_bias} = 0.8865$), and SARIMA retains moderate event representation ($\text{POD} = 0.5338$, $\text{CSI} = 0.4247$, $\text{event\_bias} = 0.7907$). By $h=24\text{ h}$, SARIMA event detection degrades substantially ($\text{POD} = 0.0992$, $\text{CSI} = 0.0955$, $\text{event\_bias} = 0.1380$).

At $h=48\text{ h}$, SARIMA experiences complete event detection failure: $\text{TP} = 0$, $\text{FP} = 0$, $\text{FN} = 1153$, and $\text{TN} = 2799$, yielding **$\text{POD} = 0.0000$** and **$\text{CSI} = 0.0000$** ($\text{event\_bias} = 0.0000$). Although 48-h SARIMA achieves positive continuous RMSE skill ($Skill_{\text{RMSE}} = +0.1325$), it fails entirely to issue any threshold exceedance alerts.

---

## 3.4 Model-Preference Reversal

Disconnects between continuous error skill and operational event metrics lead to a reversal in model selection preference across evaluation criteria (**Table 4**, `pub_table_4_ghost_skill_structure.csv`).

At $h=48\text{ h}$, ranking models by continuous error skill ($Skill_{\text{RMSE}}$) designates SARIMA as superior ($Skill_{\text{RMSE}} = +0.1325$ for SARIMA vs $-0.0793$ for LightGBM). However, evaluating the same 48-h predictions on operational event criteria completely reverses this preference: LightGBM detects exceedance events with $\text{CSI} = 0.2144$ ($\text{POD} = 0.3226$), whereas SARIMA yields $\text{CSI} = 0.0000$ ($\text{POD} = 0.0000$). The Kendall $\tau_b$ rank correlation between the predicted time series of the two models is 0.2356. Model assessment governed strictly by relative RMSE selects a model that provides zero operational utility for threshold alert decisions.

---

## 3.5 Multi-Fold Stability and Ghost-Skill Diagnosis

Multi-fold stability analysis confirms that dynamic collapse and complete event failure in 48-h SARIMA forecasts are persistent structural features across expanding folds (**Table 4**, `pub_table_4_ghost_skill_structure.csv` and `fold_stability_summary_sarima.csv`).

Across all **5 of 5 expanding folds**, 48-h SARIMA predictions exhibit dynamic collapse (`dynamic_collapse_all_folds = True`, fold-wise variance retention median = 0.0007 [0.07%], range = [0.0005, 0.0012], maximum fold value = 0.12%) and complete event failure (`complete_event_failure_all_folds = True`, $\text{POD} = 0.0000$ and $\text{CSI} = 0.0000$ in 5/5 folds). Positive continuous error skill ($Skill_{\text{RMSE}} > 0$) is present in 3 of 5 folds (median = +0.1124, range = [-0.3059, +0.2129]).

The diagnostic pattern defining ghost skill—coexistence of positive baseline-relative error skill with severe dynamic degradation and event detection failure—replicates across **3 of 5 folds** (`stability_pattern = GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS`). In this recovered rolling-origin series, 48-h SARIMA forecasts satisfy the diagnostic criteria for ghost skill (`ghost_skill_status = GHOST_SKILL_DIAGNOSTIC_SATISFIED_IN_RECOVERED_SINGLE_SERIES`).

---

## 3.6 Summary Evidence Matrix

| Model | Horizon ($h$) | $Skill_{\text{RMSE}}$ | Pooled `variance_retention` | Fold-wise `variance_retention` (Median [Min, Max]) | POD | CSI | `stability_pattern` | `ghost_skill_status` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | 1h | -0.0104 | 0.6892 | -- | 0.7734 | 0.6576 | `GHOST_PATTERN_REPLICATED_0_OF_5_FOLDS` | `NEGATIVE_SKILL_NOT_GHOST_SKILL` |
| **LightGBM** | 6h | +0.0729 | 0.5508 | -- | 0.5853 | 0.4499 | `GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS` | `MODERATE_DEGRADATION_EVENTS_RETAINED_NOT_GHOST_SKILL` |
| **LightGBM** | 24h | -0.0343 | 0.5988 | -- | 0.4719 | 0.3278 | `GHOST_PATTERN_REPLICATED_1_OF_5_FOLDS` | `NEGATIVE_SKILL_NOT_GHOST_SKILL` |
| **LightGBM** | 48h | -0.0793 | 0.8659 | -- | 0.3226 | 0.2144 | `GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS` | `NEGATIVE_SKILL_NOT_GHOST_SKILL` |
| **SARIMA** | 1h | +0.0314 | 0.7980 | -- | 0.7854 | 0.6450 | `GHOST_PATTERN_REPLICATED_0_OF_5_FOLDS` | `NOT_GHOST_SKILL` |
| **SARIMA** | 6h | +0.0827 | 0.3566 | -- | 0.5338 | 0.4247 | `GHOST_PATTERN_REPLICATED_2_OF_5_FOLDS` | `NOT_GHOST_SKILL` |
| **SARIMA** | 24h | +0.0450 | 0.0286 | 0.0316 [0.0249, 0.0404] | 0.0992 | 0.0955 | `GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS` | `STRONG_GHOST_SKILL_CANDIDATE_WITH_FOLD_HETEROGENEITY` |
| **SARIMA** | 48h | +0.1325 | 0.0037 | 0.0007 [0.0005, 0.0012] | 0.0000 | 0.0000 | `GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS` | `GHOST_SKILL_DIAGNOSTIC_SATISFIED_IN_RECOVERED_SINGLE_SERIES` |
