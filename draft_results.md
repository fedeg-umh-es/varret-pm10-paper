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

We evaluate continuous error performance across all forecast horizons ($h \in \{1, 6, 24, 48\}\text{ h}$) using root-mean-square error relative to the causal persistence baseline ($Skill_{\text{RMSE}} = 1 - \text{RMSE}_{\text{model}} / \text{RMSE}_{\text{persistence}}$). As documented in **Table 1** (`pub_table_1_error_metrics.csv`), model performance exhibits distinct horizon-dependent behavior.

At short horizons ($h=1\text{ h}$), both LightGBM ($\text{RMSE} = 8.1691\text{ }\mu\text{g/m}^3$) and SARIMA ($\text{RMSE} = 7.8488\text{ }\mu\text{g/m}^3$) exhibit modest skill relative to persistence ($\text{RMSE} = 7.9510\text{ }\mu\text{g/m}^3$), yielding $Skill_{\text{RMSE}} = -0.0274$ for LightGBM and $Skill_{\text{RMSE}} = +0.0129$ for SARIMA. At $h=6\text{ h}$, LightGBM achieves its peak relative skill ($Skill_{\text{RMSE}} = +0.0608$, $\text{RMSE} = 13.9056\text{ }\mu\text{g/m}^3$), outperforming persistence ($\text{RMSE} = 14.8053\text{ }\mu\text{g/m}^3$), while SARIMA achieves $Skill_{\text{RMSE}} = +0.0812$ ($\text{RMSE} = 13.6033\text{ }\mu\text{g/m}^3$).

At extended lead times ($h=24\text{ h}$ and $h=48\text{ h}$), the error-based skill of LightGBM degrades to negative values relative to persistence ($Skill_{\text{RMSE}} = -0.1065$ at 24 h; $Skill_{\text{RMSE}} = -0.0104$ at 48 h). Conversely, SARIMA retains positive continuous error skill relative to persistence across extended horizons, achieving $Skill_{\text{RMSE}} = +0.0450$ at 24 h ($\text{RMSE} = 15.6534\text{ }\mu\text{g/m}^3$ vs persistence $\text{RMSE} = 16.3919\text{ }\mu\text{g/m}^3$) and $Skill_{\text{RMSE}} = +0.1325$ at 48 h ($\text{RMSE} = 14.6294\text{ }\mu\text{g/m}^3$ vs persistence $\text{RMSE} = 16.8632\text{ }\mu\text{g/m}^3$). Evaluated strictly by continuous error metrics, SARIMA appears superior at extended lead times.

---

## 3.2 Dynamic Fidelity and Variance Collapse

To assess whether positive continuous error skill reflects authentic representation of time-series dynamics, we evaluate four non-redundant dynamic-fidelity metrics (**Table 2**, `pub_table_2_dynamic_fidelity.csv`): variance retention ($\text{Var}(y_{\text{pred}})/\text{Var}(y_{\text{true}})$), temporal variability ($\text{mean}(|\Delta y_{\text{pred}}|)/\text{mean}(|\Delta y_{\text{true}}|)$ computed intra-fold between contiguous 1-h steps), amplitude ratio ($\text{IQR}_{95-5}(y_{\text{pred}})/\text{IQR}_{95-5}(y_{\text{true}})$), and event amplitude retention ($\text{mean}(y_{\text{pred}}[y_{\text{true}} > p_{75}])/\text{mean}(y_{\text{true}}[y_{\text{true}} > p_{75}])$).

While LightGBM retains substantial dynamic variance at $h=48\text{ h}$ (variance retention = 0.2569, temporal variability = 0.5178), SARIMA exhibits severe structural attenuation. At $h=24\text{ h}$, SARIMA retains only 0.0286 (2.86%) of the observed time-series variance, with temporal variability collapsing to 0.1530 and amplitude ratio falling to 0.1734.

At $h=48\text{ h}$, SARIMA experiences near-total dynamic collapse: pooled variance retention collapses to **0.0037 (0.37%)**, temporal variability drops to **0.0223 (2.23%)**, and amplitude ratio falls to **0.0626 (6.26%)**. The 48-h SARIMA forecasts cease to track observed high-frequency fluctuations, collapsing toward an uninformative mean-like trajectory despite outperforming persistence in continuous RMSE. Note that variance retention, standard deviation ratio ($\text{std\_ratio} = 0.0604$), and KGE variability ($\alpha_{\text{KGE}} = 0.0604$) are non-independent algebraic expressions of the same underlying dispersion attenuation.

---

## 3.3 Exceedance Event Representation

We next evaluate operational utility for air quality alert systems by testing forecast capability to detect PM10 exceedances defined by the train-derived 75th percentile threshold ($p_{75} = 22.0\text{ }\mu\text{g/m}^3$; **Table 3**, `pub_table_3_event_metrics.csv`).

At $h=6\text{ h}$, LightGBM retains robust event detection capabilities ($\text{POD} = 0.6073$, $\text{CSI} = 0.3671$, $\text{event\_bias} = 0.8877$), while SARIMA retains moderate event representation ($\text{POD} = 0.4859$, $\text{CSI} = 0.3747$). At $h=24\text{ h}$, SARIMA event representation degrades substantially ($\text{POD} = 0.0992$, $\text{CSI} = 0.0955$, $\text{event\_bias} = 0.1380$).

At $h=48\text{ h}$, SARIMA completely fails to detect any exceedance events: $\text{TP} = 0$, $\text{FP} = 0$, $\text{FN} = 1153$, and $\text{TN} = 2799$, resulting in **$\text{POD} = 0.0000$** and **$\text{CSI} = 0.0000$** ($\text{event\_bias} = 0.0000$). Although 48-h SARIMA achieves positive continuous RMSE skill ($Skill_{\text{RMSE}} = +0.1325$), it fails entirely to provide actionable early warning for extreme events.

---

## 3.4 Model Preference Inversion (Rank Reversal)

Comparing continuous error skill against operational event metrics reveals a structural model preference inversion (**Table 4**, `pub_table_4_ghost_skill_structure.csv`).

At $h=48\text{ h}$, evaluating models by continuous error skill ($Skill_{\text{RMSE}}$) designates SARIMA as the superior model ($Skill_{\text{RMSE}} = +0.1325$ for SARIMA vs $-0.0104$ for LightGBM). However, evaluating models by operational event metrics completely reverses this ranking: LightGBM detects events with $\text{CSI} = 0.1839$ ($\text{POD} = 0.2359$), whereas SARIMA yields $\text{CSI} = 0.0000$ ($\text{POD} = 0.0000$). The Kendall $\tau_b$ correlation between predicted time series is attenuated ($\tau_b = 0.5178$).

This structural rank reversal demonstrates that optimizing or selecting forecasting models solely on continuous RMSE skill can select models that are operationally useless for threshold-based decision support.

---

## 3.5 Multi-Fold Stability and Ghost-Skill Diagnosis

To confirm that the observed dynamic collapse and event failure of SARIMA are structural properties rather than artifacts of an isolated evaluation window, we analyze metric stability across 5 expanding folds (**Table 4**, `pub_table_4_ghost_skill_structure.csv` and `fold_stability_summary_sarima.csv`).

At $h=48\text{ h}$, dynamic collapse occurs across **5 of 5 folds** (`dynamic_collapse_all_folds = True`, fold-wise variance retention median = 0.0007 [0.07%], range = [0.0005, 0.0012], maximum fold value = 0.12%). Complete event failure likewise occurs across **5 of 5 folds** (`complete_event_failure_all_folds = True`, $\text{POD} = 0.0$ and $\text{CSI} = 0.0$ in all 5 folds). Positive continuous error skill ($Skill_{\text{RMSE}} > 0$) is present in 3 of 5 folds (median = +0.1124, range = [-0.3059, +0.2129]).

Consequently, the full diagnostic pattern—coexistence of positive baseline-relative RMSE skill with severe dynamic-fidelity collapse and operational event failure—replicates across **3 of 5 folds** (`stability_pattern = GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS`). In this recovered rolling-origin series, the 48-h SARIMA forecasts satisfy the diagnostic definition of ghost skill (`ghost_skill_status = GHOST_SKILL_DIAGNOSTIC_SATISFIED_IN_RECOVERED_SINGLE_SERIES`).

---

## 3.6 Summary Evidence Matrix

| Model | Horizon ($h$) | $Skill_{\text{RMSE}}$ | Pooled `variance_retention` | Fold-wise `variance_retention` (Median [Min, Max]) | POD | CSI | `stability_pattern` | `ghost_skill_status` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | 1h | -0.0274 | 0.6676 | -- | 0.6983 | 0.5957 | `GHOST_PATTERN_REPLICATED_0_OF_5_FOLDS` | `NEGATIVE_SKILL_NOT_GHOST_SKILL` |
| **LightGBM** | 6h | +0.0608 | 0.5439 | -- | 0.6073 | 0.3671 | `GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS` | `MODERATE_DEGRADATION_EVENTS_RETAINED_NOT_GHOST_SKILL` |
| **LightGBM** | 24h | -0.1065 | 0.6814 | -- | 0.5197 | 0.3711 | `GHOST_PATTERN_REPLICATED_1_OF_5_FOLDS` | `NEGATIVE_SKILL_NOT_GHOST_SKILL` |
| **LightGBM** | 48h | -0.0104 | 0.2569 | -- | 0.2359 | 0.1839 | `GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS` | `NEGATIVE_SKILL_NOT_GHOST_SKILL` |
| **SARIMA** | 1h | +0.0129 | 0.7979 | -- | 0.7232 | 0.5714 | `GHOST_PATTERN_REPLICATED_0_OF_5_FOLDS` | `NOT_GHOST_SKILL` |
| **SARIMA** | 6h | +0.0812 | 0.3573 | -- | 0.4859 | 0.3747 | `GHOST_PATTERN_REPLICATED_2_OF_5_FOLDS` | `NOT_GHOST_SKILL` |
| **SARIMA** | 24h | +0.0450 | 0.0286 | 0.0316 [0.0249, 0.0404] | 0.0992 | 0.0955 | `GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS` | `STRONG_GHOST_SKILL_CANDIDATE_WITH_FOLD_HETEROGENEITY` |
| **SARIMA** | 48h | +0.1325 | 0.0037 | 0.0007 [0.0005, 0.0012] | 0.0000 | 0.0000 | `GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS` | `GHOST_SKILL_DIAGNOSTIC_SATISFIED_IN_RECOVERED_SINGLE_SERIES` |
