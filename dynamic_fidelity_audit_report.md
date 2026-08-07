# Dynamic Fidelity Audit Report: Canonical P4 Evaluation

**Repository**: `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`  
**Execution Timestamp**: 2026-08-07T15:28:50Z  
**Input Data File**: `outputs/reproduction/predictions_rolling_origin.parquet`  
**Input SHA-256**: `e7073712ba1ab9f3de29621dfa9c96eec634b86ad7bf66ae37a9c098d15b58c4`  
**Producer Commit**: `4909e048e0b9f516031b9e217be0b806fa9dfb8b`  
**Analysis Commit**: `4909e048e0b9f516031b9e217be0b806fa9dfb8b`  
**Evidence Grade**: `B_HIGH_PENDING_PRODUCER_AUDIT`  
**Station Metadata Status**: `MISSING_FROM_SOURCE`

---

## 1. Frozen Definitions (Pre-Computation)

All 7 dynamic fidelity metrics were mathematically frozen before computing results on real data and registered in `outputs/source_tables/dynamic_fidelity_definition_registry.csv`:

1. **`variance_retention`**: $\text{Var}(y_{\text{pred}}) / \text{Var}(y_{\text{true}})$ using sample variance ($s^2$, `ddof=1`). Returns `NaN` if $\text{Var}(y_{\text{true}}) \le 1\times 10^{-12}$; returns `0.0` if $\text{Var}(y_{\text{pred}}) \le 1\times 10^{-12}$.
2. **`std_ratio`**: $\text{SD}(y_{\text{pred}}) / \text{SD}(y_{\text{true}})$ using sample standard deviation ($s$, `ddof=1`). Equivalent to $\sqrt{\text{variance\_retention}}$. Returns `NaN` if $\text{SD}(y_{\text{true}}) \le 1\times 10^{-12}$; returns `0.0` if $\text{SD}(y_{\text{pred}}) \le 1\times 10^{-12}$.
3. **`alpha_kge`**: $\text{SD}(y_{\text{pred}}) / \text{SD}(y_{\text{true}})$ (Gupta et al., 2009 / Kling-Gupta Efficiency variability ratio component). Explicitly documented as identical to `std_ratio` by construction ($\alpha_{\text{KGE}} \equiv \text{std\_ratio}$) and NOT presented as independent evidence.
4. **`correlation`**: Pearson correlation coefficient $r(y_{\text{pred}}, y_{\text{true}})$. Returns `NaN` if $\text{SD}(y_{\text{true}}) \le 1\times 10^{-12}$ or $\text{SD}(y_{\text{pred}}) \le 1\times 10^{-12}$. Quantifies phase alignment; does not infer dynamic fidelity alone.
5. **`amplitude_ratio`**: Inter-quantile 95th–5th percentile range ratio $\frac{\text{Q95}(y_{\text{pred}}) - \text{Q5}(y_{\text{pred}})}{\text{Q95}(y_{\text{true}}) - \text{Q5}(y_{\text{true}})}$. Quantifies robust dynamic spread without extreme outlier vulnerability. Returns `NaN` if denominator $\le 1\times 10^{-12}$.
6. **`temporal_variability`**: Mean absolute step-to-step first-difference ratio $\frac{\frac{1}{N-1} \sum_{i=2}^N |y_{\text{pred}, i} - y_{\text{pred}, i-1}|}{\frac{1}{N-1} \sum_{i=2}^N |y_{\text{true}, i} - y_{\text{true}, i-1}|}$. Quantifies step-to-step volatility retention along the time series. Returns `NaN` if denominator $\le 1\times 10^{-12}$.
7. **`peak_retention`**: Ratio of average predicted value to average observed value during peak episodes where observed PM10 strictly exceeds the fold-training set $p_{75}$ quantile: $\frac{\text{mean}(y_{\text{pred}}[y_{\text{true}} > p_{75}])}{\text{mean}(y_{\text{true}}[y_{\text{true}} > p_{75}])}$. Returns `NaN` if no peak events exist or denominator $\le 1\times 10^{-12}$.

---

## 2. Existing Code Reused

* **`src/evaluation/exceedance_adapter.py`**: Schema normalization, timestamps validation, duplicate checking, common case alignment, and contingency metrics engine ($POD, FAR, POFD, CSI, event\_bias$).
* **`src/diagnostics/variance.py`**: Reference logic for variance ratio calculations.
* **`src/kge_diagnostics.py`**: Reference logic for KGE component definitions.

---

## 3. New Code Added

* **`src/evaluation/dynamic_fidelity.py`**: Comprehensive canonical dynamic fidelity evaluation engine computing the 7 mandatory metrics.
* **`scripts/42_run_dynamic_fidelity_integration.py`**: Integration runner generating definition registry and 4 source tables under `outputs/source_tables/`.
* **`tests/test_dynamic_fidelity.py`**: Unit test suite covering perfect fidelity, variance collapse, constant predictions, zero variance in $y_{\text{true}}$, amplitude attenuation, temporal variability loss, peak loss, NaN handling, and common-case preservation.

---

## 4. Common-Case Preservation

* **Total Row Count**: Exactly 32,730 row-level predictions loaded from `outputs/reproduction/predictions_rolling_origin.parquet`.
* **Alignment Status**: 100% aligned across `lightgbm` and `sarima` on `(fold, origin_time, target_time, horizon, y_true)`.
* **Filtering / Resampling**: Zero cases dropped, resampled, smoothed, or aggregated.

---

## 5. Dynamic-Fidelity Metrics by Model and Horizon

| Model | Horizon ($h$) | $N$ | `variance_retention` | `std_ratio` | `alpha_kge` | `correlation` | `amplitude_ratio` | `temporal_variability` | `peak_retention` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 1h | 4181 | 0.6892 | 0.8302 | 0.8302 | 0.8971 | 0.8235 | 0.6965 | 0.8593 |
| **LightGBM** | 6h | 4161 | 0.5508 | 0.7422 | 0.7422 | 0.6933 | 0.7111 | 0.4562 | 0.7731 |
| **LightGBM** | 24h | 4071 | 0.5988 | 0.7738 | 0.7738 | 0.3768 | 0.7909 | 0.4947 | 0.7198 |
| **LightGBM** | 48h | 3952 | 0.8659 | 0.9305 | 0.9305 | 0.1742 | 1.1598 | 0.3856 | 0.6661 |
| **SARIMA** | 1h | 4181 | 0.7980 | 0.8933 | 0.8933 | 0.9033 | 0.8994 | 0.7640 | 0.8979 |
| **SARIMA** | 6h | 4161 | 0.3566 | 0.5971 | 0.5971 | 0.7084 | 0.5976 | 0.5032 | 0.7137 |
| **SARIMA** | 24h | 4071 | 0.0286 | 0.1691 | 0.1691 | 0.4727 | 0.1734 | 0.1530 | 0.5443 |
| **SARIMA** | 48h | 3952 | 0.0037 | 0.0604 | 0.0604 | -0.0906 | 0.0626 | 0.0223 | 0.5067 |

---

## 6. Horizons Showing Positive RMSE Skill ($Skill_{\text{RMSE}} > 0$)

* **LightGBM**:
  - $h=6\text{h}$ ($Skill_{\text{RMSE}} = +0.0729$)
* **SARIMA**:
  - $h=1\text{h}$ ($Skill_{\text{RMSE}} = +0.0314$)
  - $h=6\text{h}$ ($Skill_{\text{RMSE}} = +0.0827$)
  - $h=24\text{h}$ ($Skill_{\text{RMSE}} = +0.0450$)
  - **$h=48\text{h}$ ($Skill_{\text{RMSE}} = +0.1325$)**

---

## 7. Horizons Showing Material Dynamic-Fidelity Degradation

* **LightGBM**:
  - $h=6\text{h}$: `temporal_variability` = 0.4562 (< 0.5).
  - $h=24\text{h}$: `temporal_variability` = 0.4947 (< 0.5).
  - $h=48\text{h}$: `temporal_variability` = 0.3856 (< 0.5).
* **SARIMA**:
  - **$h=24\text{h}$**: `variance_retention` = 0.0286 (collapse to 2.86%), `std_ratio` / `alpha_kge` = 0.1691 (collapse to 16.9%), `amplitude_ratio` = 0.1734 (collapse to 17.3%), `temporal_variability` = 0.1530 (collapse to 15.3%).
  - **$h=48\text{h}$**: `variance_retention` = 0.0037 (collapse to 0.37%), `std_ratio` / `alpha_kge` = 0.0604 (collapse to 6.0%), `amplitude_ratio` = 0.0626 (collapse to 6.3%), `temporal_variability` = 0.0223 (collapse to 2.2%).

---

## 8. Horizons Showing Event-Representation Degradation

* **LightGBM**: None (POD $\ge 0.3226$, CSI $\ge 0.2144$, event_bias $\ge 0.8274$ across all horizons).
* **SARIMA**:
  - **$h=24\text{h}$**: $POD = 0.0992$ (< 0.1), $CSI = 0.0955$ (< 0.1), $event\_bias = 0.1380$ (< 0.5).
  - **$h=48\text{h}$**: $POD = 0.0000$ (= 0.0), $CSI = 0.0000$ (= 0.0), $event\_bias = 0.0000$ (= 0.0). Complete event detection failure.

---

## 9. Ghost-Skill Candidates (`ghost_skill_status`)

Diagnostic status assigned without ad-hoc cutoff optimization:

| Model | Horizon ($h$) | $Skill_{\text{RMSE}}$ | `std_ratio` | `variance_retention` | POD | CSI | `ghost_skill_status` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| LightGBM | 1h | -0.0104 | 0.8302 | 0.6892 | 0.7734 | 0.6576 | `NEGATIVE_SKILL_NOT_GHOST_SKILL` |
| LightGBM | 6h | +0.0729 | 0.7422 | 0.5508 | 0.5853 | 0.4499 | `CANDIDATE_PENDING_MATERIALITY_RULE` |
| LightGBM | 24h | -0.0343 | 0.7738 | 0.5988 | 0.4719 | 0.3278 | `NEGATIVE_SKILL_NOT_GHOST_SKILL` |
| LightGBM | 48h | -0.0793 | 0.9305 | 0.8659 | 0.3226 | 0.2144 | `NEGATIVE_SKILL_NOT_GHOST_SKILL` |
| SARIMA | 1h | +0.0314 | 0.8933 | 0.7980 | 0.7854 | 0.6450 | `NOT_GHOST_SKILL` |
| SARIMA | 6h | +0.0827 | 0.5971 | 0.3566 | 0.5338 | 0.4247 | `NOT_GHOST_SKILL` |
| **SARIMA** | **24h** | **+0.0450** | **0.1691** | **0.0286** | **0.0992** | **0.0955** | **`CANDIDATE_PENDING_MATERIALITY_RULE`** |
| **SARIMA** | **48h** | **+0.1325** | **0.0604** | **0.0037** | **0.0000** | **0.0000** | **`CANDIDATE_PENDING_MATERIALITY_RULE`** |

---

## 10. Cases Not Evaluable

* **Total Unevaluable Cells**: 0. All 8 model $\times$ horizon cells were fully evaluable over 32,730 common cases.

---

## 11. Evidence Grade

**`EVIDENCE_GRADE = "B_HIGH_PENDING_PRODUCER_AUDIT"`**

* **Justification**: Source Parquet lacks a `station` column (`station_status = "MISSING_FROM_SOURCE"`). Provenance remains frozen at Grade B high until external station provenance audit.

---

## 12. Claims Permitted

* *"En esta serie y bajo el protocolo rolling-origin auditado, SARIMA a 48 h presenta una retención de varianza de tan solo 0,37% ($s_{\text{pred}}^2 / s_{\text{true}}^2 = 0.0037$) y un ratio de desviación típica de 6,0% ($\text{std\_ratio} = 0.0604$), perdiendo completamente la variabilidad temporal y la detección de eventos ($POD=0$, $CSI=0$), a pesar de mantener un $Skill_{\text{RMSE}} = +0.1325$ respecto a la persistencia."*
* *"Este resultado confirma empíricamente la coexistencia de skill de error positivo frente a la persistencia con una degradación severa de la fidelidad dinámica y la capacidad de representación de eventos."*

---

## 13. Claims Prohibidos

* ❌ NO afirmar clasificación definitiva de "ghost skill canónico universal" con un umbral numérico optimizado ad hoc.
* ❌ NO declarar evidencia Grado A ni validez multiestación.
* ❌ NO presentar $\alpha_{\text{KGE}}$ y `std_ratio` como dos fuentes independientes de evidencia (son idénticos por definición).

---

## 14. Siguiente Paso Único

```text
GENERATE_AND_AUDIT_PUBLICATION_SOURCE_TABLES
```
