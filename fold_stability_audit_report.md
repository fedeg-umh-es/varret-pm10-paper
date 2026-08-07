# Fold Stability & Non-Threshold Materiality Audit Report

**Repository**: `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`  
**Execution Timestamp**: 2026-08-07T15:42:00Z  
**Input Data File**: `outputs/reproduction/predictions_rolling_origin.parquet`  
**Input SHA-256**: `e7073712ba1ab9f3de29621dfa9c96eec634b86ad7bf66ae37a9c098d15b58c4`  
**Producer Commit**: `4909e048e0b9f516031b9e217be0b806fa9dfb8b` (VERIFIED)  
**Analysis Commit**: `4909e048e0b9f516031b9e217be0b806fa9dfb8b` (VERIFIED)  
**Evidence Status**: `B_HIGH_SOURCE_PROVENANCE_PENDING`  
**Station Metadata Status**: `MISSING_FROM_SOURCE`

---

## 1. Regla de Materialidad Cualitativa sin Umbral (Non-Threshold Materiality Rule)

La regla cualitativa de materialidad para el patrón diagnóstico de *ghost skill* se ha establecido sin calibrar ningún umbral numérico *ad hoc* a partir de estos resultados:

> **A qualitative, non-threshold materiality rule was frozen before final ghost-skill classification and was not tuned to maximize the observed effect.**

Criterio formal de descomposición en tres componentes:
$$\text{Positive Baseline Skill } (Skill_{\text{RMSE}} > 0) + \text{Multi-Dimensional Degradation} + \text{Operational/Scientific Impact (Rank Reversal or Event Failure)}$$

---

## 2. Agrupación de Métricas No Redundantes

Para evitar inflación por transformaciones algebraicas de la dispersión ($\text{std\_ratio} = \sqrt{\text{variance\_retention}} = \alpha_{\text{KGE}}$), las métricas se agrupan estrictamente en cuatro dimensiones científicas independientes:

1. **Dispersión / Amplitud**: `variance_retention` ($\text{Var}(y_{\text{pred}})/\text{Var}(y_{\text{true}})$), `amplitude_ratio` ($\text{IQR}_{95-5}(y_{\text{pred}})/\text{IQR}_{95-5}(y_{\text{true}})$).
2. **Dinamismo / Volatilidad**: `temporal_variability` ($\text{mean}(|\Delta y_{\text{pred}}|)/\text{mean}(|\Delta y_{\text{true}}|)$) calculado dentro de cada fold y estrictamente entre pasos contiguos ($\Delta t = 1\text{h}$).
3. **Asociación Temporal / Fase**: `correlation` (Pearson $r(y_{\text{pred}}, y_{\text{true}})$).
4. **Representación de Eventos de Excedencia**: `POD`, `CSI`, `event_bias`, y `event_amplitude_retention` ($\text{mean}(y_{\text{pred}}[y_{\text{true}} > p_{75}])/\text{mean}(y_{\text{true}}[y_{\text{true}} > p_{75}])$).

---

## 3. Auditoría de Estabilidad por Fold para SARIMA (24h y 48h)

Exportada en [fold_stability_summary_sarima.csv](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/outputs/source_tables/fold_stability_summary_sarima.csv) y [fold_stability_by_model_horizon_fold.csv](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/outputs/source_tables/fold_stability_by_model_horizon_fold.csv):

| Modelo | Horizonte ($h$) | Folds Totales | Dynamic Collapse All Folds? | Event Failure All Folds? | Folds con $Skill_{\text{RMSE}} > 0$ | Patrón de Estabilidad | Pooled `variance_retention` | Fold-wise `variance_retention` (Mediana [Min, Max]) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SARIMA** | **24h** | 5 | **TRUE (5/5)** | **TRUE (5/5)** | 3 / 5 | **`GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS`** | 0.0286 (2.86%) | 0.0316 [0.0249, 0.0404] (3.16%) |
| **SARIMA** | **48h** | 5 | **TRUE (5/5)** | **TRUE (5/5)** | 3 / 5 | **`GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS`** | 0.0037 (0.37%) | 0.0007 [0.0005, 0.0012] (0.07%) |

### Desglose de Niveles de Evidencia en SARIMA-48h:
- **Degradación Dinámica en Todos los Folds**: `dynamic_collapse_all_folds = TRUE` (La retención de varianza fold-wise no supera el 0.12% en ningún fold).
- **Pérdida de Eventos en Todos los Folds**: `event_failure_all_folds = TRUE` ($POD = 0.0$ y $CSI = 0.0$ en los 5 folds).
- **Skill RMSE Positivo por Fold**: $Skill_{\text{RMSE}} > 0$ en 3 de 5 folds (mediana fold-wise +0.1124, máximo +0.2129).
- **Patrón Completo Replicado**: `GHOST_PATTERN_REPLICATED_3_OF_5_FOLDS`.

---

## 4. Clasificaciones Científicas Internas

* **SARIMA 48h**: **`GHOST_SKILL_DIAGNOSTIC_SATISFIED_IN_RECOVERED_SINGLE_SERIES`**
* **SARIMA 24h**: **`STRONG_GHOST_SKILL_CANDIDATE_WITH_FOLD_HETEROGENEITY`**
* **Evidence Status**: **`B_HIGH_SOURCE_PROVENANCE_PENDING`**

---

## 5. Claims del Manuscrito (Guardia de Redacción)

### Formulación Exacta Recomendada para el Manuscrito:
> *"At 48 h, SARIMA retains positive pooled RMSE-based skill relative to persistence while retaining only 0.37% of the observed variance. The dynamic collapse itself is reproduced in all five expanding folds, with fold-wise variance retention never exceeding 0.12%, and exceedance detection fails in all five folds."*

> *"The full diagnostic pattern motivating ghost skill—positive baseline-relative error skill together with severe dynamic-fidelity degradation and operational event failure—is reproduced in three of five expanding folds at 48 h, while dynamic collapse and event failure occur in all five folds."*

### Claims Prohibidos:
* ❌ NO afirmar *"Ghost skill is universally confirmed"* como una constante sin matizar la replicación por folds (3/5 folds) y el soporte single-series.
* ❌ NO elevar a Grado A ni proclamar validez multiestación hasta auditar la procedencia de la fuente de datos de estación.

---

## 6. Cobertura de Tests

* **Pruebas en Pytest**: **39 passed in 1.80s (100% de éxito)**.

---

## 7. Siguiente Paso Único

```text
GENERATE_AND_AUDIT_PUBLICATION_SOURCE_TABLES
```
