# Canonical P4 Manuscript Architecture

**Repository**: `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`  
**Execution Timestamp**: 2026-08-07T16:51:50Z  
**Experimental Source Commit (`evidence_source_commit`)**: `95c9cbdc8c582f5657523c404afa58e61f5e1137` (Frozen experimental state)  
**Publication Packaging Commit (`publication_packaging_commit`)**: `f233a2080d8ff0428ef5bc1bd80cf8a62ddc6a78` (Versioned publication source tables)  
**Evidence Map Commit (`evidence_map_commit`)**: `ed19b71`  
**Evidence Status**: `B_HIGH_SOURCE_PROVENANCE_PENDING`  
**Station Metadata Status**: `MISSING_FROM_SOURCE`

---

## 1. Pregunta Científica Central (Research Question)

> **¿Puede la optimización de métricas continuas de error (RMSE) producir predicciones con skill positivo respecto a la persistencia que, simultáneamente, colapsen en su fidelidad dinámica y pierdan por completo su utilidad para la representación de eventos de excedencia?**

---

## 2. Secuencia Lógica de Resultados

La estructura del manuscrito sigue una secuencia deductiva estricta en 5 pasos:

$$\text{Positive Baseline Skill } (Skill_{\text{RMSE}} > 0) \longrightarrow \text{Dynamic Collapse} \longrightarrow \text{Event Failure} \longrightarrow \text{Rank Reversal} \longrightarrow \text{Ghost-Skill Diagnosis}$$

---

## 3. Bloques de Resultados Empíricos trazables frente a `manuscript_evidence_map.md`

### Bloque 1: Skill de Error Cuadrático (Result 1 — Error Skill)
* **Evidencia**: `pub_table_1_error_metrics.csv`
* **Hallazgo**: SARIMA conserva skill de error positivo respecto a la persistencia a 24 h ($Skill_{\text{RMSE}} = +0.045$) y a 48 h ($Skill_{\text{RMSE}} = +0.132$), mientras que LightGBM obtiene su mayor skill a 6 h ($+0.061$).
* **Guardia de Redacción**: *"At 48 h, SARIMA retained positive pooled RMSE-based skill relative to persistence (+0.132)."*

### Bloque 2: Colapso de Fidelidad Dinámica (Result 2 — Dynamic Fidelity)
* **Evidencia**: `pub_table_2_dynamic_fidelity.csv`
* **Hallazgo**: A 48 h, la retención de varianza de SARIMA cae al 0,37% (pooled), con una variabilidad temporal severamente atenuada (0,022) y una ratio de amplitud del 6,3%.
* **Guardia de Redacción**: *"At 48 h, SARIMA retained only 0.37% of observed variance, exhibiting strongly attenuated temporal variability (0.022)."*
* **Nota de Redundancia**: `variance_retention`, `std_ratio` y `alpha_kge` se documentan como expresiones de la misma dimensión de dispersión.

### Bloque 3: Fallo en Representación de Eventos (Result 3 — Event Failure)
* **Evidencia**: `pub_table_3_event_metrics.csv`
* **Hallazgo**: Bajo el umbral $p_{75}$ definido en entrenamiento, SARIMA a 48 h no detecta ningún evento de excedencia ($POD = 0.0, CSI = 0.0$).
* **Guardia de Redacción**: *"At 48 h, SARIMA detected none of the train-defined exceedance events, yielding POD = 0.0 and CSI = 0.0."*

### Bloque 4: Inversión de Rango entre Métricas (Result 4 — Rank Reversal)
* **Evidencia**: `pub_table_4_ghost_skill_structure.csv` (`rank_reversal_csi`, `rank_reversal_pod`)
* **Hallazgo**: La preferencia de modelo se invierte estructuralmente al evaluar con $Skill_{\text{RMSE}}$ (SARIMA preferido a 48 h) frente a $CSI/POD$ (LightGBM preferido a 48 h).
* **Guardia de Redacción**: *"Model rankings undergo structural reversal between continuous error skill and operational event metrics."*

### Bloque 5: Estabilidad por Pliegues Expansivos (Result 5 — Fold Stability)
* **Evidencia**: `pub_table_4_ghost_skill_structure.csv` (`fold_stability_summary_sarima.csv`)
* **Hallazgo**: El colapso dinámico (`variance_retention` mediana 0,07%, máximo 0,12%) y la pérdida de eventos ($POD=0.0, CSI=0.0$) ocurren en **5/5 pliegues** a 48 h. El patrón diagnóstico completo ($Skill_{\text{RMSE}} > 0$ + colapso dinámico + fallo de eventos) se replica en **3/5 pliegues**.
* **Guardia de Redacción**: *"Dynamic collapse and complete exceedance failure occurred in all five expanding folds at 48 h, while the full diagnostic pattern replicated in three of five folds."*

---

## 4. Síntesis Diagnóstica y Formulación Final P4

> **In this recovered rolling-origin series, the 48-h SARIMA forecasts satisfy the diagnostic definition of ghost skill: positive error-based skill coexists with severe degradation of dynamic fidelity and operational event representation.**

---

## 5. Restricciones de Ámbito y Afirmaciones Prohibidas

* ❌ **Prohibido afirmar**: "Ghost skill es una propiedad universal de los modelos autoregresivos".
* ❌ **Prohibido afirmar**: "Validado en 17 estaciones" (los metadatos de estación no están en la fuente).
* ❌ **Prohibido afirmar**: "Relación causal directa entre optimización de RMSE y colapso de varianza".
* ❌ **Prohibido atribuir**: Grado A de evidencia (se mantiene estrictamente `B_HIGH_SOURCE_PROVENANCE_PENDING`).
