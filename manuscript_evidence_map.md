# Canonical Manuscript Evidence Map

**Repository**: `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`  
**Execution Timestamp**: 2026-08-07T15:49:10Z  
**Experimental Source Commit (`evidence_source_commit`)**: `95c9cbdc8c582f5657523c404afa58e61f5e1137` (Frozen experimental state)  
**Publication Packaging Commit (`publication_packaging_commit`)**: `f233a2080d8ff0428ef5bc1bd80cf8a62ddc6a78` (Versioned publication source tables)  
**Input Data File**: `outputs/reproduction/predictions_rolling_origin.parquet`  
**Input SHA-256**: `e7073712ba1ab9f3de29621dfa9c96eec634b86ad7bf66ae37a9c098d15b58c4`  
**Evidence Status**: `B_HIGH_SOURCE_PROVENANCE_PENDING`  
**Station Metadata Status**: `MISSING_FROM_SOURCE`

---

## 1. Reglas de Trazabilidad y Dualidad de Commits

1. **Dualidad de Commits**:
   - `evidence_source_commit = 95c9cbdc8c582f5657523c404afa58e61f5e1137`: Identifica el commit congelado donde se fijaron los cálculos experimentales, el filtrado common support (32.730 observaciones) y el protocolo rolling-origin.
   - `publication_packaging_commit = f233a2080d8ff0428ef5bc1bd80cf8a62ddc6a78`: Identifica el commit de empaquetado de las 4 capas de tablas de publicación en `outputs/publication_tables/`.
2. **Principio No Compuesto para la Capa 4**:
   - La Capa 4 (`pub_table_4_ghost_skill_structure.csv`) reúne evidencias independientes (rank reversal, replicación por pliegues, descomposiciones pooled vs fold-wise y clasificación diagnóstica), **evitando constituir un score o índice compuesto de ghost skill**.
3. **Restricción Estricta de Redacción (Manuscript Wording Guard)**:
   - Toda afirmación debe ceñirse al soporte de una única serie recuperada bajo el grado `B_HIGH_SOURCE_PROVENANCE_PENDING`.

---

## 2. Mapa de Evidencia por Elemento del Manuscrito

| Elemento del Manuscrito | Tabla Fuente Publicada (`outputs/publication_tables/`) | Columnas Fuente Utilizadas | Modelo / Horizonte | `evidence_source_commit` | Redacción Máxima Permitida |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tabla 1 / Fig. Error** | `pub_table_1_error_metrics.csv` | `rmse`, `rmse_persistence`, `skill_rmse` | `lightgbm`, `sarima` (h=1, 6, 24, 48) | `95c9cbd` | *"SARIMA exhibits positive RMSE skill relative to persistence at 24 h (+0.045) and 48 h (+0.132), whereas LightGBM shows positive skill at 6 h (+0.061)."* |
| **Tabla 2 / Fig. Fidelidad** | `pub_table_2_dynamic_fidelity.csv` | `variance_retention`, `temporal_variability`, `amplitude_ratio`, `event_amplitude_retention` | `sarima` (h=48) | `95c9cbd` | *"At 48 h, SARIMA retains only 0.37% of observed variance, with severely attenuated temporal variability (0.022) and amplitude ratio (0.063)."* |
| **Tabla 3 / Fig. Eventos** | `pub_table_3_event_metrics.csv` | `TP`, `FP`, `FN`, `TN`, `POD`, `CSI`, `FAR`, `event_bias` | `sarima` (h=48) | `95c9cbd` | *"At 48 h, SARIMA fails to detect any exceedance events under the p75 train-defined threshold, yielding POD = 0.0 and CSI = 0.0."* |
| **Tabla 4 / Fig. Estructura** | `pub_table_4_ghost_skill_structure.csv` | `rank_reversal_csi`, `folds_with_positive_skill`, `dynamic_collapse_all_folds`, `complete_event_failure_all_folds`, `stability_pattern`, `ghost_skill_status` | `sarima` (h=48) | `95c9cbd` | *"The dynamic-fidelity collapse and complete loss of exceedance detection occurred in all five expanding folds at 48 h, while the complete diagnostic pattern combining positive RMSE skill, dynamic collapse, and event failure replicated in three of five folds."* |
| **Resultado Central P4** | `pub_table_4_ghost_skill_structure.csv` | `ghost_skill_status` (`GHOST_SKILL_DIAGNOSTIC_SATISFIED_IN_RECOVERED_SINGLE_SERIES`) | `sarima` (h=48) | `95c9cbd` | *"In this recovered rolling-origin series, the 48-h SARIMA forecasts satisfy the diagnostic definition of ghost skill: improvement relative to persistence in RMSE coexists with severe loss of dynamic fidelity and operational event representation."* |

---

## 3. Matriz de Afirmaciones Prohibidas (Guardia de Redacción)

| Afirmación Prohibida | Causa de Prohibición | Formulación Correcta Permitida |
| :--- | :--- | :--- |
| ❌ *"Ghost skill is a universal property of SARIMA across PM10 networks"* | La evidencia promana de **una única serie recuperada**. | *"In the recovered single series, SARIMA at 48 h satisfies the ghost-skill diagnostic pattern."* |
| ❌ *"Confirmed across 17 stations"* | Los metadatos de estación no están presentes en el Parquet fuente (`STATION_STATUS = MISSING_FROM_SOURCE`). | *"In the recovered consolidated series (station metadata pending verification)..."* |
| ❌ *"Grade A validated pipeline"* | Grado de evidencia congelado en `B_HIGH_SOURCE_PROVENANCE_PENDING`. | *"Assessed under Grade B source provenance pending."* |
| ❌ *"RMSE optimization directly causes variance collapse"* | No existe un experimento de aislamiento causal en este paquete. | *"Positive RMSE skill coexists with severe dynamic variance collapse."* |

---

## 4. Estado de los Gates de Investigación

- `PRODUCER_AUDIT` = **CLOSED**
- `COMMON_CASE_AUDIT` = **CLOSED**
- `EVENT_ANALYSIS` = **CLOSED**
- `DYNAMIC_FIDELITY_AUDIT` = **CLOSED**
- `FOLD_STABILITY_AUDIT` = **CLOSED**
- `MATERIALITY_RULE` = **FROZEN**
- `PUBLICATION_SOURCE_TABLES` = **FROZEN**
- `EXPERIMENTAL_GATE_SINGLE_SERIES` = **CLOSED**
- `MANUSCRIPT_EVIDENCE_MAP` = **CLOSED**
