# Publication Source Tables Audit Report

**Repository**: `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`  
**Execution Timestamp**: 2026-08-07T15:47:35Z  
**Source Commit**: `95c9cbdc8c582f5657523c404afa58e61f5e1137` (FROZEN AUDIT COMMIT)  
**Input Data File**: `outputs/reproduction/predictions_rolling_origin.parquet`  
**Input SHA-256**: `e7073712ba1ab9f3de29621dfa9c96eec634b86ad7bf66ae37a9c098d15b58c4`  
**Evidence Status**: `B_HIGH_SOURCE_PROVENANCE_PENDING`  
**Station Metadata Status**: `MISSING_FROM_SOURCE`

---

## 1. Resumen de Tablas de Publicación (`outputs/publication_tables/`)

Se han generado 4 tablas de publicación con correspondencia matemática 1:1 directa frente a la evidencia congelada auditada:

### Capa 1: Rendimiento Basado en Error ([pub_table_1_error_metrics.csv](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/outputs/publication_tables/pub_table_1_error_metrics.csv))
* **Métricas**: `model`, `horizon`, `N`, `rmse`, `rmse_persistence`, `skill_rmse`.
* **Propósito**: Proporcionar los valores estándar de error cuadrático medio y el skill relativo a persistencia sin alterar ninguna definición previa.

### Capa 2: Fidelidad Dinámica ([pub_table_2_dynamic_fidelity.csv](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/outputs/publication_tables/pub_table_2_dynamic_fidelity.csv))
* **Métricas**: `variance_retention`, `std_ratio`, `alpha_kge` (marcado explícitamente como idéntico a `std_ratio`), `correlation`, `amplitude_ratio`, `temporal_variability` ($\Delta t=1\text{h}$ contiguo dentro de fold), `event_amplitude_retention`.
* **Propósito**: Cuantificar la atenuación de varianza, volatilidad y amplitud de pico en las cuatro dimensiones dinámicas no redundantes.

### Capa 3: Representación Operacional de Eventos ([pub_table_3_event_metrics.csv](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/outputs/publication_tables/pub_table_3_event_metrics.csv))
* **Métricas**: `TP`, `FP`, `FN`, `TN`, `POD`, `FAR`, `POFD`, `CSI`, `precision`, `event_bias`, `exceedance_intensity_error`.
* **Propósito**: Proveer la tabla de contingencia de excedencia completa y las métricas operacionales de alerta temprana bajo el umbral canónico $p_{75}$.

### Capa 4: Evidencia Estructural y por Fold ([pub_table_4_ghost_skill_structure.csv](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/outputs/publication_tables/pub_table_4_ghost_skill_structure.csv))
* **Métricas**: `rank_reversal_csi`, `rank_reversal_pod`, `kendall_taub_prediction_series`, `folds_with_positive_skill`, `folds_with_concordant_degradation`, `dynamic_collapse_all_folds`, `complete_event_failure_all_folds`, `degraded_event_representation_all_folds`, `stability_pattern`, `ghost_skill_status`.
* **Propósito**: Vincular las clasificaciones diagnósticas finales conservadoras (`GHOST_SKILL_DIAGNOSTIC_SATISFIED_IN_RECOVERED_SINGLE_SERIES` a 48h y `STRONG_GHOST_SKILL_CANDIDATE_WITH_FOLD_HETEROGENEITY` a 24h) con la evidencia de estabilidad por pliegues expansivos.

---

## 2. Verificación de Procedencia de `preprocess_pm10.py`

Se ha verificado que la modificación de `src/data/preprocess_pm10.py` incluida en el commit `95c9cbdc8c582f5657523c404afa58e61f5e1137`:
- Añadió `from __future__ import annotations` e hizo la importación de `yaml` condicional/segura para ejecución en Python 3.9.
- **No realizó ninguna alteración en la lógica ni en las transformaciones numéricas de preprocesamiento de datos**.

---

## 3. Cobertura de Tests y Validación

* **Pruebas en Pytest**: **44 passed in 1.74s (100% de éxito)**.
* **Trazabilidad**: 100% de las tablas en `outputs/publication_tables/` mapean directamente al commit auditado `95c9cbdc8c582f5657523c404afa58e61f5e1137`.

---

## 4. Estado Final de Evidencia y Siguiente Paso

* **Evidence Status**: `B_HIGH_SOURCE_PROVENANCE_PENDING`
* **Conclusión de Etapa**: Las 4 tablas de publicación están empaquetadas, verificadas mediante tests automatizados y totalmente trazables frente al commit congelado.
