# Module Integration Report: Exceedance Evaluation & Rank Reversal Module

**Repository**: `/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper`  
**Execution Timestamp**: 2026-08-07T15:26:00Z  
**Input Data File**: `outputs/reproduction/predictions_rolling_origin.parquet`  
**Input SHA-256**: `e7073712ba1ab9f3de29621dfa9c96eec634b86ad7bf66ae37a9c098d15b58c4`  
**Producer Commit**: `4909e048e0b9f516031b9e217be0b806fa9dfb8b`  
**Analysis Commit**: `4909e048e0b9f516031b9e217be0b806fa9dfb8b`  
**Operational Status**: `READY_FOR_MODULE_INTEGRATION`  
**Evidence Grade**: `B_HIGH_PENDING_PRODUCER_AUDIT`

---

## 1. Funcionalidad Existente Reutilizada

| Script / Componente | Estado | Uso / Integración |
| :--- | :--- | :--- |
| `scripts/03_exceedance_analysis.py` | Preservado intacto | Definiciones de métricas base de eventos. |
| `scripts/06_build_skill_tables.py` | Preservado intacto | Definiciones de Skill relativo a la persistencia. |
| `scripts/07_murphy_decomposition.py` | Reutilizado directamente | Descomposición de MSE de Murphy (varianza, sesgo condicional, correlación). |
| `scripts/39_rank_comparison_kge_vs_phi.py` | Preservado intacto | Lógica de correlación de rangos entre métricas continuas. |

---

## 2. Funcionalidad Nueva Incorporada

1. **Adaptador de Esquema Canónico (`src/evaluation/exceedance_adapter.py`)**:
   - Trabaja de forma canónica con `origin_time`, `target_time`, `horizon`, `fold`, `model`, `y_true`, `y_pred`, `y_persistence`.
   - Soporta alias secundarios (`origin_date`, `target_date`) y fallback estricto documentado (`date`).
   - Valida coherencia temporal horaria (`target_time - origin_time == horizon * 1h`).
   - Maneja ausencia de metadata de estación (`station_status = "MISSING_FROM_SOURCE"`).
2. **Motor de Métricas de Excedencia Completo**:
   - Contingencias ($TP, FP, FN, TN$).
   - $POD$ / recall, $FAR$, $POFD$, $CSI$, precision, recall, $event\_bias$, $exceedance\_intensity\_error$.
   - Coeficiente de correlación de rangos Kendall $\tau$-b sobre las series temporales de predicción horaria ($\hat{y}_{\text{LightGBM}}$ vs $\hat{y}_{\text{SARIMA}}$), denominado `kendall_taub_prediction_series`.
   - Clasificador formal de rank reversal (`YES`, `NO`, `TRADE_OFF_ONLY`, `NOT_EVALUABLE`).
3. **Script de Integración y Generación de Source Tables (`scripts/41_run_exceedance_integration.py`)**:
   - Procesa `predictions_rolling_origin.parquet` y genera 5 source tables estandarizadas con metadatos completos y SHA-256 en `outputs/source_tables/`.

---

## 3. Auditoría del Productor

* **Rolling-Origin Real**: `VERIFIED` (5 pliegues expansivos; initial train 50%, test 10% cada uno; SARIMA actualizado secuencialmente con `.append(..., refit=False)`).
* **Preprocessing Train-Only**: `VERIFIED` (`causal_inputs` mediante `.ffill()`; `p75_train` estimado estrictamente sobre ventana de entrenamiento).
* **Ausencia de Leakage**: `VERIFIED` (Lags y ventanas móviles causales; proyectores de calendario deterministas).
* **Relación Coherente `origin_time`/`target_time`/`horizon`**: `VERIFIED` (Horizonte medido explícitamente en horas).
* **Persistencia Causal**: `VERIFIED` (`y_persistence` toma la última observación validada en `origin_time`).
* **Producer Commit Tracking**: `VERIFIED` (Commit `4909e048e0b9f516031b9e217be0b806fa9dfb8b` rastreado).
* **Source Series / Station Provenance**: `NOT_VERIFIED` (Metadata de estación ausente en el archivo Parquet de origen `predictions_rolling_origin.parquet`).

---

## 4. Alineación de Casos (Case Alignment)

* **Casos Auditados**: 32.730 filas en `predictions_rolling_origin.parquet`.
* **Identidad de Casos**: Evaluados sobre las claves `(fold, origin_time, target_time, horizon, y_true)`.
* **Resultado**: `aligned = True`. Ambos modelos (`lightgbm` y `sarima`) comparten exactamente las mismas observaciones de prueba por pliegue y horizonte.
* **Reporte Exportado**: `outputs/source_tables/case_alignment_report.csv`.

---

## 5. Auditoría de Duplicados

* **Clave de Verificación**: `(model, fold, origin_time, target_time, horizon)`.
* **Resultado**: `duplicate_count = 0`. No existen registros duplicados.
* **Reporte Exportado**: `outputs/source_tables/duplicate_report.csv`.

---

## 6. Estado de los Umbrales (Threshold Status)

1. **`PRIMARY_FIXED_THRESHOLD`**: `fold_train_p75` (`threshold_status = "VERIFIED_PRIMARY"`). Umbral fijado a priori por pliegue sobre datos de entrenamiento.
2. **`POST_HOC_DIAGNOSTIC`**: `abs_50` ($PM10 > 50\,\mu\text{g/m}^3$, `threshold_status = "PENDING_DOMAIN_VERIFICATION"`). Debido a que los datos son de resolución horaria y la norma de $50\,\mu\text{g/m}^3$ aplica formalmente a promedios diarios, este umbral se mantiene exclusivamente como diagnóstico secundario.

---

## 7. Métricas Calculadas (Resumen Primario `fold_train_p75`)

| Modelo | Horizonte ($h$) | $N$ | RMSE | $Skill_{\text{RMSE}}$ | TP | FP | FN | TN | POD | FAR | POFD | CSI | Event Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 1h | 4181 | 5.67 | -0.0104 | 966 | 220 | 283 | 2712 | 0.7734 | 0.1855 | 0.0750 | 0.6576 | 0.9496 |
| **LightGBM** | 6h | 4161 | 9.15 | 0.0729 | 727 | 374 | 515 | 2545 | 0.5853 | 0.3397 | 0.1281 | 0.4499 | 0.8865 |
| **LightGBM** | 24h | 4071 | 12.80 | -0.0343 | 571 | 532 | 639 | 2329 | 0.4719 | 0.4823 | 0.1859 | 0.3278 | 0.9116 |
| **LightGBM** | 48h | 3952 | 15.23 | -0.0793 | 372 | 582 | 781 | 2217 | 0.3226 | 0.6101 | 0.2079 | 0.2144 | 0.8274 |
| **SARIMA** | 1h | 4181 | 5.43 | 0.0314 | 981 | 272 | 268 | 2660 | 0.7854 | 0.2171 | 0.0928 | 0.6450 | 1.0032 |
| **SARIMA** | 6h | 4161 | 9.05 | 0.0827 | 663 | 319 | 579 | 2600 | 0.5338 | 0.3248 | 0.1093 | 0.4247 | 0.7907 |
| **SARIMA** | 24h | 4071 | 11.82 | 0.0450 | 120 | 47 | 1090 | 2814 | 0.0992 | 0.2814 | 0.0164 | 0.0955 | 0.1380 |
| **SARIMA** | 48h | 3952 | 12.24 | 0.1325 | 0 | 0 | 1153 | 2799 | 0.0000 | NA | 0.0000 | 0.0000 | 0.0000 |

---

## 8. Inversiones de Rango y Correlación de Predicciones

| Horizonte ($h$) | $\Delta Skill$ (LGBM - SARIMA) | $\Delta CSI$ (LGBM - SARIMA) | Kendall $\tau$-b Serie Predicciones (`kendall_taub_prediction_series`) | Reversal en CSI? | Reversal en POD? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1h** | -0.0417 (SARIMA gana) | +0.0126 (LGBM gana) | 0.8577 | **YES** | **NO** |
| **6h** | -0.0098 (SARIMA gana) | +0.0251 (LGBM gana) | 0.6172 | **YES** | **YES** |
| **24h** | -0.0794 (SARIMA gana) | +0.2323 (LGBM gana) | 0.5281 | **YES** | **YES** |
| **48h** | -0.2117 (SARIMA gana) | +0.2144 (LGBM gana) | 0.2356 | **YES** | **YES** |

### Definición Técnica de Kendall $\tau$-b:
* `kendall_taub_prediction_series` se calcula entre los vectores de predicción horaria $\hat{y}_{\text{LightGBM}}$ y $\hat{y}_{\text{SARIMA}}$ a lo largo de las horas de prueba del horizonte $h$, midiendo el grado de acuerdo en la trayectoria de los modelos.

### Resumen de Hallazgos:
* **Observación de Inversión**: Se observan inversiones de rango entre el skill continuo y CSI en los cuatro horizontes evaluados, y entre el skill continuo y POD desde 6 h, para los dos modelos comparados en esta serie.
* **Caso Específico a 48h**: At 48 h, SARIMA retains positive RMSE-based skill relative to persistence ($Skill_{\text{RMSE}}=0.1325$) while detecting none of the exceedance events defined by the train-derived ($p_{75}$) threshold ($POD=0$, $CSI=0$); LightGBM, despite negative RMSE-based skill at the same horizon, retains $CSI=0.2144$.
* **Clasificación del Hallazgo**: En esta serie y bajo el protocolo rolling-origin auditado, los rankings basados en RMSE-skill y en representación de eventos pueden divergir. SARIMA muestra el caso más acusado a 48 h, donde mantiene skill RMSE positivo frente a persistencia mientras pierde completamente los eventos definidos por el umbral ($p_{75}$) de entrenamiento. La caracterización completa como ghost skill queda condicionada a la auditoría conjunta de las métricas de fidelidad dinámica (varianza retenida, ratio std, $\alpha$-KGE, ratio de amplitud, correlación y retención de picos).

---

## 9. Cobertura de Tests

* **Tests Preexistentes y Nuevos (`tests/test_exceedance_adapter.py`)**: 30 unit tests pasando al 100%.
* **Resultado del Test Suite Global**: **30 passed in 6.22s (100% de éxito)**.

---

## 10. Evidence Grade Final

**`EVIDENCE_GRADE = "B_HIGH_PENDING_PRODUCER_AUDIT"`**

### Motivos Estrictos (EVIDENCE-GRADE GUARD):
1. La metadata de estación no está presente en el archivo Parquet de origen (`station_status = "MISSING_FROM_SOURCE"`).
2. Crear `station = "UNSPECIFIED_SINGLE_SERIES"` en tablas derivadas es una conveniencia de esquema y **NO constituye procedencia de estación verificada**.
3. De acuerdo con las reglas de guardia del proyecto, el grado de evidencia se mantiene congelado en `Grade B alta` hasta auditaciones de procedencia de estación externas.

---

## 11. Claims Permitidos y Prohibidos

### Claims Permitidos:
* *"En esta serie y bajo el protocolo rolling-origin auditado, los rankings basados en RMSE-skill y en representación de eventos divergen. A 48 h, SARIMA mantiene skill RMSE positivo frente a persistencia ($Skill_{\text{RMSE}}=0.1325$) mientras pierde completamente los eventos definidos por el umbral ($p_{75}$) de entrenamiento ($POD=0$, $CSI=0$), mientras que LightGBM conserva un $CSI=0.2144$."*

### Claims Prohibidos:
* ❌ NO afirmar validez de grado A (Grade A) ni soporte multiestación.
* ❌ NO afirmar cumplimiento del umbral regulatorio legal de $50\,\mu\text{g/m}^3$ sobre datos horarios sin la debida agregación diaria.
* ❌ NO clasificar el resultado como "ghost skill canónico verificado" hasta cruzarlo con las métricas de fidelidad dinámica.

---

## 12. Siguiente Paso Único

```text
GENERATE_AND_AUDIT_DYNAMIC_FIDELITY_TABLES
```
