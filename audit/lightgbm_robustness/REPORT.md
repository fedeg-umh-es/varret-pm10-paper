# Work Package A — LightGBM Robustness Arm
## Audit Report

---

## 1. Veredicto LightGBM

```
BLOCKED_BY_PIPELINE_OR_DATA
```

**Razón:** La serie temporal diaria de PM10 está disponible solo para 3 de las 17 estaciones
requeridas (`pm10_daily.csv`, `pm10_valencia_vivers.csv`, `pm10_zarra_emep.csv`).
Las 14 estaciones restantes (08019004, 08019028, 08019043, 08019045, 08019052,
08019054, 08263007, 22125001, 43004005, 43004006, 44013007, 44216001, 45153999,
50008001) no tienen datos brutos en `data/raw/`. El acceso a MITECO está bloqueado
por el proxy (403 Forbidden). El release asset `predictions_all_stations.csv` cubre
únicamente el periodo de evaluación (2020-2024), no el historial de entrenamiento
desde 2017. Usar solo los datos del periodo de evaluación violaría el contrato
experimental (mismos folds, mismas ventanas de entrenamiento, mismos orígenes).

---

## 2. Estado Git y Entorno

- **Repositorio:** `/home/user/varret-pm10-paper`
- **Rama:** `claude/p4-lightgbm-ems-audit-rjs3ch`
- **HEAD inicial:** `4a49b08b041c578ec5981dc1472125b2af0a4d59`
- **HEAD final:** ver commit del repo
- **Python:** 3.11.15
- **LightGBM:** 4.7.0
- **Pandas:** (ver environment.txt)

---

## 3. Pipeline Heredado

El pipeline actual para cada estación:

1. `01_generate_e1_rr_lags_only_predictions.py` — predicciones de `hgb_direct`,
   `ridge_direct`, `stl_ridge_direct`, `seasonal_naive` mediante rolling-origin
   expandente (train = todo lo anterior al origen).
2. `02_generate_sarima_predictions.py` — predicciones SARIMA(1,0,1)(1,0,1)[7]
   con ventana completa 2018-2024.
3. `combine_prediction_tables.py` — combina predicciones.
4. `05_dm_significance.py` — test DM-HLN con corrección BH.
5. `07_build_variance_retention_table.py` — calcula alpha = Var(ŷ)/Var(y).
6. `03_exceedance_analysis.py` — recall_p75 con umbral train-only.
7. `09_build_comprehensive_unified_table.py` — construye `master_diagnostic_table.csv`.

Ver `pipeline_inventory.md` para detalle completo.

---

## 4. Configuración LightGBM (Pre-Registrada)

Parámetros fijados ANTES de cualquier predicción (no ejecutados):

```json
{
  "n_estimators": 200,
  "max_depth": 8,
  "learning_rate": 0.05,
  "num_leaves": 31,
  "subsample": 0.9,
  "colsample_bytree": 0.9,
  "random_state": 42,
  "deterministic": true,
  "force_col_wise": true,
  "verbose": -1,
  "n_jobs": 1
}
```

Estos parámetros replican exactamente los usados en `run_paper_a_empirical.py`
y el módulo `src/models/lgbm_model.py` del repositorio, garantizando equidad
de capacidad respecto a `hgb_direct`.

---

## 5. Equidad Experimental

El protocolo requería:
- mismos 17 × 7 = 119 forecast origins por estación
- mismo expanding window (train = todos los datos históricos anteriores al origen)
- mismos predictores: lag_0 a lag_6 (últimos 7 días de PM10)
- mismo target: PM10 en fecha origen + h días
- misma persistencia: PM10 en fecha de origen
- mismo DM-HLN con corrección BH
- misma definición de alpha = Var(ŷ)/Var(y)
- misma definición de recall_p75

**Incumplimiento material:** Para 14/17 estaciones, no existe la serie histórica
desde 2017. Sin ella, el expanding window no puede reproducirse fielmente: el
modelo LightGBM entrenaría con significativamente menos datos en los orígenes
tempranos (2020-2021), haciendo la comparación metodológicamente inválida.

---

## 6. Auditoría de Leakage

Diseño del pipeline existente verificado: no se detectó leakage.

- `train_end < forecast_timestamp`: CONFIRMADO
- Imputadores y escaladores ajustados solo con train: CONFIRMADO
- Umbral P75 calculado solo con datos de entrenamiento: CONFIRMADO
- Sin tuning sobre el test: CONFIRMADO
- Persistencia usa solo valor causal en el origen: CONFIRMADO

Ver `leakage_report.json` para detalle.

---

## 7. Cobertura

| Métrica | Esperado | Obtenido | Estado |
| --- | ---: | ---: | --- |
| Estaciones | 17 | 3 (datos disponibles) | BLOQUEADO |
| Horizontes | 7 | N/A | BLOQUEADO |
| Filas LightGBM | 119 | 0 | BLOQUEADO |
| Total filas (con LightGBM) | 714 | 595 | BLOQUEADO |
| Filas existentes | 595 | 595 | VERIFICADO ✓ |
| Integridad 595 celdas | PASS | PASS | ✓ |

---

## 8. Resultados por horizonte

NOT_GENERATED — BLOCKED_BY_PIPELINE_OR_DATA

---

## 9. Resultados por estación

NOT_GENERATED — BLOCKED_BY_PIPELINE_OR_DATA

---

## 10. Comparación con HGB

NOT_GENERATED — BLOCKED_BY_PIPELINE_OR_DATA

---

## 11. Comparación con Ridge

NOT_GENERATED — BLOCKED_BY_PIPELINE_OR_DATA

---

## 12. Casos discordantes

NOT_GENERATED — BLOCKED_BY_PIPELINE_OR_DATA

---

## 13. Selección convencional

NOT_GENERATED — BLOCKED_BY_PIPELINE_OR_DATA

---

## 14. Selección fidelity-aware

NOT_GENERATED — BLOCKED_BY_PIPELINE_OR_DATA

---

## 15. Reversiones de selección

NOT_GENERATED — BLOCKED_BY_PIPELINE_OR_DATA

---

## 16. Pareto

NOT_GENERATED — BLOCKED_BY_PIPELINE_OR_DATA

---

## 17. Claim permitido

NINGUNO generado por LightGBM (ejecución bloqueada).

El veredicto `BLOCKED_BY_PIPELINE_OR_DATA` no constituye refutación del fenómeno.
Los resultados existentes de HGB y Ridge (en las 595 celdas) permanecen válidos
y sin modificación.

---

## 18. Claim prohibido

- LightGBM generaliza el fenómeno a "todo boosting". → NO TESTADO
- LightGBM demuestra robustez. → NO TESTADO
- Los 119 casos habrían mostrado mayor concordancia. → NO TESTADO

---

## 19. Tests

| Test | Estado |
| --- | --- |
| row_count_595 | PASS |
| station_count_17 | PASS |
| horizon_set_1_to_7 | PASS |
| model_set_5 | PASS |
| unique_keys_595 | PASS |
| rule_a_277 | PASS |
| rule_b_8 | PASS |
| per_model_119 | PASS |
| lightgbm_119_cells | BLOCKED |
| total_714_cells | BLOCKED |

---

## 20. Limitaciones

**Pipeline:**
- Solo 3/17 estaciones tienen datos brutos disponibles.
- El release asset cubre el periodo de evaluación, no el historial de entrenamiento.

**LightGBM:**
- No ejecutado. Configuración registrada en `config_snapshot.json`.
- Si los datos estuvieran disponibles, la implementación en `src/models/lgbm_model.py`
  y `run_paper_a_empirical.py` es directamente adaptable.

**Acceso:**
- MITECO bloqueado por el proxy del entorno remoto de ejecución.
- GitHub API accesible; release assets verificados pero no descargados (tamaño 73.5 MiB).

---

## 21. Acción Editorial

El veredicto `BLOCKED_BY_PIPELINE_OR_DATA` requiere recuperar los datos brutos de
las 14 estaciones faltantes antes de ejecutar el experimento. La configuración
LightGBM está pre-registrada. Una vez disponibles los datos, el experimento puede
ejecutarse sin modificar el protocolo ni los resultados existentes.

**Acción:** El propietario del repositorio debe proveer los 14 archivos CSV de PM10
diario (desde data/raw/), o habilitar acceso de red a MITECO, para que el arm de
robustez pueda ejecutarse.
