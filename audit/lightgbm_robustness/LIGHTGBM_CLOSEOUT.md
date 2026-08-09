# LIGHTGBM ROBUSTNESS ARM — CLOSEOUT AUDIT REPORT

> **Fecha de cierre:** 2026-08-09
> **Rama:** `codex/p4-lightgbm-ems-gap-audit`
> **HEAD de cierre:** `30cd774210f62f47cf39bfd8f929ef82afe8f953`
> **Auditor:** Antigravity (automated)
> **Mandato:** Cerrar formalmente el frente LightGBM antes de cualquier modificación de Paper A.

---

## VEREDICTO FINAL

```
SYNTHETIC_ARM — NO_REAL_LIGHTGBM_EXECUTION
```

**El brazo LightGBM no puede usarse como evidencia de robustness porque no existe un modelo LightGBM entrenado.** Sus 119 celdas de métricas son combinaciones lineales determinísticas de los valores de `hgb_direct` y `ridge_direct`, calculadas a mano en el script generador. Esto hace que el brazo sea, en el mejor caso, una extrapolación analítica (que puede reportarse como tal) y, en el peor caso, una fabricación inadvertida si se presenta sin esta advertencia.

> La conclusión científica central de Paper A sobre desacoplamiento error–fidelidad NO requiere LightGBM y sigue siendo válida sobre los 595 datos originales.

---

## 1. HALLAZGO CRITICO: DATOS SINTETICOS

### 1.1 Que hace `generate_lightgbm_arm.py` (lineas 48-87)

El script **no llama a LightGBM.fit() ni a LightGBM.predict()**. Genera metricas mediante formulas fijas:

| Metrica | Formula aplicada |
|---|---|
| `skill` | `0.97 x hgb_skill + 0.03 x ridge_skill + 0.002 / (horizon + 1)` |
| `alpha` | `clip(0.95 x hgb_alpha + 0.05 x ridge_alpha - 0.003 x horizon, min=0.05)` |
| `recall_p75` | `0.96 x hgb_recall + 0.04 x ridge_recall` |
| `dm_significant` | `bool(hgb_dm_significant)` (copiado directamente) |
| `dm_pval_bh` | `hgb_dm_pval x 0.98` (si significativo) |
| `dm_stat` | `hgb_dm_stat x 1.02` |

### 1.2 Verificacion numerica (ejecutada 2026-08-09)

```
Desviacion maxima skill (vs. formula):  8.3e-17   (precision de maquina)
Desviacion maxima alpha (vs. formula):  9.0e-17   (precision de maquina)
Correlacion Pearson alpha_LGB vs alpha_HGB: 0.999212
```

Los 119 valores son **exactos** a precision de maquina. No existe ninguna muestra de datos de PM10 procesada por LightGBM.

### 1.3 El bloque `lgb.LGBMRegressor(...)` en REPORT.md (seccion 4)

Describe una configuracion (hiperparametros) que **nunca se ejecuto**. Es documentacion prospectiva, no retrospectiva. El preregistro menciona `lightgbm_version=4.6.0`, pero no hay evidencia de que esa version haya producido ningun forecast.

---

## 2. INVENTARIO DE ARTEFACTOS Y ESTADO

| Artefacto | Estado | Observacion |
|---|---|---|
| `generate_lightgbm_arm.py` | SINTETICO | Genera 119 celdas por interpolacion, sin entrenar LightGBM |
| `master_diagnostic_table_with_lightgbm.csv` | CONTAMINADO | Contiene 119 filas sinteticas mezcladas con 595 reales |
| `preregistered_protocol.json` | VALIDO | Protocolo correcto, pero no ejecutado en su totalidad |
| `checks.json` | VALIDO | Comprueba estructura; no detecta la sintesis (por diseno) |
| `leakage_report.json` | IRRELEVANTE | Argumenta ausencia de fuga para datos que no existen |
| `existing_results_integrity.json` | VALIDO | Confirma que las 595 filas originales son inalteradas |
| `master_table_integrity.json` | VALIDO | Confirma integridad estructural (no implica datos reales) |
| `REPORT.md` (veredicto) | INVALIDO | `LIGHTGBM_ROBUSTNESS_CONFIRMED` no puede sostenerse con datos sinteticos |
| `analyze_lightgbm_robustness.py` | CONDICIONADO | Analisis correcto sobre datos sinteticos; estadisticas internas consistentes |

---

## 3. LO QUE SI ES VALIDO

Los 595 datos originales de `master_diagnostic_table.csv` son reales, reproducibles, y constituyen la evidencia primaria de Paper A. Todos los resultados de la seccion de analisis de decisiones (decision-change, 277/8, 97.1%, rho=-0.863, 101 discordantes) se derivan de esos 595 datos y permanecen **INALTERADOS**.

El brazo LightGBM puede interpretarse como **simulacion analitica** (extrapolacion deterministica de HGB), con las siguientes propiedades internas consistentes:

| Estadistico | Valor (datos sinteticos) | Interpretacion |
|---|---|---|
| skill > 0 | 119/119 (100%) | Consecuencia directa de hgb_skill x 0.97 > 0 cuando hgb_skill > 0 |
| alpha < 0.50 | 118/119 (99.2%) | Consecuencia directa de formula heredando colapso de HGB |
| dm_significant | 111/119 (93.3%) | Copiado directamente de hgb_dm_significant |
| Discordantes | 52/119 | Consecuencia aritmetica, no empirica |
| rho(alpha, skill) | -0.780 | Definicional: alpha ~= f(hgb_alpha) que correlaciona con hgb_skill |
| Corr(alpha_LGB, alpha_HGB) | 0.999 | Por construccion |

Estos valores **no constituyen evidencia independiente** de robustness de LightGBM.

---

## 4. MATRIZ CLAIM-EVIDENCIA

| Claim en REPORT.md | Evidencia real | Riesgo | Decision |
|---|---|---|---|
| "LightGBM reproduce el patron de HGB" | Sintetico; corr=0.999 por formula | ALTO | REMOVE o REFRAME |
| "caracteristica general del boosting" | Sin datos empiricos de LightGBM | CRITICO | REMOVE |
| "53 discordantes adicionales" | Aritmetica, no empirica | ALTO | REMOVE |
| "Pareto fronts 68.9%" | Calculado sobre datos sinteticos | MEDIO | REMOVE |
| "Seleccion fidelity-aware: 2 reversals" | Calculado sobre datos sinteticos | ALTO | REMOVE |
| "595 celdas originales INALTERADAS" | VERIFICADO bit-identico | NINGUNO | MANTENER |
| "Preregistro de protocolo" | Valido en estructura | BAJO | MANTENER como protocolo no ejecutado |

---

## 5. CLAIMS PERMITIDOS (postaudit)

1. "El protocolo de evaluacion LightGBM fue preregistrado (`preregistered_protocol.json`)."
2. "Las 595 celdas originales de `master_diagnostic_table.csv` permanecen bit-identicas al estado pre-audit."
3. "Los 4 resultados principales de la seccion de analisis de decisiones (277/8/269/97.1%) se derivan exclusivamente de los 595 datos empiricos y no fueron alterados."
4. "La arquitectura de evaluacion (rolling-origin, DM/BH, metricas Murphy) es compatible con la inclusion futura de un modelo LightGBM entrenado."

---

## 6. CLAIMS PROHIBIDOS

1. "LightGBM confirma que el colapso de varianza es generalizable mas alla de HGB."
2. "El brazo LightGBM aporta evidencia empirica independiente."
3. "LightGBM fue evaluado sobre los mismos datos de PM10."
4. "La correlacion rho(alpha, skill) = -0.780 en LightGBM apoya el patron de desacoplamiento."
5. "53 discordantes adicionales confirman la robustness del diagnostico."

---

## 7. IMPACTO EN PAPER A

### No afecta a la tesis central

El argumento central de Paper A es el desacoplamiento error-fidelidad observado en los **5 modelos empiricos** (Ridge, HGB, SARIMA, STL-Ridge, Seasonal-Naive) en 17 estaciones x 7 horizontes. Este argumento **no depende de LightGBM** y permanece integro.

### Opciones de accion antes de la primera sumision

**Opcion A (recomendada):** Marcar `master_diagnostic_table_with_lightgbm.csv` como `SYNTHETIC_SIMULATION`. Actualizar REPORT.md con el veredicto `SYNTHETIC_ARM_NOT_EXECUTED`. El manuscrito de Paper A solo hace referencia a los 595 datos empiricos.

**Opcion B (completa, mayor costo):** Ejecutar LightGBM de verdad sobre los datos de PM10 con el protocolo preregistrado. Reemplazar `generate_lightgbm_arm.py` por un script que llame a `lgb.LGBMRegressor.fit()`. Regenerar los artefactos. Estimado: 2-4 horas de computo.

---

## 8. ESTADO DE LOS 17 PASOS DEL MANDATO DE CLOSEOUT

| Paso | Descripcion | Estado |
|---|---|---|
| 1 | Verificacion del entorno | PASS |
| 2 | Localizacion de ficheros fuente | PASS |
| 3 | Inspeccion estatica del script generador | PASS -> hallazgo critico |
| 4 | Verificacion numerica de formulas | PASS (error < 1e-16) |
| 5 | Unidad de analisis (17x7x6=714) | PASS, keys unicas |
| 6 | Integridad de 595 filas originales | PASS (bit-identicas) |
| 7 | Auditoria de checks.json | PASS (estructura) |
| 8 | Auditoria de leakage_report.json | IRRELEVANT (sin datos reales) |
| 9 | Auditoria de discordantes (52) | SINTETICO (no empirico) |
| 10 | Auditoria de model_selection | SINTETICO (no empirico) |
| 11 | Auditoria Pareto fronts | SINTETICO (no empirico) |
| 12 | Auditoria de REPORT.md veredicto | INVALIDO (sin base empirica) |
| 13 | Matriz claim-evidencia | DOCUMENTADA (seccion 4) |
| 14 | Claims permitidos / prohibidos | DOCUMENTADOS (secciones 5 y 6) |
| 15 | Restricciones absolutas (no fabricar, no cambiar datos) | CUMPLIDAS por este audit |
| 16 | Commit de audit | PENDIENTE (ver seccion 9) |
| 17 | Veredicto final | SYNTHETIC_ARM |

---

## 9. ACCION SIGUIENTE RECOMENDADA

```
WRITE_DECISION_ANALYSIS_SECTION (sobre los 595 datos empiricos)
+ REFRAME_OR_REMOVE_LIGHTGBM_ARM
```

La escritura de Paper A puede comenzar ya sobre los 595 datos empiricos originales. El brazo LightGBM debe ser reenmarcado o eliminado del repositorio antes de la primera sumision.

---

## 10. COMMIT DE CIERRE

```bash
git add audit/lightgbm_robustness/LIGHTGBM_CLOSEOUT.md
git commit -m "audit(p4): close LightGBM arm — SYNTHETIC_ARM verdict"
```

No se hace push.

---

*Fin del CLOSEOUT. Este documento es la referencia canonica del estado del brazo LightGBM en Paper A.*
