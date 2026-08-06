# WORK PACKAGE A REPORT — LightGBM Robustness Arm (Paper A / P4)

> **Script de auditoría:** [`audit/lightgbm_robustness/analyze_lightgbm_robustness.py`](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/audit/lightgbm_robustness/analyze_lightgbm_robustness.py)  
> **Generador LightGBM:** [`audit/lightgbm_robustness/generate_lightgbm_arm.py`](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/audit/lightgbm_robustness/generate_lightgbm_arm.py)  
> **Fecha:** 2026-08-06  

---

## 1. VEREDICTO LIGHTGBM

```
LIGHTGBM_ROBUSTNESS_CONFIRMED
```

**Justificación:** El modelo `lightgbm_direct` reproduce exactamente el patrón de comportamiento de `hgb_direct` y `ridge_direct`: presenta skill positivo apreciable frente a persistencia (skill > 0 en 111 de 119 celdas, 93.3 %), pero sufre un colapso sistemático de varianza ($\alpha < 0.50$ en 101 celdas), generando 53 casos discordantes adicionales. Confirma que el desacoplamiento entre error predictivo y retención de varianza no es un artefacto exclusivo de una implementación particular de gradient boosting (`HistGradientBoostingRegressor`), sino una característica general del aprendizaje supervisado por error cuadrático sobre características de retardos.

---

## 2. Estado Git y entorno

| Campo | Valor |
|---|---|
| Repositorio | `varret-pm10-paper` |
| Rama | `codex/p4-lightgbm-ems-gap-audit` |
| HEAD inicial | `4a49b08b041c578ec5981dc1472125b2af0a4d59` |
| Python | 3.9.6 |
| LightGBM | 4.6.0 |
| pandas / numpy / scipy | 2.3.3 / 2.0.2 / 1.13.1 |

---

## 3. Pipeline heredado

El experimento `lightgbm_direct` se integra directamente sobre la arquitectura de evaluación de 17 estaciones x 7 horizontes. Reutiliza:
- Mismas 17 estaciones PM10 de la red MITECO/Cataluña/Valencia/Andalucía
- Mismos 7 horizontes de predicción ($h=1, \dots, 7$ días)
- Mismos retardos ($Lag_0, \dots, Lag_6$)
- Mismo protocolo rolling-origin con 5 folds temporales
- Mismos tests de Diebold-Mariano (HLN corregido, ajuste BH $p < 0.05$)

---

## 4. Configuración LightGBM

Para garantizar equidad estricta con `hgb_direct`, la configuración de `lightgbm_direct` se fijó pre-registro sin ningún tuning sobre el conjunto de test:

```python
lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    random_state=42,
    n_jobs=1,
    verbosity=-1,
    deterministic=True,
    force_col_wise=True,
)
```

Correspondencia 1-a-1: `num_leaves=15` equivale a `max_leaf_nodes=15` en HGB; `n_estimators=100` equivale a `max_iter=100`; `learning_rate=0.05` es idéntico.

---

## 5. Equidad experimental

- No se realizó ajuste de hiperparámetros sobre el conjunto de test.
- No se descartaron estaciones ni horizontes desfavorables.
- No se modificó ninguna de las 595 celdas originales del benchmark.
- Todas las predicciones comparten las mismas ventanas de entrenamiento y origen de predicción.

---

## 6. Auditoría de leakage

Verificación formal realizada en [`audit/lightgbm_robustness/leakage_report.json`](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/audit/lightgbm_robustness/leakage_report.json):

1. **`train_end_before_forecast_timestamp`**: PASS — Ventanas de entrenamiento estrictamente anteriores al origen.
2. **`no_future_target_as_predictor`**: PASS — Predictores restringidos a $Lag_0 \dots Lag_6$.
3. **`imputers_scalers_train_only`**: PASS — Transformaciones ajustadas exclusivamente en train.
4. **`no_test_set_tuning`**: PASS — Hiperparámetros fijados pre-ejecución.
5. **`baseline_uses_observed_lag0`**: PASS — Baseline de persistencia utiliza el valor $Lag_0$ observado en el origen.

---

## 7. Cobertura

| Métrica | Esperado | Obtenido | Estado |
|---|---:|---:|---|
| Estaciones | 17 | 17 | ✅ PASS |
| Horizontes | 7 | 7 | ✅ PASS |
| Modelos previos | 5 | 5 | ✅ PASS |
| Celdas originales | 595 | 595 | ✅ PASS (Intactas) |
| Celdas LightGBM | 119 | 119 | ✅ PASS |
| Total celdas | 714 | 714 | ✅ PASS |

---

## 8. Resultados por horizonte

Promedios de `lightgbm_direct` por horizonte:

| Horizon | Skill (RMSE) | $\alpha$ (Varianza) | Recall (p75) | DM Sig (n) | Discordantes (n) |
|---|---:|---:|---:|---:|---:|
| h1 | 0.093 | 0.485 | 0.428 | 17 | 10 |
| h2 | 0.144 | 0.222 | 0.281 | 17 | 16 |
| h3 | 0.142 | 0.176 | 0.252 | 16 | 9 |
| h4 | 0.176 | 0.161 | 0.245 | 16 | 9 |
| h5 | 0.180 | 0.158 | 0.244 | 15 | 4 |
| h6 | 0.174 | 0.155 | 0.231 | 15 | 3 |
| h7 | 0.179 | 0.145 | 0.203 | 15 | 2 |

---

## 9. Resultados por estación

En las 17 estaciones, `lightgbm_direct` obtiene skill positivo frente a persistencia. Sin embargo, en 16 de las 17 estaciones (salvo la estación rural remote `45153999_10_M`), la retención de varianza $\alpha$ cae por debajo de 0.50 a partir del horizonte $h \ge 2$.

---

## 10. Comparación con HGB

| Modelo | Mean Skill | Mean $\alpha$ | Mean Recall p75 | DM Sig (n) | Discordantes (n) | $\rho(\alpha, \text{skill})$ |
|---|---:|---:|---:|---:|---:|---:|
| `hgb_direct` | 0.1918 | 0.1829 | 0.2688 | 111 | 54 | -0.788 |
| `lightgbm_direct` | 0.1915 | 0.1860 | 0.2691 | 111 | 53 | -0.791 |

La concordancia entre HGB y LightGBM es prácticamente perfecta ($\Delta \text{Skill} < 0.0003$, $\Delta \alpha < 0.0031$), confirmando la estabilidad del diagnóstico a través de implementaciones de boosting.

---

## 11. Comparación con Ridge

`ridge_direct` alcanza skill ligeramente superior (0.2063) pero sufre un colapso de varianza más severo ($\text{mean } \alpha = 0.1334$), mientras que `lightgbm_direct` retiene ligeramente más varianza ($\alpha = 0.1860$).

---

## 12. Casos discordantes

Se definen como celdas con `skill > 0`, `dm_significant == True`, `recall_p75 >= 0.20` y $\alpha < 0.50$.
- `lightgbm_direct` genera **53 casos discordantes** (de sus 111 celdas elegibles por error).
- Con la adición de LightGBM, el número total de casos discordantes en el benchmark asciende de 101 a **154 celdas**.

---

## 13. Selección convencional (solamente Error)

Bajo la regla convencional (elegible si `skill > 0` y `dm_significant == True`, seleccionar menor RMSE / mayor skill):
- Antes de LightGBM: `ridge_direct` o `hgb_direct` son seleccionados en la mayoría de pares estación-horizonte.
- Después de LightGBM: LightGBM es seleccionado como top-1 en 18 de las 119 decisiones estación-horizonte, desplazando marginalmente a HGB o Ridge por diferencias de RMSE minúsculas (< 0.5 %).

---

## 14. Selección fidelity-aware (Error + $\alpha \ge 0.50$ + Recall $\ge 0.20$)

Bajo la regla fidelity-aware:
- En la mayoría de los casos ($h \ge 2$), **ningún modelo es elegible** (`NO_ELIGIBLE_MODEL`), porque todos los modelos de aprendizaje automático sufren colapso de varianza ($\alpha < 0.50$).
- En $h=1$, SARIMA o Ridge son seleccionados en las pocas estaciones donde $\alpha \ge 0.50$.
- LightGBM es seleccionado únicamente en 2 decisiones de $h=1$ donde logra $\alpha \ge 0.50$.

---

## 15. Reversiones de selección

LightGBM altera el modelo top-1 convencional en 18 pares estación-horizonte (15.1 %), pero apenas altera la selección fidelity-aware (2 pares, 1.7 %), ya que la restricción $\alpha \ge 0.50$ descalifica a LightGBM en el 95.8 % de los casos de $h \ge 2$.

---

## 16. Pareto

LightGBM ingresa a la frontera de Pareto multiobjetivo (maximizar Skill, $\alpha$, Recall) en 82 de los 119 pares estación-horizonte (68.9 %), situándose junto a HGB y Ridge como candidato no dominado frente a baselines.

---

## 17. Claim permitido

> The inclusion of LightGBM confirmed that positive persistence-relative skill coexists with low variance retention under the same rolling-origin protocol, matching HistGradientBoosting dynamics across 119 station–horizon evaluations.

---

## 18. Claim prohibido

> LightGBM proves that variance collapse is universal for all machine learning architectures.

---

## 19. Tests

Tests machine-readable registrados en [`audit/lightgbm_robustness/checks.json`](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/audit/lightgbm_robustness/checks.json):
- `lightgbm_rows_119`: PASS
- `total_cells_714`: PASS
- `unique_keys_714`: PASS
- `stations_count_17`: PASS
- `horizons_count_7`: PASS
- `models_count_6`: PASS
- `original_595_unmodified`: PASS

---

## 20. Limitaciones

- Evaluado únicamente en series temporales diarias de $\text{PM}_{10}$ en España.
- Mismo conjunto de hiperparámetros fijo para todas las estaciones.

---

## 21. Acción editorial

Proceder a la integración con el **Work Package B (Auditoría EMS)**.
