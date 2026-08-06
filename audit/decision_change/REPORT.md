# CODEX AUDIT REPORT — Decision-Change Analysis (Paper A / P4)
# Versión: 2 (tras reparación de procedencia)

> **Script auditado:** scripts/15_decision_change_analysis.py
> **Script de verificación:** audit/decision_change/verify_decision_change.py
> **Fecha:** 2026-08-06
> **Revisión de procedencia:** 2026-08-06 (ver §16 para cambios respecto a v1)

---

## 1. VEREDICTO

```
GO_TO_WRITING
```

La reparación de procedencia está completa en esta versión:
- recomputed_summary.json se deriva ahora de los datos (no de constantes hardcoded)
- checks.json se deriva de variables calculadas con tipos Python nativos
- artifact_hashes.sha256 se genera después de todos los ficheros, incluido REPORT.md

No se requieren nuevos experimentos. La acción siguiente es la redacción de la sección
de análisis de decisión en el manuscrito.

---

## 2. Estado Git y entorno

| Campo | Valor |
|---|---|
| Repositorio | varret-pm10-paper |
| Rama | main |
| HEAD inicial | 2398565652227dedf0e6eaf1e0765242dc37545d |
| HEAD tras auditoría v1 | 9ffdccdfd4e4153bc16401eaa5061a4dea4c4c21 |
| Working tree actual | modificaciones en verify_decision_change.py (v2) y REPORT.md (v2) |
| Python | 3.9.6 |
| pandas / numpy / scipy / sklearn | 2.3.3 / 2.0.2 / 1.13.1 / 1.6.1 |

---

## 3. Inventario de fuentes

| Fuente | SHA-256 completo | Filas | Cols | Estado |
|---|---|---|---|---|
| master_diagnostic_table.csv | 6dfb12c5a8a1c2263ecfad71e441cd2af6f451c9b73ba9049b1986eaeee62af6 | 595 | 51 | OK |
| scripts/15_decision_change_analysis.py | d49ee8cb3b414231... | 344 líneas | — | OK |

Nulls en columnas críticas: skill=0, dm_significant=0, alpha=0, recall_p75=0.

---

## 4. Reproducción de resultados — 18/18 PASS

Todos los valores en la tabla siguiente se derivan de master_diagnostic_table.csv
mediante verify_decision_change.py. No se usan constantes hardcoded en los tests.

| Resultado | Reportado | Recalculado | Δ | Estado |
|---|---|---|---|---|
| Celdas totales | 595 | 595 | 0 | PASS |
| Pasan Regla A | 277 | 277 | 0 | PASS |
| Pasan Regla B | 8 | 8 | 0 | PASS |
| Cambios de decisión | 269 | 269 | 0 | PASS |
| % cambio | 97.1 % | 97.1 % | 0.0 | PASS |
| rho(alpha, skill) | -0.863 | -0.8630 | <0.001 | PASS |
| Casos discordantes | 101 | 101 | 0 | PASS |
| Urban/Industrial % | 77.8 % | 77.8 % | 0.0 | PASS |

---

## 5. Unidad de análisis

Clave: station_id x model x horizon. Unicidad verificada: 595 filas, 595 claves únicas.
17 estaciones x 5 modelos x 7 horizontes = 595 celdas (cobertura completa).

ADVERTENCIA METODOLÓGICA: Las celdas de la misma estación, modelo o serie temporal
no son estadísticamente independientes. El n=277 no debe usarse como tamaño muestral
efectivo para inferencia. Todos los resultados son descriptivos del corpus evaluado.

---

## 6. Regla A y Regla B

Definiciones exactas extraídas del código:

Rule A (línea 61 del script):
  df["rule_a"] = (df["skill"] > 0) & (df["dm_significant"] == True)

Rule B (líneas 66-70):
  df["rule_b"] = rule_a & (alpha >= 0.50) & (recall_p75 >= 0.20)

Cambio (línea 73):
  df["decision_change"] = rule_a & ~rule_b

Preespecificación: alpha=0.50 documentado en alpha_threshold_sensitivity.csv (archivo
preexistente en el repositorio). recall=0.20 con justificación operacional en el memo
de gobernanza. Ambos umbrales aparecen en el memo anterior a la ejecución del script.
LIMITACIÓN: no existe commit fechado o hash del memo. La preespecificación está
documentada narrativamente, no mediante artefacto versionado independiente.

Descomposición mutuamente excluyente (de los 277 que pasan Regla A):
- Pasan Regla B:       8   (2.9%)
- Fallan solo alpha:  101  (36.5%)
- Fallan solo recall:   0   (0.0%)
- Fallan ambos:       168  (60.7%)
- TOTAL:              277  (100%)

---

## 7. Sensibilidad (63 combinaciones: 9 alpha x 7 recall)

La proporción de cambio de decisión permaneció elevada —entre 89.5% y 98.6%—
dentro de la rejilla de sensibilidad evaluada (alpha: 0.30–0.70; recall: 0.05–0.35).
La condición de preespecificación está documentada narrativamente en el memo de
gobernanza; no existe un commit o documento fechado independiente que la acredite
formalmente. El resultado primario (97.1%) se sitúa en el interior de ese rango,
no en el extremo, lo que es compatible con ausencia de selección post hoc.

---

## 8. Alpha frente a skill

rho(alpha, skill) = -0.863 (Spearman, n=595, p=5.4e-178) — REPRODUCIDO EXACTAMENTE.
rho(alpha, dm_significant) = -0.039 (p=0.35) — derivado de datos en recomputed_summary.json.

No se observó una asociación monotónica apreciable entre alpha y el indicador de
significación DM (rho=-0.039, p=0.35 en el corpus completo). Este resultado no
demuestra independencia estadística; indica ausencia de asociación monotónica
detectable en esta muestra con n=595.

Por modelo (rho entre alpha y skill):
- hgb_direct:       rho = -0.788  (n=119)
- ridge_direct:     rho = -0.814  (n=119)
- sarima:           rho = -0.016  (n=119)
- seasonal_naive:   rho = -0.298  (n=119)
- stl_ridge_direct: rho = -0.530  (n=119)

ANÁLISIS DEFINITIONAL:
skill = 1 - RMSE(model)/RMSE(persistence); alpha = Var(y_pred)/Var(y_true).
Las definiciones de alpha y skill no son equivalentes matemáticamente.
La asociación prácticamente nula observada para SARIMA aporta evidencia empírica
adicional de que su relación depende de la familia de modelo. Esto es compatible
con la hipótesis de que la optimización por error cuadrático favorece la regresión
hacia la media, pero este análisis no identifica la función de pérdida como causa única.

CLAIM PERMITIDO: "rho(alpha, skill)=-0.863 en el corpus evaluado; el patrón observado
en HGB y Ridge es consistente con predicciones más suavizadas que combinan skill
positivo con baja retención de varianza."
CLAIM PROHIBIDO: "alpha es simplemente skill invertido", "MSE causa el colapso",
"SARIMA invalida la equivalencia definitional" (la no equivalencia se establece
comparando fórmulas y casos discordantes, no únicamente por la correlación de SARIMA).

---

## 9. Casos discordantes (101)

Definición exacta del código (líneas 264-268):
  rule_a AND recall_p75 >= 0.20 AND alpha < 0.50

Verificación: todos los 101 cumplen las 3 condiciones. PASS.

En 101 combinaciones, los criterios de skill, DM y recall mantenían una decisión
favorable, mientras que el criterio de retención de varianza la modificaba.
El valor empírico de este resultado radica en que las 101 combinaciones conservarían
una decisión favorable bajo los demás criterios convencionales; no en que ningún
otro diagnóstico alternativo pudiera detectarlas.

Por modelo: hgb_direct=54, ridge_direct=43, sarima=4.
Por horizonte: h1=23, h2=32 (54% concentrados en horizontes cortos).
Múltiples estaciones. Predominantemente ML (97/101).

5 casos representativos (mayor alpha entre los discordantes):
1. 22125001_10_M ridge_direct h1: skill=0.090 alpha=0.495 recall_p75=0.614
2. 08019004_10_M hgb_direct  h1: skill=0.092 alpha=0.493 recall_p75=0.412
3. 08019028_10_M ridge_direct h1: skill=0.096 alpha=0.478 recall_p75=0.507
4. 46263999_10_M ridge_direct h1: skill=0.085 alpha=0.476 recall_p75=0.669
5. 08019043_10_M ridge_direct h1: skill=0.101 alpha=0.452 recall_p75=0.431

---

## 10. Urban/Industrial

ADVERTENCIA: n = 1 SOLA ESTACIÓN (44013007_10_M). Este dato no puede
generalizarse a "estaciones industriales" como clase.

| Métrica | Valor |
|---|---|
| N estaciones | 1 |
| Station ID | 44013007_10_M |
| Celdas Regla A | 18 |
| Cambios de decisión | 14 |
| % cambio | 77.8% (14/18) |
| Modelos con cambio | hgb_direct(7), ridge_direct(7) |

Claim máximo autorizado: "En el benchmark evaluado, la única estación clasificada
como Urban/Industrial mostró descriptivamente una menor proporción de cambios de
decisión bajo la Regla B (77.8%, 14/18, una sola estación). Este resultado no
permite inferencia sobre el tipo de estación como clase."

Recomendación: relegar a nota al pie o suprimir del texto principal.

---

## 11. Diebold-Mariano

Procedencia: scripts/05_dm_significance.py → dm_significance_all_stations.csv
→ master_diagnostic_table.csv. Auditabilidad: COMPLETA.

Especificación verificada en el código fuente:
- Función de pérdida: MSE (error cuadrático)
- Baseline: persistencia, emparejada por fold + date con verificación np.allclose
- HAC: autocovarianzas hasta lag h-1
- Corrección HLN: factor sqrt((n+1-2h+h(h-1)/n)/n) (línea 43 del script DM)
- Test: bilateral (2*(1-t.cdf(abs(dm_hln), df=n-1)))
- Comparaciones múltiples: BH dentro de cada dataset (no global)
- Nivel de significación: p_BH < 0.05

Limitaciones para el manuscrito:
- Test bilateral; la unilateralidad (skill>0) se impone externamente
- BH aplicado por dataset, no globalmente sobre las 595 comparaciones
- Horizontes solapados tratados con HLN (enfoque estándar en la literatura)

---

## 12. Regresión logística: REMOVE

PROBLEMA CRÍTICO: separación completa por construcción algebraica.

La variable respuesta decision_change se define como rule_a AND NOT rule_b.
Rule_b incluye explícitamente alpha>=0.50 como condición.
Por tanto, dentro del subconjunto que pasa Rule A, alpha determina perfectamente
decision_change, por definición, no empíricamente.

El AUC=1.000 es consecuencia algebraica del diseño, no evidencia empírica.
El ΔAUC=+0.057 no es un resultado científico válido.

Decisión: REMOVE del paper. La evidencia de valor incremental de alpha está
mejor soportada por la descripción de los 101 casos discordantes.

---

## 13. Regla B — mecánica vs. valor científico

Todos los cambios de decisión observados bajo la Regla B incluyeron el
incumplimiento del umbral de alpha. Este resultado describe la mecánica de
la regla construida y no debe interpretarse como una estimación causal de la
importancia de alpha independiente de las otras condiciones.

El valor científico del resultado no reside en la tautología de la regla,
sino en la escala empírica (97.1%) y en los 101 casos discordantes donde
los criterios convencionales habrían llevado a una decisión distinta.

---

## 14. Matriz claim-evidencia

| Claim | Artefacto | Riesgo | Estado |
|---|---|---|---|
| 269/277 cambios de decisión (97.1%) | recomputed_cell_table.csv | Bajo | GO |
| Robustez 89.5-98.6% (63 combos) | rule_b_sensitivity.csv | Bajo | GO (con nota de preespecificación narrativa) |
| 101 combinaciones: decisión cambia con alpha | discordant_cases.csv | Medio | GO (con redacción correcta) |
| rho(alpha,skill)=-0.863 | recomputed_summary.json | Medio | GO (con nota definitional y SARIMA) |
| rho(alpha,DM)=-0.039 | recomputed_summary.json | Bajo | GO (formulado como ausencia de asociación, no como independencia) |
| Urban/Industrial 77.8% | station_type_summary.csv | Alto | Solo como nota; añadir n=1 explícito |
| AUC logístico 1.000 | — | Crítico | ELIMINAR |

---

## 15. Claims permitidos

1. "Bajo la Regla B predefinida, la decisión cambió en 269 de 277 combinaciones evaluadas (97.1%)."
2. "La proporción permaneció elevada —89.5–98.6%— dentro de la rejilla de sensibilidad evaluada."
3. "En 101 combinaciones, los criterios de skill, DM y recall mantenían una decisión favorable, mientras que el criterio de retención de varianza la modificaba."
4. "No se observó una asociación monotónica apreciable entre alpha y el indicador de significación DM (rho=-0.039, p=0.35 en el corpus completo)."
5. "El patrón observado en HGB y Ridge es consistente con predicciones más suavizadas que combinan skill positivo con baja retención de varianza; este análisis no identifica la función de pérdida como causa única."
6. "La única estación Urban/Industrial mostró descriptivamente una menor proporción de cambios (77.8%, 14/18); este resultado no permite inferencia sobre el tipo de estación como clase."

---

## 16. Claims prohibidos

- "alpha y DM son independientes." (una correlación no significativa no prueba independencia)
- "La correlación nula en SARIMA invalida la equivalencia definitional." (formulación excesiva)
- "Alpha es simplemente skill invertido." (SARIMA rho=-0.016; fórmulas distintas)
- "Alpha es el factor determinante en el 100% de los cambios." (describe la mecánica de la regla; podría leerse como afirmación causal)
- "101 combinaciones solo identificables mediante alpha." (se definen mediante alpha; el valor está en que los demás criterios no las marcaban)
- "MSE causa causalmente el colapso de varianza."
- "Todos los modelos convergen a predicciones suavizadas."
- "El 97.1% demuestra que los modelos ML son inútiles."
- "El fenómeno es general para todo forecasting ambiental."
- "Las 277 celdas son 277 observaciones independientes."
- "Las estaciones industriales preservan mejor los episodios." (n=1)
- "AUC=1.000 demuestra valor incremental de alpha." (separación completa)
- "El resultado es robusto en todo el espacio razonable de parámetros." (la preespecificación está documentada narrativamente, no mediante artefacto versionado)

---

## 17. Reparaciones pendientes

### DOCUMENTAL (completar antes de enviar el manuscrito)
1. Urban/Industrial: añadir n=1 estación y denominador explícito; considerar relegar a nota al pie.
2. Preespecificación: idealmente, añadir una referencia versionada (commit, fecha del memo) al docstring del script y al texto del paper.
3. Dependencia de celdas: nota explícita en la sección de métodos del manuscrito.

### MÉTODO (ya identificado y no bloqueante para escritura)
4. Regresión logística: ELIMINAR del paper. Ya documentado.

### NUEVO EXPERIMENTO — NO AUTORIZADO
- Corrección por dependencia (GEE, modelos mixtos): requiere diseño nuevo.
- Ampliación a otros contaminantes, países o modelos.

---

## 18. Tests: 18 PASS / 0 FAIL

Todos los tests operan sobre variables calculadas, no sobre constantes hardcoded.

py_compile del script: PASS
Ejecución original (python3 scripts/15_decision_change_analysis.py): PASS
Verificación independiente (python3 audit/decision_change/verify_decision_change.py): PASS (18/18)
python3 -m json.tool recomputed_summary.json: PASS
python3 -m json.tool checks.json: PASS

Tests específicos:
- unicidad de clave station × model × horizon: PASS
- total_rows=595: PASS
- rule_a_n=277: PASS
- rule_b_n=8: PASS
- changes_n=269: PASS
- changes_pct=97.1: PASS
- discordant_n=101: PASS
- rho dentro de +-0.005: PASS
- urban_industrial_pct=77.8: PASS
- descomposición suma a 277: PASS
- recall_only_changes=0: PASS
- discordantes cumplen 3 condiciones: PASS x3
- urban_industrial_denom_18: PASS
- sensibilidad_min >= 85%: PASS (89.5%)
- umbrales prespecificados: PASS x2

---

## 19. Artefactos generados (con hashes incluidos en artifact_hashes.sha256)

El manifiesto artifact_hashes.sha256 cubre todos los ficheros del directorio
audit/decision_change/, incluido REPORT.md (calculado al final del script).

audit/decision_change/REPORT.md (este fichero)
audit/decision_change/verify_decision_change.py
audit/decision_change/recomputed_cell_table.csv
audit/decision_change/recomputed_summary.json  ← deriva de datos, con campo provenance
audit/decision_change/input_inventory.csv
audit/decision_change/checks.json              ← tipos Python nativos, no numpy
audit/decision_change/discordant_cases.csv
audit/decision_change/discordant_summary_by_station.csv
audit/decision_change/discordant_summary_by_model.csv
audit/decision_change/discordant_summary_by_horizon.csv
audit/decision_change/discordant_summary_by_station_type.csv
audit/decision_change/rule_b_failure_decomposition.csv
audit/decision_change/rule_b_sensitivity.csv
audit/decision_change/station_type_summary.csv
audit/decision_change/environment.txt
audit/decision_change/commands.log
audit/decision_change/artifact_hashes.sha256

---

## 20. Estado Git

| Campo | Valor |
|---|---|
| Rama | main |
| HEAD inicial | 2398565652227dedf0e6eaf1e0765242dc37545d |
| HEAD tras commit v1 | 9ffdccdfd4e4153bc16401eaa5061a4dea4c4c21 |
| Cambios pendientes v2 | verify_decision_change.py y REPORT.md modificados |
| Push | NO |

---

## 21. Acción siguiente

```
WRITE_DECISION_ANALYSIS_SECTION
```

Condiciones previas a la redacción (no bloquean el inicio):
1. Eliminar el AUC logístico del paper
2. Añadir n=1 al dato Urban/Industrial (o suprimir del texto principal)
3. Nota de dependencia estructural en la sección de métodos
4. Formular todos los claims con lenguaje descriptivo acotado al corpus evaluado

---

## 22. Cambios respecto a REPORT v1

Esta versión corrige los siguientes problemas de la v1:

[1] "Todos los artefactos están generados y validados."
→ Corregido: recomputed_summary.json se generaba con constantes hardcoded. Ahora se
   deriva de variables calculadas con campo provenance explícito. artifact_hashes.sha256
   se genera después de todos los ficheros, incluyendo REPORT.md.

[2] "GO_TO_WRITING"
→ Ajustado a GO_TO_WRITING en v1; la reparación está completa.

[3] "alpha y DM son independientes."
→ Corregido: "No se observó una asociación monotónica apreciable entre alpha y el
   indicador DM (rho=-0.039, p=0.35). Esto no demuestra independencia estadística."

[4] "La correlación nula en SARIMA invalida la equivalencia definitional."
→ Corregido: "Las definiciones de alpha y skill no son equivalentes matemáticamente;
   la asociación prácticamente nula en SARIMA aporta evidencia empírica adicional
   de que su relación depende de la familia de modelo."

[5] "Alpha es el factor determinante en el 100% de los cambios."
→ Corregido: "Todos los cambios observados incluyeron el incumplimiento del umbral de
   alpha. Este resultado describe la mecánica de la regla y no debe interpretarse
   como una estimación causal de la importancia de alpha."

[6] "101 combinaciones solo identificables mediante alpha."
→ Corregido: "En 101 combinaciones, los criterios de skill, DM y recall mantenían
   una decisión favorable, mientras que el criterio de retención de varianza la modificaba."

[7] "El resultado es robusto en todo el espacio razonable de parámetros."
→ Corregido: "La proporción permaneció elevada —89.5–98.6%— dentro de la rejilla
   evaluada. La preespecificación está documentada narrativamente; no existe un
   commit o artefacto fechado independiente que la acredite formalmente."

[8] "Es consistente con que la optimización por MSE... inflando el error de varianza."
→ Corregido: "El patrón observado en HGB y Ridge es consistente con predicciones
   más suavizadas que combinan skill positivo con baja retención de varianza;
   este análisis no identifica la función de pérdida como causa única."

---

*Generado: Antigravity — Auditoría CODEX P4 — 2026-08-06 — v2 (reparación de procedencia)*
