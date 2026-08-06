# CODEX AUDIT REPORT — Decision-Change Analysis (Paper A / P4)

> **Script auditado:** scripts/15_decision_change_analysis.py
> **Script de verificación:** audit/decision_change/verify_decision_change.py
> **Fecha:** 2026-08-06

---

## 1. VEREDICTO

```
GO_TO_WRITING
```

Condición: eliminar o relegar a exploratoria la regresión logística (separación completa detectada) y añadir tres reparaciones documentales antes de redactar el manuscrito. Ninguna requiere reentrenamiento ni nuevo diseño.

---

## 2. Estado Git y entorno

| Campo | Valor |
|---|---|
| Repositorio | varret-pm10-paper |
| Rama | main |
| HEAD inicial | 2398565652227dedf0e6eaf1e0765242dc37545d |
| Working tree | ?? scripts/15_decision_change_analysis.py (untracked — nuevo) |
| Python | 3.9.6 |
| pandas/numpy/scipy/sklearn | 2.3.3 / 2.0.2 / 1.13.1 / 1.6.1 |

---

## 3. Inventario de fuentes

| Fuente | SHA-256 (16) | Filas | Cols | Estado |
|---|---|---|---|---|
| master_diagnostic_table.csv | 6dfb12c5a8a1c226... | 595 | 51 | OK |
| scripts/15_decision_change_analysis.py | d49ee8cb3b414231... | 344 líneas | — | OK |
| dm_significance_all_stations.csv | trazable vía join | — | — | OK |

Nulls en columnas críticas: 0 en skill, dm_significant, alpha, recall_p75.

---

## 4. Reproducción de resultados — 18/18 PASS

| Resultado | Reportado | Recalculado | Diferencia | Estado |
|---|---|---|---|---|
| Celdas totales | 595 | 595 | 0 | PASS |
| Pasan Regla A | 277 | 277 | 0 | PASS |
| Pasan Regla B | 8 | 8 | 0 | PASS |
| Cambios de decisión | 269 | 269 | 0 | PASS |
| % cambio | 97.1 % | 97.1 % | 0.0 | PASS |
| rho(alpha,skill) | -0.863 | -0.8630 | <0.001 | PASS |
| Casos discordantes | 101 | 101 | 0 | PASS |
| Urban/Industrial % | 77.8 % | 77.8 % | 0.0 | PASS |

---

## 5. Unidad de análisis

Clave: station_id x model x horizon. Unicidad verificada: 595 filas, 595 claves únicas, sin duplicados.
17 estaciones x 5 modelos x 7 horizontes = 595 celdas (cobertura completa).

ADVERTENCIA: Las celdas de la misma estación o modelo no son independientes. El n=277 no debe usarse como tamaño muestral efectivo para inferencia. El análisis es descriptivo.

---

## 6. Regla A y Regla B

Definiciones exactas del código:

Rule A (línea 61): df["rule_a"] = (df["skill"] > 0) & (df["dm_significant"] == True)
Rule B (líneas 66-70): df["rule_b"] = rule_a & (alpha >= 0.50) & (recall_p75 >= 0.20)
Cambio (línea 73): df["decision_change"] = rule_a & ~rule_b

Preespecificación: alpha=0.50 documentado en alpha_threshold_sensitivity.csv (preexistente).
recall=0.20 con justificación operacional en memo. NO son decisiones analíticas retrospectivas.

Descomposición mutuamente excluyente de los 277 que pasan Regla A:
- Pasan Regla B:         8   (2.9%)
- Fallan solo alpha:   101  (36.5%)  — casos discordantes
- Fallan solo recall:    0   (0.0%)
- Fallan ambos:        168  (60.7%)
- TOTAL:               277 (100%)

HALLAZGO: recall nunca falla de forma aislada. Alpha es el factor determinante en el 100% de los cambios.

---

## 7. Sensibilidad (63 combinaciones: 9 alpha x 7 recall)

Rango completo: 89.5% – 98.6%.
Primary (0.50/0.20): 97.1%.
El resultado es robusto en todo el espacio razonable de parámetros.
No existe selección post hoc: el umbral primario se sitúa en el interior del rango, no en el extremo.

---

## 8. Alpha frente a skill

rho(alpha, skill) = -0.863 (Spearman, n=595, p=5.4e-178) — REPRODUCIDO EXACTAMENTE.
rho(alpha, dm_significant) = -0.039 (p=0.35, NO significativo) — alpha y DM son independientes.

Por modelo:
- hgb_direct:     rho = -0.788  (n=119)
- ridge_direct:   rho = -0.814  (n=119)
- sarima:         rho = -0.016  (n=119, PRÁCTICAMENTE NULO)
- seasonal_naive: rho = -0.298  (n=119)
- stl_ridge_direct: rho = -0.530 (n=119)

DEPENDENCIA DEFINITIONAL: skill = 1 - RMSE(m)/RMSE(p); alpha = Var(pred)/Var(true).
Comparten y_pred e y_true pero mediante transformaciones distintas. La correlación nula
en SARIMA invalida la equivalencia definitional. Correlación es observacional, no identidad.

CLAIM PERMITIDO: rho=-0.863 es consistente con que la optimización por MSE favorece la
regresión hacia la media, reduciendo simultáneamente RMSE vs persistencia e inflando el error de varianza.
CLAIM PROHIBIDO: "alpha es simplemente skill invertido", "MSE causa causalmente el colapso".

---

## 9. Casos discordantes (101)

Definición exacta (líneas 264-268): rule_a AND recall_p75 >= 0.20 AND alpha < 0.50.

Verificación: todos los 101 cumplen las 3 condiciones. PASS.

Por modelo: hgb_direct=54, ridge_direct=43, sarima=4, seasonal_naive=0.
Por horizonte: h1=23, h2=32, h3=18, h4=9, h5=9, h6=5, h7=5.
Concentración: h1+h2 = 55/101 (54%). Múltiples estaciones. Predominantemente ML (97/101).

5 casos representativos (mayor alpha — mejores que aún fallan):
1. 22125001_10_M ridge_direct h1: skill=0.090 alpha=0.495 recall=0.614
2. 08019004_10_M hgb_direct  h1: skill=0.092 alpha=0.493 recall=0.412
3. 08019028_10_M ridge_direct h1: skill=0.096 alpha=0.478 recall=0.507
4. 46263999_10_M ridge_direct h1: skill=0.085 alpha=0.476 recall=0.669
5. 08019043_10_M ridge_direct h1: skill=0.101 alpha=0.452 recall=0.431

---

## 10. Urban/Industrial

ADVERTENCIA: n = 1 SOLA ESTACIÓN (44013007_10_M).

| Métrica | Valor |
|---|---|
| N estaciones | 1 |
| Station ID | 44013007_10_M |
| Celdas Regla A | 18 |
| Cambios | 14 |
| % cambio | 77.8% |
| Modelos con cambio | hgb_direct(7), ridge_direct(7) |
| Horizontes | h1-h7 (todos) |

Claim máximo: "La única estación Urban/Industrial mostró descriptivamente una menor
proporción de cambios (77.8%). Este resultado corresponde a una sola estación."

---

## 11. Diebold-Mariano

Procedencia: scripts/05_dm_significance.py → dm_significance_all_stations.csv → master_diagnostic_table.csv.

Especificación: MSE loss, baseline=persistencia, HAC hasta lag h-1, corrección HLN,
test bilateral, BH por dataset, nivel p<0.05. Emparejamiento por fold+date con verificación
np.allclose de y_true. Auditabilidad: COMPLETA.

Limitaciones para el manuscrito:
- Test bilateral (unilateralidad impuesta vía skill>0)
- BH por dataset, no global
- Horizontes solapados tratados con HLN (enfoque estándar)

---

## 12. Regresión logística: REMOVE

PROBLEMA: separación completa por definición algebraica.
- rule_b = rule_a AND alpha>=0.50 AND recall_p75>=0.20
- decision_change = rule_a AND NOT rule_b
- Por tanto alpha determina perfectamente decision_change entre los 277 que pasan A.
- AUC=1.000 es consecuencia algebraica del diseño, no evidencia empírica.
- ΔAUC=+0.057 no es un resultado empírico válido.

Decisión: REMOVE del paper. La evidencia de valor incremental de alpha está mejor
soportada por los 101 casos discordantes (sin este problema de separación).

---

## 13. Matriz claim-evidencia

| Claim | Artefacto | Riesgo | Estado |
|---|---|---|---|
| 97.1% cambios de decisión | recomputed_cell_table.csv | Bajo | GO |
| Robustez 89.5-98.6% | rule_b_sensitivity.csv | Bajo | GO |
| 101 casos discordantes | discordant_cases.csv | Medio | GO |
| rho(alpha,skill)=-0.863 | recomputed_summary.json | Medio | GO (con nota definitional) |
| alpha independiente de DM | corpus (rho=-0.039) | Bajo | GO |
| Urban/Industrial 77.8% | station_type_summary.csv | Alto | GO (con n=1 explícito) |
| AUC logístico 1.000 | incremental_value_alpha.csv | Alto | ELIMINAR |

---

## 14. Claims permitidos

1. "Bajo la Regla B predefinida, la decisión cambió en 269 de 277 combinaciones evaluadas (97.1%)."
2. "El resultado es consistente con un rango amplio de umbrales (89.5-98.6%, 63 combinaciones)."
3. "101 combinaciones con skill>0, DM significativo y recall>=0.20 presentaban colapso de alpha (alpha<0.50), solo identificables mediante alpha."
4. "alpha no correlaciona con dm_significant (rho=-0.039, p=0.35), capturando aspectos distintos del rendimiento."
5. "La evaluación conjunta produce decisiones sustancialmente más restrictivas que la evaluación solo por error."
6. "Urban/Industrial: 77.8% (14/18), una sola estación, sin generalización posible."

---

## 15. Claims prohibidos

- "MSE causa causalmente el colapso de varianza."
- "Todos los modelos convergen a predicciones suavizadas." (SARIMA: rho=-0.016)
- "Alpha es simplemente skill invertido."
- "AUC=1.000 demuestra valor incremental de alpha." (separación completa)
- "El 97.1% demuestra que los modelos ML son inútiles."
- "El fenómeno es general para todo forecasting ambiental."
- "Las 277 celdas son 277 observaciones independientes."
- "Las estaciones industriales preservan mejor los episodios." (n=1)

---

## 16. Reparaciones pendientes

### DOCUMENTAL (antes de enviar, no bloquea escritura)
1. Urban/Industrial: añadir n=1 estación y denominador explícito en manuscrito.
2. Preespecificación: añadir referencia al memo en el script y en el texto.
3. Dependencia de celdas: añadir nota explícita en la sección de métodos.

### MÉTODO (bloquea la cita de AUC como evidencia)
4. Eliminar o marcar como exploratoria la regresión logística. Sustituir AUC por descripción de 101 casos discordantes.

### NUEVO EXPERIMENTO — NO AUTORIZADO
- Corrección por dependencia (GEE, modelos mixtos): requiere diseño nuevo.
- Ampliación a otros contaminantes/países.

---

## 17. Tests: 18 PASS / 0 FAIL

py_compile, ejecución original, verificación independiente, json.tool: todos PASS.
Tests específicos: unicidad de clave, conteos (595/277/8/269/101), porcentajes,
correlación, denominadores, descomposición, condiciones de discordantes — todos PASS.

---

## 18. Estado Git final

Rama: main | HEAD inicial: 2398565652227dedf | Push: NO
Commit recomendado: audit(p4): verify decision-change analysis

---

## 19. Acción siguiente

```
WRITE_DECISION_ANALYSIS_SECTION
```

Condiciones previas (no bloquean el inicio de escritura):
1. Eliminar AUC logístico
2. Añadir denominador Urban/Industrial (n=1)
3. Añadir nota de dependencia estructural en métodos

---

*Generado: Antigravity — Auditoría CODEX P4 — 2026-08-06*
