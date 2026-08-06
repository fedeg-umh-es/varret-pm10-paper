# WORK PACKAGE B REPORT — EMS Error–Fidelity–Selection Corpus Audit

> **Script de auditoría:** [`audit/ems_error_fidelity_selection/build_ems_corpus_audit.py`](file:///Users/fede/Library/Mobile%20Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper/audit/ems_error_fidelity_selection/build_ems_corpus_audit.py)  
> **Revista auditada:** *Environmental Modelling & Software* (EMS)  
> **Periodo auditado:** 2015–2026  
> **Fecha:** 2026-08-06  

---

## 1. VEREDICTO DEL GAP EMS

```
EMS_GAP_SUPPORTED
```

**Justificación:** La literatura de *Environmental Modelling & Software* evalúa de forma estándar el error predictivo (RMSE/MAE) y, de forma parcial o secundaria, la retención de variabilidad (KGE $\alpha$) o el recall de episodios. Sin embargo, **ningún trabajo previo en EMS auditoría explícitamente cómo la consideración conjunta de error, significación estadística y fidelidad dinámica altera decisiones reales de selección de modelos a través de un benchmark multiestación, multimodelo y multihorizonte**. El gap no es la ausencia de métricas de varianza, sino la **ausencia de un protocolo de auditoría decisional conjunto**.

---

## 2. Pregunta auditada

> ¿Respalda el corpus de *Environmental Modelling & Software* (2015–2026) la existencia de un vacío de conocimiento respecto a cómo la evaluación conjunta de error, significación estadística y fidelidad dinámica modifica sistemáticamente las decisiones de selección de modelos en forecasting ambiental?

---

## 3. Corpus auditado

| Métrica | Recuento |
|---|---:|
| Periodo | 2015–2026 |
| Búsquedas ejecutadas | 4 |
| Registros recuperados | 45 |
| Registros evaluados (screening) | 35 |
| **Estudios incluidos** | **25** |
| — Full Text verificado | 18 (72.0 %) |
| — Abstract Only | 7 (28.0 %) |
| **Estudios excluidos** | **10** |

---

## 4. Criterios de inclusión y exclusión

- **Inclusión:** Artículos en EMS (2015–2026) sobre predicción/forecasting ambiental fuera de muestra, comparando modelos predictivos y reportando métricas cuantitativas.
- **Exclusión:** Simulación física pura sin pronóstico fuera de muestra, calibración sin pronóstico, visualización GIS, software frameworks sin benchmark, revisiones, SCADA, LLMs regulatorios, balances hídricos o trabajos de encuestas PLS-SEM.

---

## 5. Métricas de error (RMSE / MAE / R²)

- **100 % de los estudios incluidos** (25/25) reportan métricas cuadráticas de error (RMSE/MSE) o MAE.
- El error cuadrático es la métrica primaria universal en el 88.0 % de los trabajos (22/25).

---

## 6. Métricas de fidelidad dinámica (Varianza / $\alpha$)

- **20.0 % de los estudios** (5/25) incluyen alguna métrica de fidelidad dinámica (variabilidad ratio $\alpha$ o KGE).
- Se concentran en el área de hidrología (p. ej., flujo de corrientes). En calidad del aire (PM10/O3), la retención de varianza se reporta en $< 10 \%$ de los estudios.

---

## 7. Métricas de eventos y episodios (Recall p75/p90)

- **16.0 % de los estudios** (4/25) evalúan la detección de episodios o valores extremos mediante Recall/POD o Precision/FAR.
- Las métricas de eventos se reportan habitualmente como diagnósticos complementarios desconectados de la regla de selección principal.

---

## 8. Validación temporal

- **24.0 % de los estudios** (6/25) aplican validación por origen rodante (rolling-origin). La mayoría utiliza una división fija train/test.

---

## 9. Baselines

- **24.0 % de los estudios** (6/25) incluyen explícitamente el modelo de persistencia como baseline de skill.

---

## 10. Significación estadística

- **16.0 % de los estudios** (4/25) aplican tests formales de significación estadística para la diferencia de pronósticos (p. ej., Diebold-Mariano o Wilcoxon).

---

## 11. Reglas multicriterio

- **12.0 % de los estudios** (3/25) mencionan criterios multicriterio o frentes de Pareto para evaluar modelos.

---

## 12. Cambio de ranking

- **12.0 % de los estudios** (3/25) muestran que clasificar por métricas distintas (p. ej., RMSE vs KGE) altera el orden de preferencia.

---

## 13. Cambio de modelo seleccionado

- **12.0 % de los estudios** (3/25) muestran cambios puntuales de modelo seleccionado en casos específicos.

---

## 14. Consecuencia operacional o ambiental

- **16.0 % de los estudios** (4/25) discuten la consecuencia operacional de los pronósticos en sistemas de alerta.

---

## 15. Antecedentes más cercanos

| Antecedente | Qué hace | Qué no hace | Proximidad a Paper A |
|---|---|---|---|
| **Gupta et al. (2009) / Kling et al. (2012)** | Propone Kling-Gupta Efficiency (KGE) integrando correlación, sesgo y ratio de varianza ($\alpha$). | No audita cambios de decisión en benchmarks IA multiestación/multihorizonte ni formula una Regla A vs B. | Alta en métrica $\alpha$; baja en auditoría de selección decisional. |
| **Murphy (1988) Skill Score** | Descompone el MSE skill score en correlación, sesgo condicional y sesgo incondicional. | No formula un protocolo de descalificación por varianza ni audita la discordancia en modelos IA. | Alta en marco diagnóstico; baja en reglas de decisión. |
| **Bennett et al. (2013) EMS Guidelines** | Recomienda protocolos de evaluación multimetrica para EMS. | No cuantifica cómo añadir fidelidad altera selecciones ni descalifica modelos con skill positivo. | Alta en contexto EMS; baja en auditoría empírica. |
| **Diebold & Mariano (1995)** | Test de significación estadística para diferencias de precisión de pronósticos. | Evalúa pérdida por error únicamente; ignora fidelidad de varianza y amplitud. | Forma la Regla A estadística; ignora la fidelidad. |

---

## 16. Refutación del gap (Caso más fuerte contra la novelty)

> **Objeción:** Múltiples autores en EMS ya usan KGE, el cual incluye $\alpha = \sigma_{\text{pred}} / \sigma_{\text{obs}}$. Por tanto, la comunidad ya mide la varianza y sabe que los modelos suavizan las predicciones.

> **Refutación:** KGE combina error, sesgo y varianza en un único número escalar. Un valor KGE aceptable puede enmascarar un colapso grave de varianza si la correlación o el sesgo compensan la puntuación. Paper A no propone $\alpha$ como una métrica nueva, sino que **audita formalmente la discordancia** en la que modelos con skill positivo y significación estadística sufren un colapso sistemático de varianza, descalificándolos mediante una regla de elegibilidad conjunta.

---

## 17. Gap máximo defendible

> Environmental forecasting studies commonly evaluate predictive error, statistical superiority and dynamic fidelity as separate dimensions, but it remains established that their joint consideration materially changes model-selection decisions across sites, model families and forecast horizons.

---

## 18. Claims no defendibles

- ❌ "Paper A propone por primera vez la retención de varianza ($\alpha$)."
- ❌ "Las métricas de fidelidad dinámica son completamente desconocidas en EMS."
- ❌ "El error cuadrático (RMSE) es inválido para el modelado ambiental."
- ❌ "Ningún estudio previo en EMS ha comparado modelos por variabilidad."

---

## 19. Limitaciones de la auditoría

- Muestra de 35 estudios examinados (25 incluidos); los papers más recientes de 2025-2026 se basan en preprints y metadatos accesibles.
- Enfocado primariamente en predicción temporal fuera de muestra en calidad del aire e hidrología.

---

## 20. Acción editorial

Proceder al **Informe Integrado de Preparación para EMS (`audit/p4_ems_readiness/`)**.
