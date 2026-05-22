# E1-RR Variance Retention Report (Daily Lags-Only)

## 1) Alcance

Este informe se limita estrictamente al paquete de trabajo de post-evaluación E1-RR:

- E1-RR daily lags-only.
- PM10 diario.
- Horizontes `h = 1..7`.
- Evaluación rolling-origin.
- Baseline obligatorio de persistencia.
- Sin variables meteorológicas.
- Sin E2-MET.
- Sin E3-PROB.

## 2) Validación de muestra

- Filas en `outputs/metrics/predictions.csv`: **26001**.
- Modelos presentes: **hgb_direct, persistence, ridge_direct**.
- Horizontes presentes: **1..7**.
- `n` mínimo por combinación `model/horizon`: **1215**.
- Conteo de `low_sample_flag=True` en `variance_retention_summary.csv`: **0**.

## 3) Tabla resumen por modelo

| model | skill mean | skill min | skill max | alpha mean | alpha min | alpha max | skill_vp mean | skill_vp min | skill_vp max | collapse_flag count | inflation_flag count | near_ideal_flag count | low_sample_flag count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hgb_direct | 0.212 | 0.055 | 0.254 | 0.145 | 0.101 | 0.298 | 0.027 | 0.016 | 0.034 | 7/7 | 0/7 | 0/7 | 0/7 |
| ridge_direct | 0.242 | 0.107 | 0.280 | 0.072 | 0.023 | 0.265 | 0.013 | 0.006 | 0.028 | 7/7 | 0/7 | 0/7 | 0/7 |

## 4) Diagnóstico textual

- Ambos modelos (hgb_direct, ridge_direct) muestran **skill positivo** frente a persistencia.
- Los modelos hgb_direct, ridge_direct muestran **baja retención de varianza** (alpha medio < 0.5).
- `hgb_direct` presenta **collapse** en **7/7** horizontes.
- `ridge_direct` presenta **collapse** en **7/7** horizontes.
- No hay evidencia de **inflation** (`inflation_flag=0`).
- No hay evidencia de comportamiento **near-ideal** (`near_ideal_flag=0`).
- `Skill_VP` queda reducido respecto al skill bruto cuando `alpha` es bajo, consistente con pronósticos suavizados.

## 5) Guardrails de redacción

- No afirmar ausencia total de predictibilidad solo por bajo `alpha`.
- No afirmar invalidez global del modelo únicamente por `collapse_flag`.
- No presentar `Skill_VP` como métrica universal de desempeño.
- No mezclar esta lectura con E2-MET ni E3-PROB.
- Mantener el encuadre como **diagnóstico auxiliar de post-evaluación**.

## 6) Frase paper-ready

“Positive persistence-relative skill was observed across horizons, but the accompanying variance-retention diagnostics indicate that this skill was largely associated with smoothed forecasts rather than near-observed dynamic variability.”
