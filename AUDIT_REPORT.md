# Auditoría de reproducibilidad — varret-pm10-paper

**Fecha:** 2026-05-16  
**Commit auditado:** `fcec736` — 2026-05-14  
**Repo:** https://github.com/fedeg-umh-es/varret-pm10-paper  
**Auditor:** Claude Code (claude-sonnet-4-6)  

---

## 1. Inventario

### Estructura (3 niveles)

```
varret-pm10-paper/
├── config/                     ← Pipeline LEGACY (run_paper_a.sh)
│   ├── config.yaml             ← Hourly PM10, h∈{1,6,24,48}, 5 folds, LightGBM+SARIMA+LSTM
│   ├── horizons.yaml           ← [1, 6, 24, 48]
│   └── thresholds.yaml         ← P75, P90 percentiles
├── configs/                    ← Pipeline P33 (scripts/ directory)
│   ├── datasets/pm10.yaml      ← Daily PM10, frequency: D
│   ├── evaluation/rolling_origin.yaml   ← max_horizon: 7, open-ended
│   ├── evaluation/variance_diagnostics.yaml
│   ├── evaluation/regime_analysis.yaml  ← horizons: [1, 6, 24, 48] ← CONTRADICCIÓN
│   └── experiments/
│       ├── exp_p33_main.yaml   ← persistence, seasonal_persistence, linear_ar, boosting_tabular
│       ├── exp_boosting.yaml
│       ├── exp_baselines.yaml
│       └── exp_linear.yaml
├── data/
│   ├── raw/                    ← VACÍO (solo .gitkeep)
│   ├── interim/                ← VACÍO
│   └── processed/              ← VACÍO
├── docs/
│   ├── data_dictionary.md
│   ├── protocol.md
│   ├── regime_analysis.md
│   └── runbook.md
├── outputs/
│   ├── models/
│   │   ├── lgbm_holdout/       ← 4 archivos .txt (modelos entrenados, h∈{1,6,24,48})
│   │   └── lgbm_rolling_origin/fold_0..4/  ← 20 archivos .txt (5 folds × 4 horizontes)
│   ├── metrics/                ← VACÍO
│   ├── predictions/            ← VACÍO
│   ├── reports/                ← VACÍO
│   └── tables/                 ← VACÍO
├── scripts/                    ← Pipeline P33 (entrada pública)
│   ├── 01_validate_raw_data.py ← Comprueba data/raw/pm10_daily.csv
│   ├── 02_build_processed_datasets.py
│   ├── 03_run_baselines.py     ← STUB (solo print)
│   ├── 04_run_linear_models.py ← STUB (solo print)
│   ├── 05_run_boosting_model.py← STUB (solo print)
│   ├── 06_build_skill_tables.py← STUB (solo print)
│   ├── 07_build_variance_retention_table.py ← Funcional (lee CSVs intermedios)
│   ├── 08_build_run_summary.py ← Funcional (escribe texto)
│   └── run_p33_pipeline.py     ← Orquestador secuencial de scripts 01-08
├── src/
│   ├── data/           ← load_data.py, preprocess_pm10.py, make_splits.py, etc.
│   ├── diagnostics/    ← variance.py (alpha = Var_pred/Var_obs)
│   ├── evaluation/     ← skill.py, metrics.py, run_rolling_origin.py, etc.
│   ├── models/         ← lgbm_model.py, sarima_model.py, persistence.py, linear_ar.py,
│   │                      seasonal_persistence.py, boosting_tabular.py, lstm_model.py
│   ├── plotting/       ← plot_master_figure.py (LightGBM + SARIMA)
│   ├── splits/         ← rolling_origin.py (open-ended, sin n_folds fijo)
│   └── training/       ← train_lgbm.py, train_sarima.py, train_persistence.py
├── tests/
│   ├── test_rolling_origin.py
│   ├── test_skill.py
│   └── test_variance.py
├── AUDIT_SUMMARY.md            ← Informe de auditoría anterior (2026-04-11)
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_paper_a.sh              ← Pipeline legacy completo (13 pasos, incluye LSTM y SARIMA)
├── run_paper_a_fixed.sh
└── run_paper_a_no_lstm.sh
```

### Métricas del repositorio

| Dato | Valor |
|---|---|
| Tamaño total | 18 MB |
| Archivos (excl. .git) | 127 |
| Commits totales | 11 |
| Último commit | fcec736, 2026-05-14 |
| Lenguaje principal | Python |
| Tests | 3 archivos en tests/ |
| Artefactos de modelo presentes | 24 archivos LightGBM (.txt) en outputs/models/ |
| Predicciones / métricas / figuras | **NINGUNA** — todos los directorios están vacíos |

### Hallazgo estructural crítico

El repositorio contiene **dos pipelines paralelos e incompatibles** que nunca se conectan:

- **Pipeline LEGACY** (`run_paper_a.sh` + `config/config.yaml`): PM10 horario, modelos LightGBM+LSTM+SARIMA, h∈{1,6,24,48}, 5 folds con partición porcentual fija.
- **Pipeline P33** (`scripts/run_p33_pipeline.py` + `configs/`): PM10 diario, modelos persistence+linear_ar+boosting_tabular, h=1..7, ventana expandida sin número fijo de folds.

El README apunta al Pipeline P33 como canónico. Los artefactos de modelo entrenados en `outputs/models/` pertenecen al Pipeline LEGACY. Las figuras del paper (según `plot_master_figure.py`) corresponden al Pipeline LEGACY. El Pipeline P33 tiene los scripts 03-06 como stubs no implementados y no puede generar ningún resultado.

---

## 2. Pipeline ejecutable

### Script principal
- **README recomienda:** `python scripts/run_p33_pipeline.py`
- **Script alternativo (legacy):** `bash run_paper_a.sh`

### Configuraciones

| Config | Pipeline | Horizontes | Modelos | Frecuencia |
|---|---|---|---|---|
| `config/config.yaml` | LEGACY | [1, 6, 24, 48] horas | LightGBM, LSTM, SARIMA | Horaria |
| `configs/experiments/exp_p33_main.yaml` | P33 | 1..7 días | persistence, seasonal_persistence, linear_ar, boosting_tabular | Diaria |
| `configs/evaluation/rolling_origin.yaml` | P33 | max_horizon=7 | — | — |
| `configs/evaluation/regime_analysis.yaml` | P33 | [1, 6, 24, 48] | — | **INCONSISTENTE** |

### Dataset

| Atributo | Configuración |
|---|---|
| Ruta (Pipeline LEGACY) | `data/raw/pm10_measurements.csv` |
| Ruta (Pipeline P33) | `data/raw/pm10_daily.csv` |
| Estado real | **AUSENTE** — `data/raw/` contiene solo `.gitkeep` |
| Instrucciones de descarga | No existen URL, query EEA ni checksum |
| Estación declarada | Sin referencia explícita en el código (solo en el manuscrito) |

### Dependencias

`requirements.txt` especifica solo: `pandas, numpy, scikit-learn, matplotlib, pyyaml, pytest` — sin versiones pin.  
Librerías requeridas por el código pero **no declaradas**: `lightgbm`, `statsmodels`.  
`pyproject.toml`: `dependencies = []` (vacío).

---

## 3. Verificación de hipótesis (H1–H15)

| ID | Hipótesis | Veredicto | Evidencia (archivo:línea o función) | Nota |
|---|---|---|---|---|
| H1 | Modelos son LightGBM y SARIMA | **REFUTADA** para el pipeline canónico P33 | `configs/experiments/exp_p33_main.yaml:6-9` — lista persistence, seasonal_persistence, linear_ar, boosting_tabular. LightGBM y SARIMA existen en `src/models/` pero NO son parte del pipeline P33. | Los artefactos de modelo en outputs/ confirman que LightGBM+SARIMA se entrenaron solo en el pipeline LEGACY. La figura `plot_master_figure.py:43-46` muestra MODELS=['lgbm','sarima'] y corresponde al pipeline LEGACY. |
| H2 | Horizontes h∈{1,6,24,48} | **REFUTADA** para el pipeline canónico P33 | `configs/evaluation/rolling_origin.yaml:2` — `max_horizon: 7` (días 1–7 sin selección de subconjunto). `config/horizons.yaml:6-9` — sí declara [1,6,24,48] pero pertenece al pipeline LEGACY. | El Pipeline P33 usa h=1..7. El Pipeline LEGACY usa h∈{1,6,24,48}. Los artefactos de modelo entrenados (lgbm_h1.txt, lgbm_h6.txt, lgbm_h24.txt, lgbm_h48.txt) son del LEGACY. |
| H3 | Existe persistence como baseline explícito con ŷ(t+h\|t)=y(t) | **CONFIRMADA** | `src/models/persistence.py:25-26` — `return np.repeat(float(values[-1]), horizon)`. Declarado en `exp_p33_main.yaml:7` y en `exp_baselines.yaml:4`. | La implementación es correcta: repite el último valor observado. |
| H4 | Rolling-origin con 5 expanding-window folds | **PARCIALMENTE REFUTADA** | Pipeline LEGACY: `config/config.yaml:64-67` — `n_folds: 5`. Pipeline P33: `src/splits/rolling_origin.py:24-67` — sin `n_folds`, genera todos los orígenes posibles desde `min_train_size=60`. Los artefactos de modelo muestran fold_0..fold_4 (5 folds del LEGACY). | El manuscrito declara 5 folds. El pipeline P33 canónico no implementa un número fijo de folds. |
| H5 | Preprocessing y scaler entrenados solo con train (sin fuga) | **REFUTADA** para el pipeline LEGACY; **NO VERIFICABLE** para P33 | `src/data/preprocess_pm10.py:36-48` — `mean = series.mean()` y `std = series.std()` sobre la serie completa antes de cualquier split. Fuga confirmada: estadísticos de normalización calculados con datos de test. Pipeline P33: scripts 02-05 son stubs, sin implementación. | La normalización global pre-split es fuga de datos. El `protocol.md` declara explícitamente "train-only preprocessing" pero el código la viola. |
| H6 | P75 y P90 recalibrados por fold usando solo observaciones de entrenamiento | **NO VERIFICABLE** | Scripts P33 03-05 son stubs (`print("Run baselines...")`). `config/thresholds.yaml:9` — método percentil, pero sin implementación de recalibración por fold. | Ninguna implementación activa calcula umbrales por fold. |
| H7 | VR(h) = 100 · Var_pred(h) / Var_obs(h) | **REFUTADA** (discrepancia de escala y nombre) | `src/diagnostics/variance.py:42-46` — `alpha = predicted_variance / observed_variance` (ratio 0–∞, sin ×100). `src/evaluation/compute_metrics_by_horizon.py:48-51` — `var_pct = 100 * var_pred / var_obs` (porcentaje 0–100+). El schema P33 (`src/data/schema.py:6-16`) usa `alpha`, no `var_pct`. | El manuscrito llama VR a una métrica en %, el código P33 la implementa como ratio sin escala. El código LEGACY sí implementa el porcentaje (var_pct). Dos implementaciones distintas con nombres distintos en el mismo repo. |
| H8 | Skill_RMSE = 1 − RMSE_m(h) / RMSE_p(h) | **CONFIRMADA** | `src/evaluation/skill.py:10` — `return 1.0 - (model_error / baseline_error)`. `src/evaluation/compute_metrics_by_horizon.py:41-43` — `skill = 1 - (rmse_model / rmse_persist)`. | Fórmula correcta en ambos pipelines. |
| H9 | Skill_VP = Skill_RMSE(h) · min(1, VR(h)/100) | **REFUTADA** (falta el cap `min(1,...)`) | `src/diagnostics/variance.py:34` — `skill_vp = skill * alpha`. `src/evaluation/compute_metrics_by_horizon.py:56-57` — `skill_vp = skill * (var_pct / 100)`. Ninguna implementación aplica `min(1, ...)`. | Si alpha > 1 (inflación de varianza), el código amplifica skill_vp. El manuscrito declara cap en 1 para este caso. |
| H10 | Cifras de Tabla 2 reproducibles desde el código | **NO VERIFICABLE EN ESTA SESIÓN** | No hay datos en `data/raw/`. No hay predicciones en `outputs/predictions/`. Scripts P33 03-06 son stubs. | Comando para intentar reproducción (requiere dataset): `bash run_paper_a.sh`. La Tabla 2 declara LightGBM: Skill≈0.92–0.96, VR≈78–99%; SARIMA: Skill≈0.89, VR<2%. No hay forma de verificar estos números sin los datos. |
| H11 | Figuras del paper generadas desde el repo con leyendas que coinciden con modelos declarados | **PARCIALMENTE REFUTADA** | `src/plotting/plot_master_figure.py:43-46` — `MODELS=['lgbm','sarima']`, `LABELS={'lgbm':'LightGBM','sarima':'SARIMA'}`. Las figuras requerirían ejecutar el pipeline LEGACY completo. No existen figuras en `outputs/`. | El código genera figuras con LightGBM + SARIMA, que coincide con el texto del manuscrito. Sin embargo el PDF compilado reporta leyendas adicionales (Linear AR, Seasonal Persistence, XGBoost) que NO aparecen en ningún script de plotting del repo. Las figuras actuales del PDF NO pueden ser de este repo. |
| H12 | Semilla aleatoria fijada para LightGBM y splits | **CONFIRMADA** | `config/config.yaml:62` — `random_state: 42` para splits. `config/config.yaml:84` — `random_state: 42` para LightGBM. `src/models/lgbm_model.py:56` — `random_state=self.lgb_params.get("random_state", 42)`. `configs/evaluation/regime_analysis.yaml:24` — `random_state: 42` para clustering. | Semilla fijada en configuración y propagada a los modelos. |
| H13 | Dataset crudo (PM10 Casa de Campo 2023, EEA) con URL, query o checksum documentado | **REFUTADA** | `data/raw/` contiene solo `.gitkeep`. `docs/runbook.md:1` — "Place the daily PM10 raw dataset in `data/raw/`" sin URL ni instrucciones de descarga. `docs/data_dictionary.md` — no menciona fuente. | El dataset no está en el repo ni en instrucciones de descarga reproducibles. La estación "Casa de Campo (Madrid)" solo aparece en el manuscrito, no en el código. |
| H14 | Repo declara licencia MIT y referencia DOI Zenodo 10.5281/zenodo.20185328 | **CONFIRMADA** | `LICENSE` existe. `CITATION.cff:9` — `license: MIT`. `README.md:2` — badge DOI 10.5281/zenodo.20185328. `CITATION.cff` — referencia el mismo DOI en el campo `repository-code`. | `CITATION.cff` no incluye campo `doi:` explícito, pero la referencia al DOI existe en README. |
| H15 | README documenta cómo reproducir tablas y figuras paso a paso | **REFUTADA** | `README.md:102-127` — proporciona `pip install` y `python scripts/run_p33_pipeline.py` pero: (1) no indica cómo obtener el dataset; (2) no mapea qué script produce qué tabla/figura del paper; (3) no advierte que scripts 03-06 son stubs; (4) el pipeline de figuras (`plot_master_figure.py`) requiere `run_paper_a.sh`, no `run_p33_pipeline.py`. | Un revisor no puede reproducir nada con las instrucciones actuales. |

---

## 4. Reproducibilidad estándar Q1

| Ítem | Estado | Nota |
|---|---|---|
| README con quickstart reproducible | [ ] | Existe README pero el quickstart es incompleto: no documenta origen del dato y apunta al pipeline P33 que tiene stubs. |
| Lista de dependencias con versiones pin | [ ] | `requirements.txt` sin versiones. `pyproject.toml` con `dependencies=[]`. `lightgbm` y `statsmodels` no declarados en ningún archivo. |
| Script único o makefile que regenera tablas y figuras desde datos crudos | [ ] | `run_paper_a.sh` y `run_p33_pipeline.py` existen pero son de pipelines distintos. Ninguno produce tablas/figuras del manuscrito en estado ejecutable actual. |
| Datos en repo o instrucciones unívocas de descarga | [ ] | `data/raw/` vacío. Sin URL EEA, sin query, sin checksum. |
| Semillas fijadas para todos los componentes estocásticos | [x] | `random_state: 42` en configuración y modelos LightGBM. SARIMA no tiene componente estocástica en inferencia. |
| Tests unitarios o checks de integridad mínimos | [x] | 3 archivos de tests en `tests/` (rolling_origin, skill, variance). Conceptualmente correctos y ejecutables. |
| Documentación de cada métrica (VR, Skill_RMSE, Skill_VP, exceedance) con referencia a definición en el paper | [ ] | `docs/protocol.md` describe cualitativamente. No existe equivalencia explícita entre nombres de variables en código (`alpha`, `var_pct`, `skill_vp`) y notación del manuscrito (VR, Skill_RMSE, Skill_VP). |
| Citation file (CITATION.cff) con DOI Zenodo | [x] | `CITATION.cff` existe y referencia el repositorio con DOI. Le falta campo `doi:` explícito. |
| Licencia archivada (LICENSE file) | [x] | `LICENSE` existe (MIT). |
| CHANGELOG o tags de versiones alineados con DOI release | [ ] | No existe CHANGELOG. No hay tags de versión en el repo. La versión en CITATION.cff es 1.0.0 pero no hay tag git correspondiente. |
| CI mínimo (GitHub Actions) que al menos importe el paquete sin fallar | [ ] | No existe directorio `.github/workflows/`. Sin CI. |

**Resumen: 4/11 ítems cumplidos.**

---

## 5. Mismatches manuscrito vs. código (prioridad alta)

### MISMATCH 1 — Modelos del paper vs. modelos del pipeline canónico
- **Qué dice el manuscrito:** Compara LightGBM vs. SARIMA.
- **Qué ejecuta el código P33 canónico:** `exp_p33_main.yaml` — persistence, seasonal_persistence, linear_ar, boosting_tabular. Sin LightGBM ni SARIMA en el pipeline P33.
- **Qué ejecuta el código LEGACY:** LightGBM + SARIMA + LSTM + Persistence (sí coincide con el manuscrito).
- **Severidad:** BLOQUEANTE. La narrativa del paper y el pipeline "oficial" del repo son para experimentos distintos.
- **Fix mínimo:** Designar el pipeline LEGACY (run_paper_a.sh) como canónico, o migrar LightGBM+SARIMA al pipeline P33.

### MISMATCH 2 — Horizontes del paper vs. horizontes del pipeline canónico
- **Qué dice el manuscrito:** h ∈ {1, 6, 24, 48}.
- **Qué ejecuta el pipeline P33:** h = 1, 2, 3, 4, 5, 6, 7 (días consecutivos, sin selección).
- **Qué ejecuta el pipeline LEGACY:** h ∈ {1, 6, 24, 48} — coincide con el manuscrito.
- **Qué declara la figure code:** `config/horizons.yaml` = [1, 6, 24, 48], y los eje x de las figuras usarían estos valores.
- **Severidad:** BLOQUEANTE. El pipeline canónico no produce los horizontes que el manuscrito reporta.
- **Fix mínimo:** Actualizar `configs/evaluation/rolling_origin.yaml` para seleccionar subconjunto {1,6,24,48} o declarar el pipeline LEGACY como canónico.

### MISMATCH 3 — Frecuencia del dataset
- **Qué dice el manuscrito:** PM10 horario, 8.760 observaciones (1 año × 1h).
- **Qué declara el pipeline P33:** `configs/datasets/pm10.yaml:5` — `frequency: D` (diario). `data_dictionary.md:4` — "daily timestamp".
- **Qué declara el pipeline LEGACY:** `config/config.yaml:55-56` — LSTM context_length=168 horas (7 días), lags horarios.
- **Evidencia de los artefactos:** `lgbm_h1.txt` contiene feature `hour` (0-23) y `dayofweek` (0-6) — confirma entrenamiento con data horaria. fold_0 con 4356 muestras (≈ 50% de 8.760).
- **Severidad:** BLOQUEANTE. Los pipelines contradí entre sí y uno contradice el manuscrito.
- **Fix mínimo:** Eliminar la ambigüedad: si el paper usa PM10 horario, el dataset canónico es horario y configs/ debe reflejarlo.

### MISMATCH 4 — Fuga de datos en normalización
- **Qué dice el manuscrito/protocolo:** "train-only preprocessing" (`docs/protocol.md:3`).
- **Qué hace el código:** `src/data/preprocess_pm10.py:36-48` normaliza con media y desviación de la serie completa antes de hacer splits.
- **Severidad:** MAYOR. Invalida las comparaciones de skill si los resultados de la Tabla 2 se produjeron con este código.
- **Fix mínimo:** Mover la normalización al interior del bucle de folds, calculando mean/std solo sobre train_idx en cada fold.

### MISMATCH 5 — Fórmula de VR: porcentaje vs. ratio
- **Qué dice el manuscrito:** VR(h) = 100 · Var_pred(h) / Var_obs(h) — escala 0–100+%.
- **Qué implementa el pipeline P33:** `src/diagnostics/variance.py:42-46` — `alpha = Var_pred/Var_obs` — escala 0–∞.
- **Qué implementa el pipeline LEGACY:** `src/evaluation/compute_metrics_by_horizon.py:48-51` — `var_pct = 100 * var_pred / var_obs` — coincide con el manuscrito.
- **Severidad:** MAYOR. Las tablas del paper declaradas en % son incompatibles con el schema P33 que usa `alpha` sin ×100.
- **Fix mínimo:** Unificar en un nombre (`VR` o `alpha`) y escalar consistentemente. Documentar la equivalencia.

### MISMATCH 6 — Cap min(1,...) ausente en Skill_VP
- **Qué dice el manuscrito:** Skill_VP(h) = Skill_RMSE(h) · min(1, VR(h)/100).
- **Qué hace el código:** `src/diagnostics/variance.py:34` — `skill_vp = skill * alpha` sin cap. `compute_metrics_by_horizon.py:56-57` — `skill_vp = skill * (var_pct / 100)` sin cap.
- **Severidad:** MENOR (solo afecta casos de inflación de varianza, alpha>1).
- **Fix mínimo:** Añadir `min(1, alpha)` o `min(1, var_pct/100)` antes de la multiplicación.

### MISMATCH 7 — Número de folds en rolling-origin
- **Qué dice el manuscrito:** "5 expanding-window folds".
- **Qué implementa el pipeline P33:** `src/splits/rolling_origin.py:24-67` — ventana expandible con todos los orígenes posibles (sin cap en 5 folds).
- **Qué implementa el pipeline LEGACY:** `config/config.yaml:64-67` — `n_folds: 5` — coincide con el manuscrito.
- **Severidad:** MAYOR para el pipeline P33 (no reproducible como declarado).
- **Fix mínimo:** Documentar explícitamente qué pipeline usa 5 folds.

### MISMATCH 8 — Leyendas de figuras del PDF no presentes en el código de figuras
- **Qué dice el PDF compilado (según el prompt):** Figuras 2 y 3 muestran LightGBM + Linear AR + Seasonal Persistence; Figura 4 muestra XGBoost + SARIMA + Persistence.
- **Qué genera `plot_master_figure.py`:** Solo LightGBM + SARIMA (MODELS=['lgbm','sarima']). Sin XGBoost ni Linear AR en ningún script de plotting.
- **Severidad:** BLOQUEANTE. Las figuras enviadas al editor no pueden proceder del código actual del repo.
- **Fix mínimo:** Identificar el script que generó las figuras del PDF (no está en el repo) y o añadirlo o regenerar las figuras con el código existente.

---

## 6. Qué falta para defender APR

Las siguientes acciones están ordenadas por impacto editorial:

**ACCIÓN 1: DESIGNAR un único pipeline canónico y eliminar la ambigüedad** — `README.md` + `scripts/` o `run_paper_a.sh` — El repo tiene dos pipelines incompatibles. Hay que decidir cuál produce el paper (casi seguramente el LEGACY con LightGBM+SARIMA+h∈{1,6,24,48}) y reorientar el README, los configs y el directorio raíz para apuntar solo a ese. El pipeline P33 actual con stubs actúa como ruido y confunde la trazabilidad.

**ACCIÓN 2: PROPORCIONAR instrucciones unívocas de descarga del dataset** — `docs/runbook.md` o `README.md` — Sin URL EEA, sin query, sin nombre de estación en el código, sin checksum. Un revisor Q1 no puede obtener los datos. Añadir URL de descarga, parámetros de la query API EEA (estación, pollutante, año), y hash SHA256 del CSV resultante.

**ACCIÓN 3: CORREGIR la fuga de datos en normalización** — `src/data/preprocess_pm10.py:36-48` — Mover mean/std al interior del bucle de folds. Sin este fix, los resultados de la Tabla 2 son potencialmente inflados y no defensibles ante revisión.

**ACCIÓN 4: REGENERAR las figuras del paper desde el código del repo** — `src/plotting/plot_master_figure.py` — Las figuras del PDF enviado al editor muestran modelos (XGBoost, Linear AR, Seasonal Persistence) que no existen en el código de plotting. Hay que hacer que las figuras en `outputs/figures/` sean el resultado ejecutable del script actual, o identificar y añadir el script que generó las figuras originales.

**ACCIÓN 5: PINEAR dependencias con versiones exactas** — `requirements.txt` — Añadir versiones específicas para pandas, numpy, scikit-learn, lightgbm, statsmodels, matplotlib, pyyaml, pytest. Sin versiones pin, el código puede dejar de reproducir resultados en 6 meses.

**ACCIÓN 6: AÑADIR CI mínimo (GitHub Actions)** — `.github/workflows/test.yml` — Un workflow que instale dependencias y ejecute `pytest` sin datos (los 3 tests existentes no requieren datos). Esto permite a revisores verificar que el entorno de código es al menos importable y que los tests pasan.

**ACCIÓN 7: UNIFICAR la terminología de métricas** — `src/diagnostics/variance.py`, `src/evaluation/compute_metrics_by_horizon.py`, `docs/data_dictionary.md` — Hay dos nombres (`alpha`, `var_pct`) para la misma métrica en diferentes partes del código, con diferente escala (ratio vs. porcentaje). Añadir una tabla de correspondencia nombre-código ↔ notación-paper en el data dictionary.

---

## 7. Riesgos editoriales detectados

Un revisor Q1 de APR que abra el repo encontraría los siguientes problemas inmediatos:

1. **Dataset no obtenible:** `data/raw/` vacío, sin URL ni instrucciones. El paper no se puede reproducir sin los datos. Esto es un rechazo automático en muchas revistas Q1 con política de datos abiertos.

2. **Pipeline canónico no ejecutable:** `python scripts/run_p33_pipeline.py` falla silenciosamente en el paso 3 (stubs que solo imprimen texto). No hay error; el script termina con "éxito" pero no produce ningún resultado.

3. **Figuras del PDF no generables:** Las figuras enviadas muestran modelos que no existen en el código de plotting. Un revisor que ejecute `plot_master_figure.py` obtendrá figuras con solo LightGBM+SARIMA, no las figuras del artículo.

4. **Dos pipelines en conflicto:** El directorio `config/` apunta a PM10 horario; `configs/` apunta a PM10 diario. El README no documenta la diferencia. El README dice "el pipeline P33" pero el único pipeline con LightGBM+SARIMA es el LEGACY.

5. **`requirements.txt` incompleto:** `lightgbm` y `statsmodels` no declarados. Un `pip install -r requirements.txt` seguido de ejecución del pipeline LEGACY falla con ImportError.

6. **`pyproject.toml` con `dependencies = []`:** Un revisor que instale el paquete via `pip install .` no obtiene ninguna dependencia.

7. **Normalización pre-split (posible data leakage):** Revisores con experiencia en series temporales identificarán que `preprocess_pm10.py` normaliza la serie completa antes de los splits. Esto invalida la interpretación de los resultados como "leakage-free".

8. **Versión 1.0.0 en CITATION.cff sin tag git correspondiente:** El DOI de Zenodo debería corresponder a un release taggeado. Si el DOI existe pero el código en Zenodo no corresponde al commit actual, hay inconsistencia de archivado.

---

## 8. Información que el auditor no pudo verificar

1. **H10 — Cifras de Tabla 2 del manuscrito:** No verificable. No hay datos raw, no hay predicciones almacenadas, y los scripts de evaluación del pipeline P33 son stubs. El comando para intentarlo sería `bash run_paper_a.sh` con `data/raw/pm10_measurements.csv` del EEA.

2. **Correspondencia exacta entre el PDF compilado y el código:** El auditor no tuvo acceso al PDF del manuscrito. Las incoherencias de figuras reportadas en el prompt se aceptaron como input y se verificó que el código de plotting no genera esas leyendas adicionales.

3. **Validez del DOI Zenodo 10.5281/zenodo.20185328:** El auditor no verificó si el DOI es activo y si el snapshot archivado corresponde al commit actual.

4. **Reproducibilidad numérica completa (H10):** Requeriría ejecutar el pipeline completo con datos reales. Estimación de tiempo: 30-60 minutos según AUDIT_SUMMARY.md del propio repo. No se ejecutó por ausencia de datos.

5. **`run_paper_a_fixed.sh` y `run_paper_a_no_lstm.sh`:** Estos scripts existen pero no se auditaron en detalle. Podrían contener el pipeline corregido que generó las figuras del PDF.

6. **Correspondencia entre los artefactos de modelo en `outputs/models/` y los datos de entrenamiento:** Los modelos LightGBM existen para 5 folds y holdout, con features horarias (lag_1, lag_6, lag_24, lag_48, hour, dayofweek, month) y ≈4.356 muestras en fold_0 (50% de ~8.760). Esto es consistente con PM10 horario 2023. Sin embargo no se puede verificar que fue exactamente la estación Casa de Campo y año 2023 sin los datos originales.

---

## Respuestas a las 5 preguntas clave del criterio de éxito

**1. ¿Qué modelos entrena el código realmente?**  
El pipeline P33 canónico (`scripts/run_p33_pipeline.py`) solo implementa conceptualmente `persistence` y `seasonal_persistence`; los scripts para `linear_ar` y `boosting_tabular` son stubs vacíos. El pipeline LEGACY (`run_paper_a.sh`) sí entrena LightGBM y SARIMA (implementaciones completas en `src/training/`). Los artefactos de modelo en `outputs/models/` son LightGBM entrenado con el pipeline LEGACY en PM10 horario con h∈{1,6,24,48}.

**2. ¿Qué horizontes evalúa?**  
El pipeline P33 evalúa h=1..7 (días). El pipeline LEGACY evalúa h∈{1,6,24,48} (horas). Los artefactos de modelo entrenados corresponden a h∈{1,6,24,48}. El manuscrito declara h∈{1,6,24,48}.

**3. ¿Las figuras del PDF se generan desde este código?**  
No completamente. `plot_master_figure.py` genera figuras con LightGBM+SARIMA que coincide con la narrativa del manuscrito, pero no puede generar las leyendas adicionales (XGBoost, Linear AR, Seasonal Persistence) reportadas en el PDF. Las figuras del PDF NO están presentes en `outputs/` y no pueden trazarse al código actual.

**4. ¿Las cifras de Tabla 2 son reproducibles desde este código?**  
No verificable. Sin datos en `data/raw/`, sin predicciones en `outputs/`, y con el pipeline de evaluación incompleto (stubs en scripts 03-06). Adicionalmente, existe una fuga de datos en la normalización que potencialmente afecta los números reportados.

**5. ¿Qué tres cosas mínimas hay que arreglar antes de someter a APR?**  
(1) Designar un pipeline canónico único que produzca los resultados del paper — actualmente hay dos pipelines incompatibles y el canónico no funciona.  
(2) Documentar el dataset con URL EEA, parámetros de query y checksum — sin datos no hay reproducibilidad posible.  
(3) Corregir la normalización pre-split (fuga de datos) y regenerar todos los resultados para que los números del papel sean defendibles ante revisión de métodos.
