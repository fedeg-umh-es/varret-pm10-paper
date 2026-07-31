# PROG-P1-02 — Auditoría de inferencia temporal y bootstrap

## Veredicto

`BLOCKED_BY_LOCAL_STATE`

Los defectos temporales localizados en el código se corrigieron y la suite final
pasa, pero los artefactos inferenciales presentes no son trazables a los inputs
disponibles. No se puede declarar válido el protocolo ni autorizar
PROG-P1-03 hasta recuperar los inputs canónicos multiestación y regenerar, en
una tarea posterior, las salidas afectadas.

## Entorno

| Campo | Valor |
|---|---|
| Hostname | `MacBook-Neo-de-fede.local` |
| Sistema | Darwin 25.5.0, arm64 |
| Repositorio | `/Users/fede/repos/varret-pm10-paper` |
| Worktree de auditoría | `/Users/fede/Documents/Codex/2026-07-31/files-mentioned-by-the-user-repos/work/varret-pm10-paper-prog-p1-02` |
| Rama inicial | `main` |
| Rama de reparación | `codex/prog-p1-02` |
| HEAD inicial | `a070f946f0f1ee33f0781fc0865000ce971af63e` |
| Estado inicial | limpio; `main...origin/main [behind 1]` |
| Remoto | `origin https://github.com/fedeg-umh-es/varret-pm10-paper` |
| Python del sistema | 3.9.6; sin entorno virtual y sin dependencias del proyecto |
| Python de auditoría | 3.12.13, entorno aislado externo al repositorio |

Dependencias de auditoría: NumPy 2.5.1, Pandas 3.0.5, SciPy 1.18.0,
statsmodels 0.14.6, scikit-learn 1.9.0, pytest 9.1.1 y PyYAML 6.0.3.

## Alcance y mapa mecánico

| Componente | Archivo/función | Entrada | Salida | Unidad estadística |
|---|---|---|---|---|
| Predicciones rolling-origin | `scripts/01_generate_e1_rr_lags_only_predictions.py::_generate_predictions_for_horizon` | PM10 diario | `predictions_*.csv` | estación × modelo × horizonte × origen |
| Predicciones SARIMA dispersas | `scripts/02_generate_sarima_predictions.py::_generate_predictions` | PM10 diario, stride configurable | `predictions_sarima_*.csv` | estación × horizonte × origen |
| Skill | ambos scripts, `_build_skill_summary` | predicciones de modelo y persistencia | `skill_*.csv` | estación × modelo × horizonte |
| Emparejamiento reparado | `src/evaluation/pairing.py::pair_model_and_baseline` | modelo y persistencia | pares finitos uno-a-uno | fold × origen × fecha objetivo × horizonte |
| DM/HAC/HLN | `scripts/05_dm_significance.py::build_dm_table` | predicciones emparejadas | `dm_*.csv` | estación × modelo × horizonte |
| Bootstrap de alpha | `src/diagnostics/variance.py::_bootstrap_alpha_ci` | secuencia ordenada de predicciones | IC percentil | estación × modelo × horizonte, bloques de origen |
| Tabla de varianza | `scripts/07_build_variance_retention_table.py` | predicciones y skill | `variance_retention_*.csv` | estación × modelo × horizonte |
| Unificación | `scripts/build_unified_{predictions,dm,variance}_table.py` | tablas por estación | tablas multiestación | estación × modelo × horizonte |
| Consumo de CI | `scripts/audit_alpha_bootstrap_ci.py`; `scripts/build_overleaf_consistency_prompt.py` | tabla de varianza | auditoría/claims auxiliares | celda estación-modelo-horizonte |
| H* | no existe implementación ejecutable | — | — | — |
| Yule–Walker | no existe implementación ejecutable | — | — | — |

Se inspeccionaron además `run_all_stations_v2.sh`, `config/config.yaml`,
`config/horizons.yaml`, `configs/evaluation/rolling_origin.yaml`,
`docs/e1_rr_post_evaluation_contract.md`, `paper_a.tex`, los tres CSV de
predicciones disponibles, los tres CSV de skill, las tres tablas de
variance-retention y toda la suite bajo `tests/`.

## Contrato inferencial encontrado

| Campo | Contrato real |
|---|---|
| Estimando DM | media de `SE_persistence - SE_model`; signo positivo favorece al modelo |
| Unidad | origen rolling, separada por dataset/estación, modelo y horizonte |
| Baseline | persistencia |
| Pérdida | error cuadrático; el skill descriptivo usa RMSE y MAE relativos |
| Emparejamiento | intersección uno-a-uno por `fold, origin_date, date, horizon`, con `y_true` común y filtrado finito conjunto |
| DM | bilateral; estadístico positivo favorece al modelo |
| HAC | Bartlett/Newey–West; lag `ceil(horizon / min_positive_origin_stride) - 1` |
| HLN | activa por defecto; usa horizonte efectivo `hac_lag + 1`; distribución t con `n-1` grados de libertad |
| Bootstrap | circular moving-block sobre orígenes ordenados, dentro de estación/modelo/horizonte |
| Bloque | `max(h, ceil(n^(1/3)))`, acotado por `n` |
| Bootstrap RNG | 1000 réplicas, semilla 42, IC percentil 95 % |
| Multiplicidad | Benjamini–Hochberg por dataset sobre modelos × horizontes; controla FDR |
| H* | ausente; `paper_a.tex:91` declara explícitamente “no H*” |
| Censura | ausente porque no se deriva H* |
| Yule–Walker | ausente; no se usa como baseline, test ni sustituto empírico |

El repositorio no contiene el manuscrito denominado `predictability-bound`;
el manuscrito local es Paper A sobre skill/variance-retention y el paquete se
identifica internamente como P33. Esta discrepancia se conserva como blocker de
alcance, no se resolvió por inferencia.

## Hallazgos

| ID | Severidad | Evidencia | Consecuencia | Resolución |
|---|---|---|---|---|
| P1-02-001 | CRITICAL | HEAD inicial `src/diagnostics/variance.py:69-73` remuestreaba índices IID; `scripts/build_overleaf_consistency_prompt.py:520-526` usa esos IC en evidencia agregada | IC demasiado estrechos o mal calibrados bajo dependencia y solapamiento | Reemplazado por circular moving-block en `src/diagnostics/variance.py:79-131`; test de orden y reproducibilidad |
| P1-02-002 | CRITICAL | HEAD inicial `scripts/05_dm_significance.py:76-90` y builders de skill hacían joins sin `validate`, sin control de duplicados y sin filtro finito conjunto | join many-to-many, `n` y diferencial de pérdidas adulterados | Helper único en `src/evaluation/pairing.py:9-58`; aplicado a DM y ambos builders de skill |
| P1-02-003 | MAJOR | HEAD inicial `scripts/05_dm_significance.py:33-44` fijaba lag `h-1`, aunque `run_all_stations_v2.sh:30-36` usa stride SARIMA 14; degenerados eran solo `NaN` | varianza inadecuada para series dispersas y falta de trazabilidad | HAC Bartlett, lag derivado del stride, HLN configurable y estados explícitos en `scripts/05_dm_significance.py:27-126` |
| P1-02-004 | CRITICAL | `outputs/metrics/predictions.csv` contiene 3 modelos y `skill_summary.csv` 2, pero `outputs/tables/variance_retention_summary.csv` contiene 5; el patrón se repite en Valencia y Zarra | las tablas congeladas no se pueden reproducir desde los inputs presentes | Pendiente: recuperar inputs/manifiestos canónicos; no se sobrescribieron resultados |
| P1-02-005 | MAJOR | solo hay 3 tablas de predicciones y ninguna salida `dm_*.csv`, mientras `scripts/build_overleaf_consistency_prompt.py:518` refiere 17 estaciones | imposible verificar mecánicamente el universo inferencial publicado | Pendiente: aportar las 17 predicciones por estación y salidas/configuración correspondientes |
| P1-02-006 | INFO | `paper_a.tex:91`; búsquedas exhaustivas sin implementación H*/Yule–Walker | no hay mezcla observable entre skill, significación, censura y referencia lineal | Sin cambio; se documenta ausencia |
| P1-02-007 | MINOR | `main` estaba un commit detrás de `origin/main` | el HEAD auditado no es la punta remota conocida | Se fijó y registró el HEAD; no se hizo pull ni merge |

## Reparaciones

### P1-02-001

- Archivo: `src/diagnostics/variance.py`
- Cambio: bootstrap circular por bloques de origen, semilla y réplicas
  explícitas, regla de bloque documentada, descarte explícito de réplicas
  inválidas.
- Minimalidad: conserva el mismo estimando `alpha`, el mismo IC percentil y el
  esquema de salida; solo cambia la unidad de remuestreo.

### P1-02-002

- Archivos: `src/evaluation/pairing.py`,
  `scripts/01_generate_e1_rr_lags_only_predictions.py`,
  `scripts/02_generate_sarima_predictions.py`,
  `scripts/05_dm_significance.py`.
- Cambio: una única intersección validada con claves temporales completas,
  control de duplicados, `y_true` común, orden cronológico y eliminación
  conjunta de no finitos.
- Minimalidad: no cambia modelos, folds, horizontes ni pérdidas.

### P1-02-003

- Archivo: `scripts/05_dm_significance.py`
- Cambio: HAC Bartlett con denominador `n`, lag explícito dependiente del stride,
  HLN activable/desactivable, estados degenerados y columnas de trazabilidad.
- Minimalidad: conserva DM bilateral, pérdida cuadrática, t de Student y BH por
  dataset.

No se modificó `paper_a.tex`, ningún archivo Overleaf ni resultados canónicos.

## Tests y ejecución

Primera ejecución dirigida:

```text
python -m pytest -q tests/test_temporal_inference.py tests/test_skill.py \
  tests/test_variance.py tests/test_variance_retention_schema.py \
  tests/test_rolling_origin.py
```

Resultado inicial: 20 aprobados, 2 fallidos. Los fallos identificaron un fixture
con `y_true` incoherente y la autocovarianza con denominador `n-lag`; ambos
fueron corregidos. Repetición: **22 aprobados, 0 fallidos**.

Suite completa:

```text
python -m pytest -q
```

Resultado final: **28 aprobados, 0 fallidos, 0 omitidos**.

Validación diagnóstica en memoria, sin escribir resultados:

- `predictions.csv`: 14 DM, todos `ok`, 1215–1274 pares.
- `predictions_valencia_vivers.csv`: 35 DM, 34 `ok` y 1
  `all_loss_differentials_equal`, 98–1479 pares.
- `predictions_zarra_emep.csv`: 35 DM, 34 `ok` y 1
  `all_loss_differentials_equal`, 99–1463 pares.
- Sensibilidad de bloque para `hgb_direct, h=7`, 300 réplicas y bloques
  5/11/22: IC `(0.05834, 0.19244)`, `(0.05477, 0.18439)` y
  `(0.05632, 0.20919)`.

No se ejecutó el pipeline completo: requeriría regenerar resultados congelados,
acción prohibida en PROG-P1-02, y faltan los inputs de 17 estaciones.

## Resultados potencialmente obsoletos

Todos los IC `alpha_ci_low/high` generados con el bootstrap IID y todos sus
consumidores deben regenerarse en PROG-P1-03 después de resolver los blockers:

- `outputs/tables/variance_retention_*.csv`;
- tablas unificadas de variance-retention;
- `outputs/audit/bootstrap_alpha_audit.{csv,md}`;
- figuras y textos que usen los IC de alpha;
- `outputs/metrics/dm_*.csv` y la tabla DM unificada, por el nuevo HAC/lag/estado;
- resúmenes que consumen `dm_significant`.

No se regeneró ninguno en esta auditoría.

## Blockers pendientes

1. Identificar el repositorio/manuscrito canónico de
   `P1 / predictability-bound`, o confirmar explícitamente que este Paper A/P33
   es el objetivo correcto.
2. Aportar las predicciones canónicas de las 17 estaciones, con configuración,
   manifiestos y hashes.
3. Explicar o sustituir las tablas `variance_retention_*.csv` incompatibles con
   los CSV de predicción y skill presentes.
4. Aportar o regenerar en PROG-P1-03 las salidas DM; actualmente no existe
   ninguna salida DM congelada que auditar.

## Estado de ruta crítica

`PROG-P1-03_NOT_ALLOWED`

La ejecución de P1-03 solo puede autorizarse tras resolver los blockers 1–3 y
confirmar que el conjunto canónico de inputs queda trazado.
