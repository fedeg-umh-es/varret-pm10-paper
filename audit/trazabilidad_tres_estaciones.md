# Auditoría de trazabilidad — Elche, Valencia-Vivers, Zarra

Modo: **solo lectura** sobre artefactos científicos. No se ejecutó pipeline,
bootstrap, entrenamiento, inferencia ni scripts de métricas/tablas. Los únicos
ficheros nuevos creados están bajo `audit/`.

## 1. Entorno Git

| Campo | Valor |
|---|---|
| Ruta local | `/home/user/varret-pm10-paper` |
| Repositorio | `fedeg-umh-es/varret-pm10-paper` |
| Remote origin | `http://local_proxy@127.0.0.1:41729/git/fedeg-umh-es/varret-pm10-paper` (proxy de sesión → `github.com/fedeg-umh-es/varret-pm10-paper`) |
| Rama actual | `claude/auditoria-hstar-censura-cota-4sqaa5` (rama del PR #7; **no** `main`) |
| Commit SHA (completo) | `25f124d8c1ebbb06133b39d0ecd369059de15957` |
| Commit SHA (corto) | `25f124d` |
| Fecha del commit | `2026-07-23T11:33:16+00:00` |
| Working tree (antes) | **limpio** (`git status --short` sin salida) |

Nota: los seis ficheros de datos son idénticos a los de `main` (no fueron
tocados por los commits de manuscrito del PR #7). La **procedencia canónica** de
cada fichero se registra abajo como "último commit que lo modificó", que es
independiente de la rama inspeccionada.

## 2. Tabla principal (3 filas)

| station | raw_data_path | raw_exists | raw_sha256 | raw_size_bytes | raw_rows | predictions_path | predictions_exists | predictions_sha256 | predictions_size_bytes | predictions_rows | source_repo | branch | commit_sha | working_tree_status | hash_status | observations |
|---|---|---|---|---:|---:|---|---|---|---:|---:|---|---|---|---|---|---|
| Elche | `data/raw/pm10_daily.csv` | yes | `5ab9cd34f6d6764d6b96fea0cdd7980878155de3f359b7db56206dfefa401726` | 82827 | 2350 | `outputs/metrics/predictions.csv` | yes | `915a821ca74c7d51507d8b6cd3420c8d141ec26f1478a580d9792398641b58ee` | 2027461 | 26001 | fedeg-umh-es/varret-pm10-paper | claude/auditoria-hstar-censura-cota-4sqaa5 | 25f124d8c1ebbb06133b39d0ecd369059de15957 | clean | **VERIFICADO** | raw last-touched e7e40fd (2026-05-22); pred last-touched e7e40fd (2026-05-22). Raw incluye temp,hr,ws,wd (no usados). |
| Valencia-Vivers | `data/raw/pm10_valencia_vivers.csv` | yes | `3eabf0e518c07abc608ba434e349731e840e0c951ace4c773cc52d36594533cf` | 42738 | 2679 | `outputs/metrics/predictions_valencia_vivers.csv` | yes | `bfb21ff09652c390daf15b72cad2086fe13eacdd85d9e5e736a87e2d9b51890a` | 4606894 | 52028 | fedeg-umh-es/varret-pm10-paper | claude/auditoria-hstar-censura-cota-4sqaa5 | 25f124d8c1ebbb06133b39d0ecd369059de15957 | clean | **VERIFICADO** | raw last-touched 975336c (2026-05-23); pred last-touched e2120f8 (2026-05-25). |
| Zarra | `data/raw/pm10_zarra_emep.csv` | yes | `f4630d0df551da201011f91d24c8811f87311c1426f256f610f140f2160259ec` | 43550 | 2804 | `outputs/metrics/predictions_zarra_emep.csv` | yes | `601440c364f266696c8ed8e8aa4a368a8150a4da4e27a5b9a315bcca52fce1a7` | 4326018 | 52136 | fedeg-umh-es/varret-pm10-paper | claude/auditoria-hstar-censura-cota-4sqaa5 | 25f124d8c1ebbb06133b39d0ecd369059de15957 | clean | **VERIFICADO** | raw last-touched 975336c (2026-05-23); pred last-touched e2120f8 (2026-05-25). |

Notas de conteo: `raw_rows` y `predictions_rows` = líneas físicas − 1 (todas las
CSV tienen cabecera). Líneas físicas contadas con `awk 'END{print NR}'` (cuenta
la última línea aunque no tenga salto final). SHA-256 con `sha256sum` sobre los
bytes originales del fichero.

Git blob SHA (referencia, no confundir con SHA-256 de contenido):
Elche raw `964af58`, pred `3b997c6`; Valencia raw `f8d1bb7`, pred `d8feaa7`;
Zarra raw `158269e`, pred `cd1eafa`.

## 3. Subtabla — esquema de predicciones

Las tres tablas de predicciones comparten cabecera idéntica:
`dataset,model,fold,origin_date,horizon,date,y_true,y_pred` (8 columnas). El
mapeo campo-requerido → columna es, por tanto, el mismo para Elche,
Valencia-Vivers y Zarra:

| station | required_field | detected_column | status | observation |
|---|---|---|---|---|
| (las 3) | station | `dataset` | AMBIGUO | no hay columna literal `station`; `dataset` codifica el id estación-dataset (p.ej. `e1_rr_daily`) |
| (las 3) | origin / forecast origin | `origin_date` | PRESENTE | `fold` coincide con el origen en esta tabla |
| (las 3) | forecast timestamp | `date` | PRESENTE | `date` es la fecha objetivo del pronóstico |
| (las 3) | horizon | `horizon` | PRESENTE | |
| (las 3) | model | `model` | PRESENTE | incluye el valor de modelo `persistence` |
| (las 3) | baseline / persistencia | — | AUSENTE | la persistencia es un valor de `model`, no una columna separada |
| (las 3) | y_true | `y_true` | PRESENTE | |
| (las 3) | y_pred | `y_pred` | PRESENTE | |
| (las 3) | train_end | — | AUSENTE | |
| (las 3) | fold / split | `fold` | PRESENTE | |
| (las 3) | valid_pair | — | AUSENTE | |

## 4. Incidencias

- Ninguna que impida el cierre. Los tres pares raw+predictions existen, son
  legibles, versionados y con SHA-256 calculado.
- La rama inspeccionada es la del PR #7, no `main`; los ficheros de datos son
  idénticos a `main` (último commit que los tocó es anterior al PR #7).

## 5. Limitaciones

- El remote es el proxy interno de la sesión; mapea al repo canónico de GitHub
  indicado.
- No se verificó el contenido numérico de las predicciones (fuera de alcance:
  solo cabeceras y conteos).

## 6. Comandos ejecutados (solo lectura)

`git remote get-url origin`, `git branch --show-current`, `git rev-parse [--short] HEAD`,
`git log -1 --format`, `git status --short`, `git ls-files --error-unmatch`,
`git rev-parse HEAD:<file>`, `stat -c`, `file -b`, `sha256sum`,
`awk 'END{print NR}'`, `head -1 | awk -F','`.

## 7. Veredicto

```text
RESULTADO DE AUDITORÍA
Elche: VERIFICADO
Valencia-Vivers: VERIFICADO
Zarra: VERIFICADO

BOOTSTRAP EJECUTADO: NO
TABLAS REGENERADAS: NO
MODELOS REENTRENADOS: NO
PREDICCIONES REGENERADAS: NO
ARTEFACTOS CIENTÍFICOS MODIFICADOS: NO

COMMIT AUDITADO:
25f124d8c1ebbb06133b39d0ecd369059de15957

ARCHIVOS DE AUDITORÍA CREADOS:
- audit/trazabilidad_tres_estaciones.md
- audit/trazabilidad_tres_estaciones.csv
- audit/trazabilidad_tres_estaciones.json
```
