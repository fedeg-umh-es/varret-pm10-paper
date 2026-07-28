# `predictions_all_stations.csv` — manifiesto (no versionado directamente en git)

## Por qué no está en este directorio

77,080,400 bytes (≈73.5 MiB), 895,737 filas. Según `storage_policy.csv`, este
archivo se clasifica `PACKAGE_AS_RELEASE_ASSET`: comprometerlo al historial de
git casi duplicaría permanentemente el tamaño del repositorio (`.git` actual
≈93 MiB) sin posibilidad de reducirlo después sin reescribir historia (acción
prohibida en esta tarea).

## Identidad del archivo

| Campo | Valor |
|---|---|
| Nombre original | `predictions_all_stations.csv` |
| SHA-256 | `2551b3a2acf549e94cd92e39386b96aca2358cca41fed7f85c933d35c77ef823` |
| Tamaño | 77,080,400 bytes |
| Filas | 895,737 (+ 1 cabecera) |
| Columnas | `dataset,model,fold,origin_date,horizon,date,y_true,y_pred` |
| Cobertura | 17 datasets (estaciones), 7 horizontes, 6 valores de `model` (5 modelos evaluados + `persistence`) |
| Origen | RECOVERED FROM LOCAL SOURCE COPY — `/home/fede/investigacion/repos/varret-pm10-paper/outputs/metrics/predictions_all_stations.csv` |
| Fecha de descubrimiento | 2026-07-28 (tarea P3-11R) |

## Dónde encontrarlo

1. **Copia de respaldo de solo lectura** (verificada, mismo hash):
   `/home/fede/backups/p3-12r-recovered-evidence/source-copy/outputs/metrics/predictions_all_stations.csv`
2. **Copia local en el repositorio de trabajo** (gitignored, no versionada):
   `outputs/p3_11r_evidence_reconciliation/recovered_candidates/predictions_all_stations.csv`
   (tarea P3-11R) — mismo hash.
3. **Release asset de GitHub** (mecanismo de distribución gobernado, ver
   sección siguiente).

## Release asset

- Repositorio: `fedeg-umh-es/varret-pm10-paper`
- Ver `outputs/p3_12r_artifact_consolidation/submission_decision.json` y
  `final_report.md` para el estado exacto (creado como release **draft**,
  pendiente de publicación explícita por el responsable del proyecto — una
  release publicada es contenido público inmediato y esta tarea prefiere
  dejarla en borrador para revisión antes de esa publicación, igual que el
  PR #14 se mantiene en draft).
- Tag: `paper-a-row-level-evidence-v1`
- Verificar integridad tras descargar:
  ```bash
  sha256sum predictions_all_stations.csv
  # debe coincidir con: 2551b3a2acf549e94cd92e39386b96aca2358cca41fed7f85c933d35c77ef823
  ```

## Cómo se usó en esta consolidación

Se leyó de forma completa (una pasada, sin pandas) para recomputar
`alpha = Var(ŷ)/Var(y)` (ddof=0) y `skill` por celda
`(dataset,model,horizon)`, sin entrenar ni modificar nada. El resultado
coincide con `aggregates/variance_retention_all_stations.csv` con una
diferencia máxima de 1.68e-13 en las 595 celdas. Ver
`outputs/p3_12r_artifact_consolidation/recomputation_comparison.csv`.

## Limitación conocida: `skill` de SARIMA no es recomputable celda-a-celda desde este archivo

**Causa técnica.** Las filas `model=="persistence"` de este CSV cubren
2020-01-01 a 2024-12-30 (rango del script base,
`01_generate_e1_rr_lags_only_predictions.py --start-year 2020 --end-year
2024`). Las filas `model=="sarima"` cubren 2018-01-17 a 2024-12-30 (rango
completo de la serie, generado por `scripts/02_generate_sarima_predictions.py
--origin-step 14`, sin restricción de años). Ese mismo script genera
internamente sus propias filas de persistencia sobre el rango completo
2018-2024 para emparejar con SARIMA (`_predict_one_origin`,
`_build_skill_summary`, inner join exacto por `origin_date`/`date`), pero
`scripts/combine_prediction_tables.py` descarta esas filas de persistencia
propias de SARIMA al combinar las tablas row-level (`sarima_only =
sarima_predictions[model == "sarima"]`), para no duplicar persistencia ya
presente desde el script base. El resultado: ~30-45% de los pares
`(origen, fecha_objetivo)` de SARIMA (los anteriores a 2020) no tienen
contraparte de persistencia en este CSV.

**Archivo fuente canónico.** El `skill` publicado para SARIMA no proviene de
este CSV row-level, sino de `skill_sarima_{station}.csv` (salida directa de
`scripts/02_generate_sarima_predictions.py`), donde el emparejamiento SARIMA
vs. persistencia es exacto (mismo origen, misma fecha objetivo, mismo
horizonte) sobre el rango completo 2018-2024. `scripts/
combine_prediction_tables.py` conserva ese valor de skill (concat +
`drop_duplicates(keep="last")`), por lo que la cifra publicada en
`aggregates/variance_retention_all_stations.csv` es correcta y está
correctamente emparejada; simplemente no es re-derivable solo a partir de
`predictions_all_stations.csv`.

**Alcance.** 119 de 595 celdas (17 estaciones × 7 horizontes = toda la
rejilla SARIMA). Afecta únicamente a `skill`. `alpha`, `collapse_flag` y
`near_ideal_flag` de SARIMA coinciden con precisión de punto flotante
(diferencia máxima 8.88e-16) porque no dependen de la persistencia ni de
este emparejamiento.

**Sin impacto en los claims centrales.** El recómputo restringido al solape
2020-2024 reproduce una mediana de skill SARIMA (0.2098) muy cercana a la
citada en el manuscrito (0.208); no cambia el signo, el ranking entre
modelos, ni los conteos de colapso (110/119) o near-ideal (0/119) citados en
`paper_a_ems.tex`.

**Limitación exacta del release row-level.** `predictions_all_stations.csv`
(y el release `paper-a-row-level-evidence-v1` que lo distribuye) permite
reproducir de forma independiente y completa: `alpha`, `collapse_flag` y
`near_ideal_flag` para las 595 celdas, y `skill` para 476/595 celdas (los
cuatro modelos no-SARIMA). No permite reproducir de forma independiente el
`skill` celda-a-celda de SARIMA (119 celdas); para verificar esas cifras es
necesario `skill_sarima_{station}.csv`, que no forma parte de este release.
