# P3-12R — Informe final de consolidación de artefactos

## 1. Fuente recuperada

`/home/fede/investigacion/repos/varret-pm10-paper` (identificada en P3-11R;
índice git corrupto, nunca respaldada formalmente hasta esta tarea). No se
modificó nada en esa ruta en ningún momento de P3-12R.

## 2. Backup

`/home/fede/backups/p3-12r-recovered-evidence/source-copy/` — rsync de solo
lectura, 624 archivos, excluyendo `.git`, `__pycache__`, `.venv`, `venv`.
Manifiesto completo en `source_manifest.csv` / `.sha256`.

## 3. Hashes

Todos los artefactos recuperados fueron hasheados (SHA-256) antes y después
de cada copia; **100% de coincidencia** entre origen → backup → estructura
gobernada `evidence/paper_a/` → rutas operacionales `outputs/tables/`. Ver
`evidence/paper_a/manifests/provenance_manifest.csv`.

## 4. Política de almacenamiento

6 tablas agregadas + 14 tablas por estación → `TRACK_IN_GIT` (todas < 432 KiB).
`predictions_all_stations.csv` (77 MiB, 895,737 filas) → `PACKAGE_AS_RELEASE_ASSET`
(evita casi duplicar permanentemente el `.git` del repo). Ver `storage_policy.csv`.

## 5. Artefactos copiados

`evidence/paper_a/{aggregates,source_tables,metadata,manifests,row_level}/`
(27 archivos) + copias operacionales en `outputs/tables/` (20 archivos nuevos,
para que los scripts existentes sigan funcionando sin modificar sus rutas
hardcodeadas).

## 6. Row-level preservation

`predictions_all_stations.csv` preservado en 3 ubicaciones de solo lectura
(backup rsync, copia gitignored de P3-11R, origen intacto) y subido como
**release draft** de GitHub (`paper-a-row-level-evidence-v1`, sin publicar),
asset de 77,080,400 bytes verificado. No se incorporó al historial de git.

## 7. Cobertura

17/17 estaciones, 7/7 horizontes, 5/5 familias de modelo evaluadas + 1
baseline de persistencia, 119 celdas por familia, 595 celdas totales,
895,737 filas row-level. Todo verificado, nada alterado.

## 8. Recomputación

Entorno Python aislado (`/tmp/p3-12r-venv`, numpy/pandas/matplotlib/scipy/
pyarrow/seaborn/pytest; entorno global del sistema intacto, sin `pandas`).
`alpha`, `n`, `collapse_flag`, `near_ideal_flag` recomputados directamente
desde `predictions_all_stations.csv` agrupando por (estación, modelo,
horizonte) — coinciden con `variance_retention_all_stations.csv` en 595/595
celdas. `skill` coincide exacto en 476/595 (HGB, Ridge, seasonal_naive,
STL+Ridge); SARIMA (119 celdas) difiere por una causa metodológica
identificada (su rango temporal se extiende a 2018-2019, fuera de la
cobertura del baseline de persistencia compartido en el CSV row-level) — no
es un problema de los datos. Ver `recomputed_diagnostics.csv` y
`recomputation_comparison.csv`.

## 9. Tolerancia

1e-10 aplicada. `alpha`: diferencia máxima 4.44e-16 (precisión de máquina,
muy por debajo de la tolerancia). `n`, `collapse_flag`, `near_ideal_flag`:
coincidencia exacta (diferencia 0) en 595/595 celdas.

## 10. Excepciones nombradas

Huesca–HGB–h1: alpha reportado 0.544090062862485, recomputado
0.5440900628624852 (diff 2.22e-16). Barcelona Vall d'Hebron–Ridge–h1: alpha
reportado 0.5024007945076104, recomputado idéntico (diff 0.0). Ambas
`collapse_status` coinciden (False). Ver `named_exception_audit.csv`.

## 11. Figura 5

**Regenerada** (no solo "regenerable"): `figure5_scatter_skill_alpha.pdf` y
`figure_skill_alpha_five_models.pdf`, desde el CSV validado, con el script ya
corregido (Var/Var). Inspección visual confirma: mismos puntos, mismas
trayectorias, mismos cuadrantes, misma leyenda; único cambio observable es
el texto del rótulo del eje X. `status=PASS` en ambas. Ver
`figure5_regeneration_audit.csv` y `pdf_visual_audit.md`.

## 12. Tablas

`model_family_diagnostic_summary.{csv,tex,md}` y `new_story_decision.md`
regenerados y comparados byte a byte contra las versiones publicadas: 0
`NUMERIC_MISMATCH`. Único hallazgo: 1 celda `ROUNDING_ONLY` a nivel ULP en
el CSV, y diferencias `FORMATTING_ONLY` (envoltura landscape + caption ya
preexistente) en el `.tex` de la raíz. Ver `table_regeneration_audit.csv`.

## 13. Data availability

Actualizada en `paper_a_ems.tex`, `submission_package/ems/paper_a_ems.tex`,
`data_availability_statement.txt` para distinguir explícitamente: datos
crudos (públicos, MITECO), datos versionados (aggregates/source_tables en
`evidence/paper_a/`), y predicciones row-level (archivadas localmente con
hash documentado, preparadas como release asset, **no** públicas todavía —
redacción corregida para no sobre-afirmar disponibilidad mientras el release
sigue en borrador). Ver `data_availability_audit.csv`.

## 14. Code availability

Actualizada para citar la verificación de recómputo (entorno Python,
paquetes exactos, tolerancia alcanzada) además del repositorio de scripts.

## 15. Compilación

`paper_a.tex` → `paper_a.pdf`, exit 0, 0 referencias/citas indefinidas, 17
páginas, sha256 `6e462fb98a99a935973902b96c5f4d857e3d1df8cdb8a555ae5812d9f610cec8`.
`paper_a_ems.tex` → `paper_a_ems.pdf` (vía `submission_package/ems/`, tras
corregir 2 referencias rotas preexistentes), exit 0, 0 indefinidas, 17
páginas, sha256 `d631e198ab7dabaa619830d3b882e76e1d15daf17d11d0cfdec4de84109c64e0`.
`supplementary_material_ems.tex` no es standalone; validado como parte de la
compilación exitosa de `paper_a_ems.tex`.

## 16. Inspección visual

Portada y página de Figura 5 de ambos PDFs inspeccionadas: sin clipping, sin
símbolos corruptos, leyendas y referencias cruzadas correctas, rótulo
Var/Var confirmado en ambos documentos. Ver `pdf_visual_audit.md`.

## 17. Paquete EM&S

`submission_package/ems/` corregido (2 bugs de referencia preexistentes que
impedían compilar), `file_manifest.csv` regenerado (34 archivos, antes solo
15), figuras 5 y five-models sincronizadas con las versiones regeneradas,
artefactos de compilación (.aux/.log/etc.) limpiados. `predictions_all_stations.csv`
explícitamente NO incluido. `overall_package_status=PASS`. Ver
`submission_package_audit.csv`.

## 18. Clasificación A/B/C

**A_COMPLETE.** Las 9 condiciones de la tarea se cumplen (row-level
preservada, hashes, procedencia, source tables, scripts, Figura 5
regenerada, tablas regeneradas, números coincidentes, paquete editorial
consistente). Ver `evidence_status_final.json`, incluyendo el "invariance
check" que confirma que ningún número canónico ni claim científico fue
tocado.

## 19. Blockers

Ninguno bloqueante para `SUBMISSION_GO`. Pendientes no bloqueantes: (a)
recómputo independiente de DM/exceedance/Murphy (solo esquema validado, no
valores); (b) publicación efectiva del release draft; (c) fusión de las
ramas/PRs a `main` (decisión humana); (d) reparación del índice git corrupto
de la copia recuperada (no intentada, fuera de alcance).

## 20. Decisión de submission

**`SUBMISSION_GO`** — puerta de evidencia/reproducibilidad técnica despejada.
No implica envío del manuscrito ni fusión de PRs (ninguno de los dos se
realizó). Ver `submission_decision.json` para el razonamiento completo y los
límites explícitos de alcance de esta decisión.

## 21. Outputs

Los 16 archivos requeridos generados en `outputs/p3_12r_artifact_consolidation/`
(ver `README.md` de esa carpeta para el índice), más la estructura
`evidence/paper_a/` consolidada.

## 22. Rama

`evidence/p3-12r-consolidate-recovered-artifacts`, creada desde un worktree
aislado (`/tmp/p3-12r-consolidation`) a partir de
`origin/audit/p3-11r-evidence-reconciliation`, con un merge adicional de
`origin/editorial/p3-08-ems-adaptation` para alcanzar `paper_a_ems.tex` /
`submission_package/ems/`.

## 23. Commit

`5665165` — "Consolidate recovered Paper A evidence artifacts" (85 archivos,
+13,466/-21 líneas). Precedido por el merge commit `fa6c5be`.

## 24. Push

`git push -u origin evidence/p3-12r-consolidate-recovered-artifacts` — exitoso.

## 25. PR

[#15](https://github.com/fedeg-umh-es/varret-pm10-paper/pull/15) creado como
draft, base `audit/p3-11r-evidence-reconciliation`, sin fusionar.
[#14](https://github.com/fedeg-umh-es/varret-pm10-paper/pull/14) actualizado
(descripción ampliada con el resultado de la consolidación y enlace a #15),
sigue draft/open/sin fusionar. Además: release draft
`paper-a-row-level-evidence-v1` creado (sin publicar) con el asset
`predictions_all_stations.csv`.

## 26. Árbol final

`git status --short` → limpio (sin cambios pendientes) tras el commit. Único
estado residual: el release de GitHub permanece en borrador (no publicado) y
las 2 PRs permanecen sin fusionar, ambos por diseño.

## 27. Siguiente acción mínima

Revisión humana de #15 → decidir si fusionar #15 en #14, luego #14 en
`main`, y si publicar el release `paper-a-row-level-evidence-v1`. Ninguna
acción adicional de datos/reentrenamiento es necesaria.
