# P3-11R — Informe final de reconciliación de evidencia (Paper A)

## 1. Estado inicial

Repositorio: `/home/fede/repos/varret-pm10-paper`, rama `main`, commit
`2939d26fb4e3447347ed9d6ba8e2d244462cb67b`, árbol de trabajo limpio salvo dos
cambios de la tarea P3-11 previa (rótulo Var/Var en 2 scripts de figuras, sin
commitear) y `paper_a.pdf` recompilado. Ver `local_state_before/git_status.txt`.

## 2. Backups

- `outputs/p3_11r_evidence_reconciliation/local_state_before/` — snapshot de
  git status/diff + manifiesto con hash de `dist/`, `outputs/`, `results/`,
  `data/`, `scripts/`, `paper_a.tex/pdf` (226 archivos).
- `outputs/p3_11r_evidence_reconciliation/recovered_candidates/` — copias de
  solo lectura, hash verificado, de los 7 artefactos críticos recuperados desde
  `/home/fede/investigacion/repos/varret-pm10-paper`.
- Backup completo del repo principal de la tarea P3-11 previa sigue disponible
  en `/home/fede/repos/varret-pm10-paper-local-backup-20260728-180016/`.

## 3. Refs inspeccionadas

`git fetch --all --tags --prune` ejecutado. 16 ramas remotas, 5 ramas locales,
1 tag (`v1.0.0`), 4 worktrees registrados (3 prunable/inexistentes en disco).
`git fsck --full --no-reflogs --unreachable`: 1 commit inalcanzable (amend
menor, sin relevancia). Detalle en `repository_ref_inventory.csv`.

## 4. Commits encontrados

De los 5 SHA solicitados explícitamente: 3 existen (`8f7e345...` — evidence
package; `2939d26...` — HEAD de main; `e8fe4d6` — EMS adaptation), 2 no existen
en ningún lugar (`570959d3...` del prompt P3-11 original, y
`0dcac8031dcfd547fbb12e752bca1c3b53f65e20` — nota: SÍ existe un commit similar
`0dcac80fdf91a5d09f80d43f9d13fa425aae4630`, con sufijo distinto al solicitado).

## 5. Ramas encontradas

`paper/paper-a-evidence-package` (PR #11, draft) es la más relevante por nombre;
contiene documentación agregada (`paper_package/paper_a/*`) pero **no** datos
row-level — ver sección 8. `editorial/p3-08-ems-adaptation` (PR #13, draft) y
`editorial/p3-07-journal-routing` (PR #12, draft) son adaptaciones editoriales
sin datos nuevos.

## 6. PRs y artifacts inspeccionados

13 PRs (#1–#13) listados vía `gh pr list --state all`. 8 workflow runs, todos
`failure` en 3–5s (fallos de configuración de CI de mayo, no ejecuciones de
pipeline). 0 workflow artifacts (`gh api .../actions/artifacts` →
`total_count=0`). 1 release (`v1.0.0`, sin assets relevantes). Code search
global de GitHub: 0 resultados para `variance_retention_all_stations.csv`, 1
resultado irrelevante (repo no relacionado) para `master_diagnostic_table.csv`.

## 7. Artefactos locales

Búsqueda sin límite de profundidad en `/home/fede`, `/tmp` (excluyendo `.git`,
venvs, cachés) encontró el hallazgo decisivo de esta tarea: **todos** los
artefactos objetivo presentes en
`/home/fede/investigacion/repos/varret-pm10-paper/`, un directorio local nunca
antes inspeccionado en la tarea P3-11 previa (que solo buscó en
`/home/fede/repos`). Detalle en `local_artifact_inventory.csv`.

## 8. Candidatos

- `variance_retention_all_stations.csv` (161,572 bytes, 595 filas)
- `predictions_all_stations.csv` (77,080,400 bytes, 895,737 filas row-level)
- `master_diagnostic_table.csv` (415,591 bytes, 595 filas, 47 columnas)
- `dm_significance_all_stations.csv`, `exceedance_all_stations.csv`,
  `murphy_decomposition_all_stations.csv`, `alpha_threshold_sensitivity.csv`
- 14 tablas `variance_retention_<station_id>.csv` adicionales (las que faltaban
  en el repo git principal)
- Descartado: `paper_package/paper_a/canonical_numbers.csv` (rama
  `paper/paper-a-evidence-package`) — es documentación agregada, no evidencia
  row-level; contiene además una inconsistencia interna (su
  `reproducibility_manifest.json` señala `variance_retention_summary.csv` como
  "per_cell_table" pese a que ese archivo tiene solo 35 filas / 1 estación en
  todo su historial de git).

## 9. Validación de candidatos

Todos los candidatos "ALL_STATIONS" tienen el esquema correcto. El candidato
principal (`variance_retention_all_stations.csv`) fue validado con precisión
total: 595 filas, 17 estaciones, 5 modelos, 7 horizontes, 0 duplicados, 0
valores no finitos, y coincidencia EXACTA con todas las cifras canónicas
citadas en la tarea (colapsos 118/118/110 de 119, medianas de alpha
0.150627/0.087430/0.095113/0.999781/1.398663, near-ideal 0/0/0/0/20, y las
excepciones Huesca HGB h=1 α≈0.544090 y Barcelona Vall d'Hebron Ridge h=1
α≈0.502401 — coincidencia hasta el último dígito). Detalle en
`candidate_validation.csv`.

## 10. Disponibilidad row-level

**SÍ disponible.** `predictions_all_stations.csv`: 895,737 filas,
columnas `dataset,model,fold,origin_date,horizon,date,y_true,y_pred`. Incluye
`model=="persistence"` como baseline explícito (mismo patrón ya validado por la
auditoría previa PR #9 para las 3 estaciones). Cubre los campos requeridos por
la regla científica salvo una columna explícita de "persistence forecast"
separada (está codificada como valor de `model`, igual que en el corpus de 3
estaciones ya certificado) y metadata explícita de fuente por fila (existe a
nivel de tabla agregada, no en el CSV row-level).

## 11. Cobertura de estaciones

17/17. IDs verificados: `03014002_10_M, 08019004_10_M, 08019028_10_M,
08019043_10_M, 08019045_10_M, 08019052_10_M, 08019054_10_M, 08263007_10_M,
22125001_10_M, 43004005_10_M, 43004006_10_M, 44013007_10_M, 44216001_10_M,
45153999_10_M, 46250043_10_M, 46263999_10_M, 50008001_10_M`.

## 12. Cobertura de horizontes

7/7 (h=1..7).

## 13. Cobertura de modelos

5/5 modelos evaluados (`hgb_direct, ridge_direct, sarima, seasonal_naive,
stl_ridge_direct`) + `persistence` como baseline = 6 valores de `model` en el
CSV row-level.

## 14. Recomputación

Ejecutada sin entrenar nada: lectura completa de `predictions_all_stations.csv`
(895,737 filas) con Python estándar (csv + math, sin pandas), acumulación de
sumas por celda `(dataset,model,horizon)`, cálculo directo de
`alpha=Var(ŷ)/Var(y)` (ddof=0) y `skill=1-RMSE_modelo/RMSE_persistencia`.
Resultado: alpha coincide con el CSV agregado a precisión de punto flotante
(diff máximo 1.68e-13) en las 595 celdas; skill coincide exactamente para 4/5
modelos y difiere solo para SARIMA por una causa metodológica identificada y
documentada (cadencia de origen dispersa, no un problema de los datos). Detalle
completo en `recomputed_diagnostics.csv`.

## 15. Figura 5

Clasificación: **REGENERABLE** (con el PDF actualmente publicado en estado
`STALE_LABEL`: datos correctos, rótulo de eje todavía en notación SD/SD
antigua). El script fuente ya fue corregido a Var/Var (cambio realizado en la
tarea P3-11 previa, sin commitear). El bloqueo restante es puramente de
entorno (falta pandas/numpy/matplotlib), no de datos. Detalle en
`figure5_audit.md`.

## 16. Definición de alpha

Confirmada como `Var(ŷ)/Var(y)`, ddof=0, sin cambios respecto a la definición
científica bloqueada. `src/diagnostics/variance.py` ya la implementa
correctamente; no se tocó.

## 17. Clasificación A/B/C

**A_COMPLETE** en cuanto a existencia y verificabilidad de la evidencia (ver
`evidence_classification.json`), con la salvedad crítica de que esa evidencia
no está consolidada en el repositorio git versionado.

## 18. Claims permitidos

Ver `submission_decision.json` → `claims_allowed`. En síntesis: todos los
números y definiciones ya publicados en `paper_a.tex` sobre el corpus de 17
estaciones están respaldados por evidencia ahora verificada.

## 19. Claims prohibidos

Ver `submission_decision.json` → `claims_prohibited`. En síntesis: no declarar
el pipeline como reproducible desde el repositorio público, no declarar Figura
5 como corregida, no dar por verificados los valores de DM/exceedance/Murphy
(solo su esquema), no enviar el manuscrito.

## 20. Opciones

5 opciones evaluadas en `decision_options.csv`. Recomendada: **"recover
17-station artifacts"** (consolidar lo ya encontrado y verificado; bajo-medio
esfuerzo, alta fortaleza científica, bajo riesgo editorial, sin cambios
narrativos al manuscrito).

## 21. Decisión de submission

**`SUBMISSION_HOLD_RECOVERABLE`**. Ver `submission_decision.json` y
`decision_log_entry.md` (Decision ID
`2026-07-28-paper-a-evidence-reconciliation`).

## 22. Outputs

Todos los 13 archivos/directorios requeridos generados en
`outputs/p3_11r_evidence_reconciliation/` (ver `README.md` para el índice).

## 23. Rama

`audit/p3-11r-evidence-reconciliation` (creada para esta auditoría; ver sección
siguiente para el estado de versionado).

## 24. Commit

Ver estado de versionado al final de esta tarea (commit "Reconcile Paper A
multi-station evidence before submission", solo informes + correcciones de
rotulación).

## 25. Push

Realizado a `origin/audit/p3-11r-evidence-reconciliation` (ver confirmación al
cierre de la tarea).

## 26. PR

PR draft de auditoría creado contra `main`, sin fusionar.

## 27. Blockers

1. Entorno Python local sin `pandas`/`numpy`/`matplotlib` — bloquea la
   regeneración real de Figura 5 (no bloquea la evidencia en sí).
2. Índice git corrupto en `/home/fede/investigacion/repos/varret-pm10-paper`
   (ficheros AppleDouble de macOS) — impide una cadena de procedencia git
   limpia para la evidencia recuperada.
3. La evidencia recuperada no está consolidada en el repositorio git principal
   — requiere una decisión explícita del responsable del proyecto antes de
   incorporarla.
4. DM/exceedance/Murphy encontrados pero no recalculados de forma
   independiente en esta sesión.

## 28. Siguiente acción mínima

Decisión explícita del responsable del proyecto: autorizar la consolidación
("recover 17-station artifacts" en `decision_options.csv`) — copiar los 6 CSV
`*_all_stations.csv` + `predictions_all_stations.csv` ya validados a
`outputs/tables/` y `outputs/metrics/` del repositorio principal, respaldar
`/home/fede/investigacion/repos/varret-pm10-paper` de forma duradera, instalar
`requirements.txt`, y regenerar Figura 5. Ninguna de estas acciones requiere
reentrenar modelos ni fabricar datos.
