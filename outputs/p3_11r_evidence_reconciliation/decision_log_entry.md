# Decision Log Entry

**Decision ID:** `2026-07-28-paper-a-evidence-reconciliation`
**Task:** P3-11R — reconciliar la evidencia canónica de Paper A antes de submission
**Author:** Claude (sesión Claude Code), autorizada por fedeg@umh.es

> Nota: no existía `DECISION_LOG_TEMPLATE.md` en el repositorio en el momento de
> esta auditoría; esta entrada usa directamente la estructura solicitada por la
> tarea P3-11R. Es una entrada nueva; no se modificó ningún documento canónico
> previo.

## Question

¿Existe en algún repositorio, rama, commit, release, artifact, PR, worktree,
backup o almacenamiento local accesible un paquete completo y regenerable que
soporte los claims de 17 estaciones de Paper A?

## Available evidence

- El repositorio git principal (`/home/fede/repos/varret-pm10-paper`, todas las
  ramas locales y remotas, incluyendo `paper/paper-a-evidence-package` que se
  autodenomina "canonical evidence package") **no contiene** en ningún commit de
  su historial completo `variance_retention_all_stations.csv`,
  `master_diagnostic_table.csv`, `predictions_all_stations.csv`,
  `dm_significance_all_stations.csv` ni `exceedance_all_stations.csv`.
- GitHub (releases, PRs, workflow artifacts, code search global, 4 repos
  relacionados de la organización) tampoco contiene estos artefactos en
  ningún lugar indexable.
- Búsqueda de sistema de archivos completa (`/home/fede`, `/tmp`) encontró una
  copia local **no versionada por git de forma utilizable**
  (`/home/fede/investigacion/repos/varret-pm10-paper`, índice de git corrupto)
  que contiene **todos** los artefactos buscados, incluyendo
  `predictions_all_stations.csv` (895,737 filas row-level).
- Esa copia fue validada exhaustivamente: 595 filas / 17 estaciones / 5 modelos /
  7 horizontes en `variance_retention_all_stations.csv`; coincidencia EXACTA
  (hasta precisión de punto flotante) con todas las cifras canónicas citadas en
  la tarea (colapsos 118/118/110 de 119, medianas de alpha, near-ideal
  0/0/0/0/20, y las dos excepciones nombradas — Huesca HGB h=1, Barcelona Vall
  d'Hebron Ridge h=1).
- `alpha` fue **recomputado de forma independiente** desde
  `predictions_all_stations.csv` (sin entrenar nada) y coincide con el CSV
  agregado a 1.68e-13 de diferencia máxima en las 595 celdas — prueba directa de
  que la tabla agregada proviene de datos row-level reales.
- Una copia idéntica (mismo SHA-256) del CSV principal se encontró también en la
  papelera de reciclaje del usuario, corroborando su autenticidad y estabilidad.
- Detalle completo en `repository_ref_inventory.csv`, `github_search_inventory.csv`,
  `local_artifact_inventory.csv`, `backup_worktree_inventory.csv`,
  `candidate_validation.csv`, `recomputed_diagnostics.csv`, `figure5_audit.md`.

## Decision

**`SUBMISSION_HOLD_RECOVERABLE`**

La evidencia de 17 estaciones existe y fue verificada de forma independiente,
pero vive únicamente en una copia local no consolidada, con git corrupto, fuera
del repositorio versionado y de cualquier respaldo formal. No se envía el
manuscrito hasta consolidar esa evidencia en el repositorio canónico. Ver
`submission_decision.json` para el detalle completo del razonamiento y las
alternativas descartadas.

## Claims allowed

Ver lista completa en `submission_decision.json` → `claims_allowed`. Resumen: los
claims numéricos ya publicados en `paper_a.tex` (colapso 118/118/110 de 119,
medianas de alpha, near-ideal 0/0/0/0/20, definición
`alpha=Var(ŷ)/Var(y)`) están respaldados por evidencia verificada y pueden
mantenerse tal cual — no requieren cambio de contenido científico.

## Claims prohibited

Ver lista completa en `submission_decision.json` → `claims_prohibited`. Resumen:
no afirmar que el pipeline es "reproducible desde el repositorio público" (no lo
es todavía), no declarar Figura 5 como corregida (el PDF publicado sigue con el
rótulo antiguo), no dar por verificados numéricamente DM/exceedance/Murphy (solo
se verificó su esquema), no enviar el manuscrito dentro de esta tarea.

## Required next evidence

1. Decisión explícita del responsable del proyecto sobre consolidar
   `variance_retention_all_stations.csv` y demás tablas `*_all_stations.csv` en
   el repositorio git principal (opción "recover 17-station artifacts" en
   `decision_options.csv`).
2. Respaldo adicional fuera de `/home/fede/investigacion` de esos mismos
   archivos (ya iniciado en `outputs/p3_11r_evidence_reconciliation/recovered_candidates/`).
3. Instalación del entorno Python (`requirements.txt`) para regenerar Figura 5
   con el rótulo ya corregido en el script.
4. Recómputo independiente de `dm_significance_all_stations.csv` y
   `exceedance_all_stations.csv` (mismo método usado aquí para alpha).
5. Diagnóstico de la corrupción del índice git en
   `/home/fede/investigacion/repos/varret-pm10-paper` para restablecer una
   procedencia git limpia si es posible.

## Effect on manuscript

Ninguno en esta tarea. `paper_a.tex` no fue modificado. Los claims narrativos
actuales coinciden numéricamente con la evidencia recuperada, por lo que no se
anticipa ningún cambio de contenido científico — solo consolidación de la
evidencia que lo respalda y la corrección visual pendiente de Figura 5.
