# P3-11R — Reconciliación de evidencia canónica de Paper A

Auditoría de solo-lectura (más recuperación no destructiva de artefactos)
ejecutada el 2026-07-28 para resolver la contradicción entre el estado editorial
(PR #7 merged, "17-station daily PM10 study", target Environmental Modelling &
Software) y el estado local reportado por la tarea previa (P3-11:
`variance_retention_all_stations.csv: NOT_FOUND`).

## Resultado en una frase

La evidencia de 17 estaciones **existe y fue verificada de forma independiente**
(recómputo de `alpha` desde 895,737 filas de predicciones row-level, coincidencia
exacta con las cifras canónicas), pero vive únicamente en una copia local no
consolidada (`/home/fede/investigacion/repos/varret-pm10-paper`, con índice git
corrupto), nunca incorporada al repositorio git versionado. Decisión:
**`SUBMISSION_HOLD_RECOVERABLE`**.

## Cómo leer estos outputs

| Archivo | Contenido |
|---|---|
| `final_report.md` | Informe completo con las 28 secciones de salida obligatoria de la tarea |
| `repository_ref_inventory.csv` | Verificación de cada SHA/rama solicitados explícitamente |
| `github_search_inventory.csv` | Auditoría de GitHub (branches, tags, releases, PRs, workflow artifacts, code search) en el repo principal y 4 repos relacionados |
| `local_artifact_inventory.csv` | Todos los artefactos relevantes encontrados en la búsqueda de sistema de archivos completa |
| `backup_worktree_inventory.csv` | Backups, worktrees y copias locales inspeccionadas |
| `candidate_validation.csv` | Validación estructural/numérica de cada candidato encontrado |
| `recomputed_diagnostics.csv` | Recómputo independiente de alpha/skill/collapse desde datos row-level (sin entrenamiento) |
| `figure5_audit.md` | Auditoría específica de Figura 5: estado actual, causa raíz, clasificación |
| `evidence_classification.json` | Clasificación A/B/C con criterio explícito |
| `decision_options.csv` | Matriz de opciones para el paso siguiente |
| `submission_decision.json` | Decisión de submission con razonamiento y claims permitidos/prohibidos |
| `decision_log_entry.md` | Entrada de decision log (`2026-07-28-paper-a-evidence-reconciliation`) |
| `local_state_before/` | Snapshot no destructivo del estado local previo a esta auditoría |
| `recovered_candidates/` | Copias de solo lectura (hash verificado) de los artefactos recuperados, para respaldo |

## Qué NO se hizo

- No se reentrenó ningún modelo.
- No se regeneraron predicciones mediante entrenamiento.
- No se fabricó ni reconstruyó manualmente ningún dato.
- No se modificó ni se borró nada en `/home/fede/investigacion/repos/varret-pm10-paper`
  (solo lectura + copia).
- No se envió el manuscrito a ninguna revista.
- No se fusionó ningún PR.
