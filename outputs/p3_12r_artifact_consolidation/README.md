# P3-12R — Consolidación de artefactos recuperados y liberación de submission

Continuación de la tarea P3-11R (que encontró y validó, pero no consolidó, la
evidencia de 17 estaciones de Paper A). Esta tarea consolida esos artefactos
de forma gobernada en el repositorio, regenera Figura 5 y las tablas
principales desde la evidencia validada, verifica la reproducibilidad del
manuscrito y emite una decisión final de submission.

## Resultado en una frase

`SUBMISSION_GO` — las 6 condiciones explícitas (clasificación A_COMPLETE,
0 discrepancias numéricas, regeneración de figura PASS, regeneración de
tablas PASS, compilación PASS, paquete de submission PASS) se cumplen. No se
envió el manuscrito ni se fusionó ningún PR/rama — eso queda para revisión
humana.

## Cómo leer estos outputs

| Archivo | Contenido |
|---|---|
| `final_report.md` | Informe completo con las 27 secciones de salida obligatoria |
| `source_manifest.csv` / `.sha256` | Manifiesto completo (624 archivos) del backup de solo lectura de la fuente recuperada |
| `recovered_artifact_validation.csv` | Validación estructural (filas, columnas, encoding, duplicados, cobertura) de los 7 artefactos principales + 14 tablas por estación |
| `storage_policy.csv` | Política de almacenamiento por artefacto (TRACK_IN_GIT vs PACKAGE_AS_RELEASE_ASSET) |
| `environment_freeze.txt` | `pip freeze` del entorno Python aislado usado para toda la regeneración |
| `recomputed_diagnostics.csv` | Recómputo de alpha/skill/n/collapse/near_ideal desde predicciones row-level, agrupado por estación×modelo×horizonte |
| `recomputation_comparison.csv` | Comparación celda a celda contra `variance_retention_all_stations.csv` con tolerancia 1e-10 |
| `named_exception_audit.csv` | Verificación específica de Huesca-HGB-h1 y Barcelona Vall d'Hebron-Ridge-h1 |
| `figure5_regeneration_audit.csv` | Auditoría de la regeneración de `figure5_scatter_skill_alpha.pdf` y `figure_skill_alpha_five_models.pdf` |
| `table_regeneration_audit.csv` | Comparación de tablas regeneradas vs. publicadas (IDENTICAL/ROUNDING_ONLY/FORMATTING_ONLY/NUMERIC_MISMATCH) |
| `data_availability_audit.csv` | Estado de disponibilidad por categoría de dato (raw/processed/row-level/aggregate/code) |
| `submission_package_audit.csv` | Checklist de consistencia del paquete `submission_package/ems/` |
| `pdf_visual_audit.md` | Inspección visual de los PDFs compilados y de Figura 5 antes/después |
| `evidence_status_final.json` | Clasificación final A/B/C con criterio explícito e invariance check |
| `submission_decision.json` | Decisión de submission con razonamiento, alternativas descartadas y límites de alcance |
| `before/`, `after/` | Copias de `figure5_scatter_skill_alpha.pdf` antes y después de la regeneración |

## Qué NO se hizo

- No se reentrenó ningún modelo.
- No se fabricaron ni modificaron datos ni predicciones.
- No se editaron métricas manualmente.
- No se cambió el corpus, los resultados ni los claims del manuscrito.
- No se borró la copia recuperada (`/home/fede/investigacion/repos/varret-pm10-paper`); solo se leyó y se respaldó.
- No se sobrescribió ningún archivo sin backup previo.
- No se fusionó ningún PR.
- No se envió el manuscrito a Elsevier / Environmental Modelling & Software.
