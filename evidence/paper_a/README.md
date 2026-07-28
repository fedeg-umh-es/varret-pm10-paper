# Paper A — evidence package (consolidated)

Este directorio consolida, de forma gobernada, la evidencia de 17 estaciones
para Paper A ("Ghost Skill & Dynamic Fidelity" / variance-retention study)
recuperada de una copia local no versionada durante la auditoría P3-11R y
consolidada en la tarea P3-12R.

## Procedencia (RECOVERED FROM LOCAL SOURCE COPY)

Todos los archivos de este directorio son **RECOVERED FROM LOCAL SOURCE COPY**,
no re-ejecuciones ni regeneraciones nuevas del pipeline. Se recuperaron de:

```
source location:  /home/fede/investigacion/repos/varret-pm10-paper
discovery date:   2026-07-28 (tarea P3-11R)
consolidation date: 2026-07-28 (tarea P3-12R)
```

Esa copia local es un directorio con un índice de git corrupto (ficheros
AppleDouble de macOS confundidos con packs de git), aparentemente una
sincronización del entorno de trabajo original del autor (Federico García
Crespí) en su máquina Mac, nunca incorporada al repositorio git canónico
(`fedeg-umh-es/varret-pm10-paper`) ni a GitHub. No se afirma que estos archivos
sean "la ejecución original recuperada" con más precisión de la que la
evidencia permite: se afirma únicamente que son bytes idénticos (verificados
por SHA-256) a los archivos encontrados en esa copia local, y que su contenido
fue **verificado por recómputo independiente** (ver
`outputs/p3_12r_artifact_consolidation/recomputation_comparison.csv`) contra
las cifras ya publicadas en el manuscrito.

Antes de esta consolidación se tomó un backup de solo lectura de la fuente
completa en `/home/fede/backups/p3-12r-recovered-evidence/source-copy/`
(rsync, sin modificar el origen). El manifiesto completo de esa copia está en
`outputs/p3_12r_artifact_consolidation/source_manifest.csv` /
`.sha256`.

## Estructura

```
evidence/paper_a/
├── README.md                    (este archivo)
├── manifests/
│   └── provenance_manifest.csv  (procedencia archivo por archivo)
├── aggregates/                  (tablas agregadas por celda estacion x modelo x horizonte)
├── source_tables/               (17 tablas por estacion individual)
├── metadata/                    (metadatos de estaciones y cobertura)
└── row_level/
    └── predictions_all_stations_manifest.md  (el CSV row-level de 895,737 filas
        NO se versiona aqui -- ver storage_policy.csv; se distribuye como
        release asset de GitHub, referenciado por hash y ubicacion estable)
```

## Cobertura verificada

- 17 estaciones (IDs MITECO/EMEP; ver `metadata/station_metadata.csv`)
- 7 horizontes (h=1..7)
- 5 familias de modelo evaluadas (`hgb_direct, ridge_direct, sarima,
  seasonal_naive, stl_ridge_direct`) + `persistence` como baseline
- 595 celdas agregadas (17×5×7); 119 celdas por modelo
- 895,737 filas row-level en `predictions_all_stations.csv` (ver release asset)

## Definición canónica (sin cambios)

```
alpha(h) = Var(y_hat_h) / Var(y_h), ddof = 0
near_ideal = skill > 0 AND 0.8 <= alpha <= 1.2
```

## Comando de validación

La validación estructural y el recómputo independiente de `alpha`/`skill`
desde el CSV row-level se ejecutaron con Python estándar (sin pandas, para no
depender de un entorno) y están documentados en:

```
outputs/p3_12r_artifact_consolidation/recovered_artifact_validation.csv
outputs/p3_12r_artifact_consolidation/recomputed_diagnostics.csv
outputs/p3_12r_artifact_consolidation/recomputation_comparison.csv
outputs/p3_12r_artifact_consolidation/named_exception_audit.csv
```

## Limitaciones conocidas

- La copia fuente (`/home/fede/investigacion/repos/varret-pm10-paper`) tiene un
  índice de git corrupto; no hay una cadena de procedencia git limpia
  (commits/branches inspeccionables) para estos archivos exactos, solo
  verificación a nivel de contenido (hash + recómputo).
- `dm_significance_all_stations.csv`, `exceedance_all_stations.csv` y
  `murphy_decomposition_all_stations.csv` fueron validados en esquema y
  cobertura, pero sus valores numéricos (test DM, recall/precision/F1,
  descomposición de Murphy) no fueron recomputados de forma independiente en
  esta tarea (fuera de alcance de tiempo); solo `alpha`/`skill` desde
  predicciones row-level.
- `predictions_all_stations.csv` no está versionado directamente en este
  directorio por tamaño (77 MiB); ver `row_level/predictions_all_stations_manifest.md`
  para su ubicación y hash.
