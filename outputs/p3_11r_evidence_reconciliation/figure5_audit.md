# Auditoría de Figura 5 — `figure5_scatter_skill_alpha.pdf`

## Estado actual publicado (main, commit 2939d26)

| Campo | Valor |
|---|---|
| PDF actual | `outputs/figures/figure5_scatter_skill_alpha.pdf` (y copia en raiz `figure5_scatter_skill_alpha.pdf`) |
| sha256 | `781515d4e3756941bfc4fdfa7f7298cd0807db4a77fc9bcf7cb68140d19ed2d1` |
| Script generador | `scripts/generate_figures_3_to_6.py` (funcion `figure5_skill_alpha`) |
| CSV fuente declarado por el script | `outputs/tables/variance_retention_all_stations.csv` |
| Branch | `main` |
| Commit que introdujo el PDF actual | `2939d26fb4e3447347ed9d6ba8e2d244462cb67b` (merge PR #7; el diff de ese merge muestra `figure5_scatter_skill_alpha.pdf \| Bin 56631 -> 22377 bytes`, es decir, la version "premium" fue re-renderizada ahi mismo) |
| Rotulo del eje X (segun historial del script, version "premium") | `$\alpha = s_{\hat{y}} / s_{y}$` — notacion SD/SD, **incorrecta** frente a la definicion canonica |

`pdftotext` no puede extraer la formula matematica del eje (esta renderizada como trazos vectoriales de matplotlib mathtext, no como texto), pero el codigo fuente que genero ese PDF exacto (previo a la correccion aplicada en esta sesion) contenia literalmente `s_{\hat{y}} / s_{y}$`, confirmado por `git show` del script en el mismo commit.

## Corrección aplicada en esta sesión (y en la sesión P3-11 previa)

En la tarea P3-11 previa se corrigio el texto del rotulo en el arbol de trabajo local
(sin commitear entonces) de:

```
$\alpha = s_{\hat{y}} / s_{y}$   (SD/SD)
```

a:

```
$\alpha = \mathrm{Var}(\hat{y}) / \mathrm{Var}(y)$   (Var/Var)
```

en `scripts/generate_figures_3_to_6.py` y `scripts/14_generate_skill_alpha_figure.py`.
Ese cambio sigue presente en el arbol de trabajo (no commiteado hasta esta tarea P3-11R).

## Hallazgo decisivo de esta tarea (P3-11R, Fase 3)

A diferencia de la sesion P3-11 previa (que concluyo `SOURCE_MISSING` /
`BLOCKED` porque `variance_retention_all_stations.csv` no aparecia en
`/home/fede/repos`), esta tarea amplio la busqueda a todo el sistema de archivos
(`/home/fede`, `/tmp`, etc.) y **encontro el CSV fuente completo y validado** en:

```
/home/fede/investigacion/repos/varret-pm10-paper/outputs/tables/variance_retention_all_stations.csv
sha256: 5cb0c3a4240962b2fbe3a9ba11ebaf2e633fe7bca95ade32a506c0a34a28318d
595 filas, 17 estaciones, 5 modelos, 7 horizontes
```

Ese CSV fue validado exhaustivamente contra las cifras canonicas (coincidencia
exacta, ver `candidate_validation.csv`) y ademas **recomputado de forma
independiente** desde `predictions_all_stations.csv` (895,737 filas row-level,
tambien encontrado en la misma ubicacion), con `alpha` coincidiendo a precision
de punto flotante (diff maximo 1.68e-13) en las 595 celdas — ver
`recomputed_diagnostics.csv`.

## Clasificación

**REGENERABLE** (actualización respecto a la clasificación previa `SOURCE_MISSING`).

Justificación:
- CSV fuente: **encontrado y validado** (antes: ausente).
- Script generador: **ya corregido** en el árbol de trabajo local (rótulo Var/Var).
- Bloqueo restante: puramente de **entorno** — el Python del sistema en
  `/home/fede/repos/varret-pm10-paper` no tiene instalados `pandas`/`numpy`/`matplotlib`
  (`requirements.txt`), por lo que el script no se pudo ejecutar en esta sesión sin
  instalar dependencias (acción no autorizada explícitamente en el alcance de esta
  tarea — instalar paquetes no es "entrenar modelos" ni "regenerar predicciones",
  pero tampoco se pidió explícitamente permiso para modificar el entorno Python, así
  que se deja como próxima acción mínima).

El PDF **actualmente publicado** en `main` (hash `781515d4...`) sigue teniendo el
**rótulo antiguo (STALE_LABEL)**: no ha sido re-renderizado todavía. Ambos estados
coexisten y deben reportarse:

- Artefacto publicado hoy: `STALE_LABEL` (datos correctos, rótulo incorrecto).
- Camino a la corrección: `REGENERABLE` (todo lo necesario existe; falta solo
  ejecutar el pipeline de figuras con el CSV recuperado y el script ya corregido).

No se sustituyó la figura por ninguna versión aproximada. No se regeneró el PDF en
esta sesión (se mantiene la restricción de "no reconstruir manualmente resultados").

## Siguiente acción mínima para cerrar Figura 5

1. Copiar (extracción, ya pre-autorizada) `variance_retention_all_stations.csv`
   validado a `outputs/tables/variance_retention_all_stations.csv` en
   `/home/fede/repos/varret-pm10-paper` — **pendiente de decisión explícita** (ver
   `decision_options.csv`), porque implica introducir datos recuperados de una
   fuente externa al repo git en el árbol de trabajo canónico.
2. Instalar `pandas`, `numpy`, `matplotlib` (ligero, no es reentrenamiento) —
   pendiente de autorización de cambio de entorno.
3. Ejecutar `python3 scripts/generate_figures_3_to_6.py` (o la función específica
   de Figura 5) y verificar que `data_changed=false` frente a los números ya
   publicados, y que el único cambio visual es el rótulo del eje.
4. Sustituir `outputs/figures/figure5_scatter_skill_alpha.pdf` y la copia en raíz,
   documentando el hash antes/después.
