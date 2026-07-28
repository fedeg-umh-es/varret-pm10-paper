# Auditoría visual de PDFs — P3-12R

## Manuscrito canónico (`paper_a.tex` → `paper_a.pdf`)

- Compilación: `latexmk -pdf -interaction=nonstopmode -halt-on-error` → exit 0.
- `grep -i "undefined"` en `paper_a.log`: 0 coincidencias.
- SHA-256 del PDF: `6e462fb98a99a935973902b96c5f4d857e3d1df8cdb8a555ae5812d9f610cec8`
- 17 páginas.
- Página con Figura 5 (interna "Figure 3" en la numeración del documento,
  correspondiente al archivo `figure5_scatter_skill_alpha.pdf`) renderizada e
  inspeccionada visualmente (ver `pdf_visual/paper_a_p10-10.png`):
  - Rótulo del eje X: **`Variance Retention Coefficient (α = Var(ŷ)/Var(y))`** — confirmado Var/Var, no SD/SD.
  - Puntos de dispersión de fondo (595 celdas) presentes y sin recorte.
  - Trayectorias h=1..7 de los 5 modelos, leyenda, cuadrantes I-IV y banda
    "Near-Ideal Band [0.8, 1.2]" presentes y legibles.
  - Sin símbolos corruptos, sin solapamiento de texto, sin clipping en los
    bordes del área de trazado.
  - Caption y numeración de figura consistentes con el texto circundante.

## Manuscrito EM&S (`submission_package/ems/paper_a_ems.tex` → `paper_a_ems.pdf`)

- Bloqueos previos resueltos (ver Fase 13 / final_report.md):
  1. Faltaba `elsarticle.cls` — compilar desde `submission_package/ems/` (donde
     ya viven las clases `.cls`/`.bst`), no desde la raíz del repo.
  2. Faltaba `model_family_diagnostic_summary.tex` en `submission_package/ems/`
     — copiado desde la raíz (contenido idéntico al ya usado por `paper_a.tex`).
  3. Referencia rota `\input{supplementary_material.tex}` (el archivo real se
     llama `supplementary_material_ems.tex`, contenido idéntico al
     `supplementary_material.tex` de la raíz, solo renombrado) — corregida la
     referencia a `\input{supplementary_material_ems.tex}`. Cambio mecánico de
     referencia, sin alterar contenido.
- Compilación tras las correcciones: exit 0.
- `grep -i "undefined"` en `paper_a_ems.log`: 0 coincidencias.
- SHA-256 del PDF: `c84d269a84b4e0990633a1e62a4dd32f71c5b8c88317b92df9a79310e4e54bba`
- 17 páginas.
- Portada, título, autores, afiliaciones, abstract y keywords renderizados
  correctamente (ver `pdf_visual/ems_p1-01.png`).
- Rótulo de Figura 5 confirmado vía extracción de texto:
  `Variance Retention Coefficient (α = Var(ŷ)/Var(y))` — Var/Var correcto.
- `supplementary_material_ems.tex` no es un documento standalone (no tiene
  `\documentclass`; es un fragmento `\input`-eado dentro de `paper_a_ems.tex`);
  su compilación exitosa queda validada como parte de la compilación exitosa
  de `paper_a_ems.tex` completo (incluye `\input{prisma_reporting_audit_summary.tex}`,
  también verificado sin errores).

## Figuras regeneradas — comparación visual lado a lado

- `figure5_scatter_skill_alpha.pdf`: renders PNG a 150dpi de la versión
  publicada anterior (`before/`) y la regenerada (`after/`) comparados
  visualmente: mismos puntos, mismas trayectorias, mismos cuadrantes, misma
  leyenda, misma banda near-ideal. Única diferencia observable: el texto del
  rótulo del eje X. Ver `figure5_regeneration_audit.csv` para el detalle
  cuantitativo del diff de píxeles y su explicación (desplazamiento sub-pixel
  del bounding box "tight" por el cambio de longitud del texto del rótulo).

## Limitaciones de esta auditoría visual

- No se inspeccionaron visualmente las 17 páginas completas de cada PDF,
  solo la portada y la página que contiene la Figura 5 (las páginas de mayor
  riesgo de regresión visual dado el alcance de esta tarea).
- No se ejecutó un diff de imagen automatizado pixel-a-pixel de las 17 páginas
  completas; la verificación de "sin cambios de contenido" se apoya
  principalmente en que `paper_a.tex` no fue modificado (solo se sustituyó el
  archivo binario de Figura 5) y en que `model_family_diagnostic_summary.tex`
  (la tabla principal) resultó IDENTICAL/FORMATTING_ONLY en el recómputo (ver
  `table_regeneration_audit.csv`).
