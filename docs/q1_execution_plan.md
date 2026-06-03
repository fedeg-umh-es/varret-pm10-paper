# q1_execution_plan.md

# varret-pm10-paper: plan de cierre Q1/Q2

## Veredicto

Este paper debe cerrarse como paper diagnostico post-evaluacion de forecasting PM10. Su valor no esta en proponer un modelo nuevo, sino en demostrar que skill positivo frente a persistencia puede coexistir con colapso severo de varianza, reduciendo la fidelidad dinamica de las trayectorias pronosticadas.

Prioridad: cerrar despues de `paper2H` y antes de abrir KGE/DILATE/DPR.

Fuente canonica del manuscrito: Overleaf.

Repo de soporte: `/Users/federicogarciacrespi/Public/varret-pm10-paper`.

## Pregunta y claim

Pregunta:

> Puede un modelo mostrar Skill(h) positivo frente a persistencia y, al mismo tiempo, colapsar la varianza observada de la serie PM10?

Claim principal:

> Si. Por eso la evaluacion baseline-relative debe complementarse con un diagnostico de retencion de varianza alpha(h). Skill(h) responde si el modelo reduce error frente a persistencia; alpha(h) responde si la trayectoria predicha conserva variabilidad; Skill_VP(h) solo resume ambas lecturas como diagnostico auxiliar.

## Papel dentro de la linea

Este paper es la referencia de fidelity diagnostics / ghost skill.

No debe mezclarse con:

- paper2H cross-domain;
- H*(relax)/H*(strict) como contribucion principal;
- TAC Madrid-Irlanda;
- DPR/KGE/DILATE como contribucion central;
- probabilistic forecasting.

Puede compartir:

- rolling-origin;
- train-only preprocessing;
- persistence baseline;
- Skill(h);
- evaluacion por horizonte.

## Decisiones editoriales

### Target si se mantiene 3 estaciones

- `Air Quality, Atmosphere & Health`
- `Environmental Monitoring and Assessment`
- `Water, Air, & Soil Pollution`

Razon:

- caso PM10 focalizado;
- diagnostico operativo interpretable;
- menor exigencia de validacion extensa.

### Target si se amplia a all-stations

- `Atmospheric Environment`
- `Environmental Modelling & Software`

Razon:

- mayor generalidad empirica;
- mas fuerza contra objecion "too narrow";
- mejor encaje si se incluye exceedance relevance y Murphy decomposition.

Decision pendiente:

- cerrar rapido con 3 estaciones o reforzar con all-stations para un paper mas fuerte.

## Scope recomendado

### Opcion A: cierre rapido y limpio

Scope:

- 3 estaciones;
- 2 modelos;
- h=1..7;
- Skill(h), alpha(h), Skill_VP(h);
- collapse flags;
- exceedance relevance si ya esta maduro.

Ventaja:

- cierre rapido;
- narrativa muy clara;
- bajo riesgo tecnico.

Riesgo:

- reviewer puede decir "too narrow".

### Opcion B: cierre fuerte Q1

Scope:

- all-stations PM10;
- mantener 3 estaciones como casos explicativos;
- resumen agregado por tipo de estacion;
- collapse rates;
- exceedance recall;
- Murphy decomposition si apoya claramente.

Ventaja:

- reduce riesgo "too narrow";
- convierte el fenomeno en evidencia sistematica;
- mejor para Atmospheric Environment / EMS.

Riesgo:

- mas limpieza tecnica;
- mayor control de outputs;
- mas riesgo de dispersion narrativa.

Recomendacion:

- Si el objetivo es publicacion fuerte: Opcion B.
- Si el objetivo es cerrar rapido: Opcion A.

## Cierre por fases

### Fase 1: congelar scope

Objetivo:

Decidir si el paper final es 3-station diagnostic o all-stations diagnostic.

Acciones:

1. Revisar outputs all-stations existentes.
2. Confirmar si las figuras 3-7 ya reflejan all-stations o solo una seleccion.
3. Revisar si exceedance recall y Murphy decomposition estan maduros.
4. Escribir decision en `docs/q1_closeout_decisions.md`.

Criterio de cierre:

- una frase de scope final que pueda ir en abstract y methods.

### Fase 2: limpiar narrativa

Objetivo:

Evitar que el paper parezca SLR, benchmark o modelo nuevo.

Estructura recomendada:

1. Introduction:
   - skill positivo no implica fidelidad dinamica;
   - PM10 operativo necesita trayectorias que no colapsen variabilidad;
   - reporting gaps como motivacion breve.
2. Diagnostic framework:
   - Skill(h);
   - alpha(h);
   - collapse flags;
   - Skill_VP(h) como auxiliar.
3. Case study:
   - estaciones, modelos, rolling-origin.
4. Results:
   - skill positivo;
   - alpha colapsado;
   - quadrant/scatter;
   - exceedance relevance;
   - Murphy decomposition si se mantiene.
5. Discussion:
   - por que ocurre;
   - consecuencias operativas;
   - limites del diagnostico.

Criterio de cierre:

- cada seccion apoya el claim "positive skill can coexist with variance collapse".

### Fase 3: reducir SLR a proporcion adecuada

Objetivo:

Que el SLR no robe el paper.

Acciones:

1. Mantener reporting gaps como motivacion breve.
2. Mover detalles PRISMA extensos a suplemento o nota de datos si distraen.
3. No convertir el paper en "SLR + diagnostico"; debe ser "diagnostico con motivacion de reporting gap".

Criterio de cierre:

- SLR ocupa solo lo necesario para justificar por que hace falta el diagnostico.

### Fase 4: consolidar outputs canonicos

Objetivo:

Eliminar duplicacion visual y confusion.

Acciones:

1. Elegir una ubicacion canonica:
   - preferente: `outputs/figures/` para figuras;
   - `results/` para tablas/CSVs.
2. Evitar duplicados en raiz salvo si la journal package lo necesita.
3. Identificar cuales son figuras finales:
   - skill profiles;
   - alpha profiles;
   - skill-alpha scatter;
   - collapse rates;
   - station map;
   - exceedance recall;
   - Murphy decomposition.
4. Crear `docs/figure_table_traceability.md`.

Criterio de cierre:

- cada figura del manuscrito apunta a un archivo canonico y a un script productor.

### Fase 5: resolver deuda tecnica minima

Objetivo:

Que el repo no debilite la submission.

Acciones:

1. Resolver o documentar `scripts/06_build_skill_tables.py` si sigue siendo placeholder.
2. Confirmar que scripts activos no contienen rutas absolutas.
3. Mantener `scripts/p34_utils/` como legacy si no entra en pipeline final.
4. Separar companion KGE/P34 fuera del paper principal:
   - mantener en `docs/companion_kge/`;
   - no citarlo como resultado principal salvo appendix.
5. Revisar `.gitignore` y no versionar caches/tmp/PDFs exportados si no son deliverables.

Criterio de cierre:

- pipeline final sin rutas absolutas y sin placeholders activos.

### Fase 6: reforzar resultados

Objetivo:

Reducir riesgo "descriptive" o "incremental".

Resultados imprescindibles:

- Skill(h) positivo frente a persistencia;
- alpha(h) bajo / collapse flags;
- tabla resumen model x station;
- scatter skill-alpha;
- interpretacion operacional.

Resultados de alto impacto si estan maduros:

- exceedance recall P75/P90;
- Murphy decomposition;
- all-stations collapse rates;
- mapa por estacion.

No incluir si no estan maduros:

- KGE training;
- DPR;
- diferenciable physics;
- nuevos modelos deep learning.

Criterio de cierre:

- el paper demuestra un fenomeno robusto, no solo una anecdota.

### Fase 7: submission package

Acciones:

1. Sincronizar Overleaf y repo.
2. Generar PDF final.
3. Exportar fuentes.
4. Revisar captions.
5. Preparar cover letter:
   - "post-evaluation diagnostic";
   - "positive skill is incomplete";
   - "variance retention qualifies operational credibility".

## Cambios imprescindibles antes de submit

- Decidir 3 estaciones vs all-stations.
- Limpiar SLR para que no distraiga.
- Resolver trazabilidad figura/tabla.
- Eliminar rutas absolutas del pipeline final.
- Resolver placeholder `scripts/06_build_skill_tables.py` o excluirlo claramente.
- Separar KGE/P34 como companion/backlog.

## Mejoras de alto impacto

- All-stations si los outputs estan listos.
- Exceedance recall como implicacion operativa.
- Murphy decomposition como explicacion de por que RMSE mejora mientras alpha cae.
- Tabla final clara de cuadrantes skill-alpha.

## No hacer

- No abrir KGE/DILATE como experimento principal ahora.
- No meter DPR.
- No mezclar con paper2H.
- No presentar Skill_VP como metrica universal.
- No aumentar modelos por apariencia.
- No dejar que SLR/PRISMA domine el manuscrito.

## Proximo paso tecnico

Antes de editar el manuscrito:

1. Decidir scope A o B.
2. Crear `docs/figure_table_traceability.md`.
3. Auditar `scripts/06_build_skill_tables.py`.
4. Definir lista final de figuras y tablas.
5. Sincronizar Overleaf con el repo.
