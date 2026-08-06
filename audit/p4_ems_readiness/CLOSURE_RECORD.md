# P4 Audit — Registro Documental de Cierre

## Veredicto canónico (corregido)

```
AUDIT_COMPLETE_WITH_MATERIAL_BLOCKERS
```

Este documento sustituye y corrige el veredicto `EMS_READY_FOR_TARGETED_REWRITE`
registrado en `REPORT.md` (commit ab5eb4d). El veredicto corregido fue establecido
en revisión editorial post-commit el 2026-08-06. `REPORT.md` no se modifica para
preservar la integridad del artefacto comprometido; este fichero es el registro
canónico de cierre.

---

## Metadatos de Cierre

| Campo | Valor |
|-------|-------|
| Fecha de cierre | 2026-08-06 |
| URL de PR | https://github.com/fedeg-umh-es/varret-pm10-paper/pull/18 |
| Estado de PR | Cerrada sin merge (draft, no fusionada) |
| Rama de auditoría | `claude/p4-lightgbm-ems-audit-rjs3ch` |
| Base | `main` (SHA `4a49b08b041c578ec5981dc1472125b2af0a4d59`) |

---

## Commits de Auditoría

| Hash | Descripción |
|------|-------------|
| `fdb73b2` | chore(p4): preregister LightGBM robustness arm |
| `a181f7d` | feat(p4): add LightGBM robustness arm — BLOCKED_BY_PIPELINE_OR_DATA |
| `7731570` | audit(p4): map EMS error-fidelity-selection gap |
| `ab5eb4d` | audit(p4): assess EMS readiness after LightGBM and corpus review |

Todos los commits están presentes en local y en remoto. La rama remota no se elimina.

---

## Estado de los Brazos

### WP-A — LightGBM Robustness Arm

**Veredicto:** `BLOCKED_BY_PIPELINE_OR_DATA`

El protocolo LightGBM fue pre-registrado correctamente en `config_snapshot.json`
antes de cualquier acceso a datos. No se ejecutó ninguna predicción. Los 14 ficheros
de datos brutos ausentes en `data/raw/` impiden la ejecución. El brazo de robustez
permanece **bloqueado mientras falten esos datos**.

La evidencia existente de 595 celdas no fue modificada (8/8 checks de integridad PASS;
Rule A=277, Rule B=8, decision_change=269; sin leakage detectado).

La configuración pre-registrada en `config_snapshot.json` queda preservada como
referencia documental. Cualquier eventual reactivación exigiría revalidarla frente
a los datos, el entorno y el código disponibles en ese momento.

### WP-B — EMS Corpus Gap Audit

**Veredicto:** `EMS_GAP_PARTIALLY_SUPPORTED`

La búsqueda fue web-trazada (23 consultas documentadas en `commands.log`), no una
revisión sistemática Scopus/WoS preregistrada. El veredicto no debe elevarse a
`SUPPORTED` sin evidencia de corpus adicional. Las discrepancias documentales,
afirmaciones imprecisas y referencias pendientes de verificación (incluyendo el DOI
estimado de P002) están identificadas explícitamente en `REPORT.md` de WP-B, sección
de limitaciones.

---

## Correcciones Editoriales Post-Commit

Las tres correcciones siguientes fueron establecidas en revisión editorial del
2026-08-06 y son canónicas. No se reflejan en los REPORT.md comprometidos para
preservar la integridad de los artefactos con hash verificado; este documento
es la fuente autoritativa.

1. **Sobre afirmaciones con respaldo:**
   "No se detectan datos fabricados. Las discrepancias documentales, afirmaciones
   imprecisas y referencias pendientes de verificación están identificadas
   explícitamente en el informe."
   *(Corrige: "No se detectan datos fabricados ni afirmaciones sin respaldo
   en los ficheros comprometidos", que contradecía las limitaciones documentadas.)*

2. **Sobre el estado de WP-A:**
   "El brazo de robustez permanece bloqueado mientras falten esos datos."
   *(Corrige: "permanece incompleto de forma indefinida".)*

3. **Sobre `config_snapshot.json`:**
   "La configuración pre-registrada en `config_snapshot.json` queda preservada
   como referencia documental; cualquier eventual reactivación exigiría revalidarla
   frente a los datos, el entorno y el código disponibles en ese momento."
   *(Corrige: "La configuración pre-registrada en `config_snapshot.json` permanece
   válida", que sobreafirmaba validez futura del entorno experimental.)*

---

## Snapshot Histórico de P4

El commit canónico histórico de P4 (`390685f1f1312954ee67513f3e0db11b2670e7f9`)
**no se encuentra en el repositorio remoto** `fedeg-umh-es/varret-pm10-paper`
(verificado vía GitHub API: 422 No commit found). Este commit existiría en el
repositorio local del propietario (`/Users/fede/...`) pero no ha sido publicado
en el remoto. La auditoría no puede confirmar ni refutar su integridad en el
entorno de ejecución remoto.

**Acción recomendada para el propietario:** verificar la presencia de
`390685f1f1312954ee67513f3e0db11b2670e7f9` en el repositorio local y, si
corresponde a un snapshot histórico que debe preservarse, considerar publicarlo
o etiquetarlo explícitamente en el remoto.

---

## Declaraciones de Cierre

1. **P4 permanece cerrado como snapshot histórico inmutable.** Esta auditoría es un
   acto de verificación documental, no una reapertura científica.

2. **La PR #18 se cierra sin merge.** La rama `claude/p4-lightgbm-ems-audit-rjs3ch`,
   los cuatro commits, los artefactos y los hashes se preservan como evidencia
   documental del proceso de auditoría.

3. **No existe autorización para reescribir el manuscrito.** Las secciones
   `claims_allowed.md` y `claims_prohibited.md` son documentación de gobernanza,
   no instrucciones de redacción activas.

4. **Ninguna reapertura implícita.** La auditoría identificó condiciones bajo las
   cuales una reactivación podría plantearse (provisión de datos LightGBM; revisión
   Scopus/WoS), pero no autoriza ninguna de esas acciones. Cualquier reactivación
   requiere decisión explícita del propietario fuera de este proceso.

5. **No hay tareas programadas ni monitorización posterior.** Este documento es el
   cierre definitivo de la sesión de auditoría.

---

## Elementos Preservados

| Elemento | Estado |
|----------|--------|
| Rama remota `claude/p4-lightgbm-ems-audit-rjs3ch` | PRESERVADA (no eliminada) |
| Commits fdb73b2, a181f7d, 7731570, ab5eb4d | PRESERVADOS en local y remoto |
| Artefactos WP-A (24 ficheros) | PRESERVADOS, hashes verificados |
| Artefactos WP-B (16 ficheros) | PRESERVADOS, hashes verificados |
| Artefactos integración (5 ficheros) | PRESERVADOS, hashes verificados |
| URL PR #18 | PRESERVADA (PR cerrada, no eliminada) |
| Hashes SHA-256 en artifact_hashes.sha256 | PRESERVADOS, todos OK |
| Snapshot histórico P4 en remoto | NO CONFIRMADO (ver sección anterior) |
