# Paper A — EMS Submission Technical Closeout

- **Fecha de cierre técnico:** 28 de julio de 2026
- **Repositorio canónico:** `fedeg-umh-es/varret-pm10-paper`
- **Estado:** `SUBMISSION_READY`
- **Destino editorial preparado:** Environmental Modelling & Software
- **Envío realizado:** No

## 1. Veredicto

```text
SUBMISSION_READY
```

El repositorio, el manuscrito y el paquete de entrega quedaron técnicamente preparados. No se realizó ningún envío automático ni manual al portal de EMS.

## 2. Clasificación científica

Este cierre pertenece al Paper A sobre variance retention, variance collapse, ghost skill y `Skill_VP` en forecasting multi-step de PM10.

No pertenece a `paper2H`, `hstar`, `benchmark-pm-hstar` ni a otros proyectos de forecasting.

## 3. Hallazgo SARIMA

```text
cause: combine_prediction_tables.py descarta las filas de persistencia propias de SARIMA
       (2018-2019) al exportar el CSV row-level
cells affected: 119/595
pairing valid: SÍ en el cómputo canónico (skill_sarima_{station}.csv, inner join exacto)
canonical source: scripts/02_generate_sarima_predictions.py
claim impact: ninguno
repair performed: SÍ — commit 4e3a97a "docs: clarify SARIMA skill provenance and row-level
       limitations" en la rama de PR #15, verificado con cero cambios numéricos y fusionado
```

La limitación afecta a la evidencia SARIMA celda a celda disponible en el release row-level, pero no invalida el cálculo canónico ni los claims agregados documentados.

## 4. Pull request #14

```text
verdict: APPROVE
merge performed: SÍ, mediante squash
merge SHA: c2e088a0fa9303c6436a81df867099f81d34106d
remaining issue: ninguno
```

## 5. Pull request #15

```text
verdict: APPROVE, tras el commit de documentación SARIMA
retarget performed: SÍ, de audit/p3-11r-evidence-reconciliation a main
merge performed: SÍ, mediante squash
merge SHA: 45cfed9aa6ce20c50b7a584068556cb27c7afd60
remaining issue: ninguno
```

La ausencia de duplicación se verificó mediante comparación de hashes de blob y no únicamente mediante la vista de GitHub.

## 6. Pull request #16

El Data Availability Statement mantenía después de la publicación del release el texto previo a la publicación, indicando que los datos estaban disponibles bajo solicitud.

Se corrigió mediante el PR #16.

```text
merge SHA: 7a8c6e5632fec2ce6922c53229e8470777527732
```

Este commit constituye el `main` final usado para la compilación y la congelación del paquete.

## 7. Release publicado

```text
verdict: PUBLISHED
published: true
published_at: 2026-07-28T18:23:33Z
tag: paper-a-row-level-evidence-v1
asset: predictions_all_stations.csv
size: 77,080,400 bytes
rows: 895,737
sha256: 2551b3a2acf549e94cd92e39386b96aca2358cca41fed7f85c933d35c77ef823
```

## 8. Tests

```text
passed: 15
failed: 0
deleted tests: 0
modified tests: 0
```

Dos archivos de test no se recopilaron por ausencia de `lightgbm`. Esta situación era preexistente desde el PR #5 y no fue introducida ni ocultada durante el cierre.

No se relajó ni eliminó ningún test.

## 9. Compilación final

```text
main SHA: 7a8c6e5632fec2ce6922c53229e8470777527732
pages: 17
fatal errors: 0
undefined references: 0
undefined citations: 0
```

La compilación final se realizó desde un checkout limpio de `main`.

La auditoría visual confirmó:

- Figura 5 con el eje `α = Var(ŷ)/Var(y)` correctamente rotulado.
- Tabla 2 con:
  - skill SARIMA: `0.208`;
  - α SARIMA: `0.095`;
  - colapso: `92.4%`.
- El suplemento se integra correctamente en el documento maestro.

## 10. Paquete final congelado

Directorio:

```text
dist/paper_a_ems_submission_final/
```

Archivo ZIP:

```text
dist/paper_a_ems_submission_final.zip
```

Estado:

```text
files: 30
unzip test: sin errores
complete: SÍ
sha256: d9547679e009b16a441623e341e149edde08e685895b3a283acabc5dbfa693a4
```

El paquete debe conservarse como artefacto congelado. No debe regenerarse ni sobrescribirse sin una nueva revisión técnica completa y una justificación explícita.

### Paquete anterior sustituido antes del envío (2026-07-29)

```text
Previous package superseded before submission:
- reason: removal of four image files not referenced by any submitted TeX source
- submission impact: none
- manuscript content changed: no
```

El paquete anterior (34 archivos, `sha256: 512502e5c3ad9d410e91224609757e931b89de015fba14af98c8765bf21432dc`) contenía cuatro PDF de figura no citados por ningún `.tex` del paquete (`figure2_skill_variance_retention.pdf`, `figure6_station_collapse_rates.pdf`, `figure7_station_map_spain.pdf` — 8.36 MB, el mayor archivo del paquete — y `figure_skill_alpha_five_models.pdf`), confirmado mecánicamente mediante `grep` de `\includegraphics` contra los cuatro archivos `.tex` del paquete. Se retiraron por no cumplir ninguna de las condiciones que justifican su inclusión (citada en el manuscrito o suplemento; exigida por la revista como archivo independiente; parte de material suplementario declarado). También se corrigió `file_manifest.csv` (eliminando esas cuatro entradas y la línea de `graphical_abstract_source.csv`, que nunca estuvo realmente en la carpeta) y se actualizó `README.md` del paquete con una nota de este cierre.

El hash anterior (`512502e5...`) queda invalidado y no debe presentarse como el artefacto final; el hash vigente es el indicado arriba (`d9547679...`). Ningún archivo de manuscrito, figura citada, tabla, cifra o dato cambió — únicamente se depuraron cuatro archivos auxiliares no referenciados.

## 11. Estado de los claims

### Completamente soportados

- Colapso de varianza:
  - `118/119`;
  - `118/119`;
  - `110/119`, según los criterios correspondientes.
- Medianas de `α`.
- Casos near-ideal.
- Mediana de skill SARIMA, aproximadamente `0.208`.

### Parcialmente soportados desde el release row-level

- Skill SARIMA celda a celda.

La limitación está documentada explícitamente y no debe ocultarse.

### Sin soporte

Ninguno.

### Contradichos

Ninguno.

## 12. Estado operativo final

No quedan acciones técnicas automáticas pendientes.

La única acción restante es humana:

1. revisar el contenido de:

   ```text
   dist/paper_a_ems_submission_final.zip
   ```

2. comprobar visualmente los archivos que se subirán;
3. realizar el envío manual en el portal editorial cuando se decida.

Este documento no acredita que el manuscrito haya sido enviado.
