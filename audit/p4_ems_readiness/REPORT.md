# P4 Audit — EMS Readiness Assessment
## Integration Report (WP-A + WP-B)

---

## 1. Veredicto Final

```
EMS_READY_FOR_TARGETED_REWRITE
```

**Razón:** Los dos brazos del P4 audit concluyen de la siguiente manera:
- **WP-A (LightGBM):** `BLOCKED_BY_PIPELINE_OR_DATA` — el brazo de robustez no pudo ejecutarse
  por ausencia de datos brutos en 14/17 estaciones. Este veredicto NO invalida ni modifica
  la evidencia existente (595 celdas verificadas íntegras). No es una refutación del fenómeno.
- **WP-B (EMS corpus):** `EMS_GAP_PARTIALLY_SUPPORTED` — la combinación específica de criterio
  de error + retención de varianza (alpha) + recall de extremos como regla de selección conjunta
  está ausente del corpus EMS 2015-2026 según la búsqueda realizada. El debate Williams 2025 en
  EMS provee contexto editorial favorable pero no cierra la brecha específica de Paper A.

La evidencia existente (595 celdas, Rule A=277, Rule B=8, decision_change=269/277=97.1%,
sin leakage) es suficiente para justificar una revisión del manuscrito orientada a EMS
con el framing correcto. El reencuadre debe: (i) abandonar los 7 gaps heterogéneos previos;
(ii) centrarse en el hallazgo empírico de la regla de selección conjunta; (iii) citar el
debate EMS como contexto; (iv) no reclamar robustez LightGBM hasta que los datos estén disponibles.

---

## 2. Síntesis de Brazos

### 2.1 WP-A: LightGBM Robustness Arm

| Dimensión | Resultado |
|-----------|-----------|
| Veredicto | BLOCKED_BY_PIPELINE_OR_DATA |
| Razón | 14/17 estaciones sin datos brutos; MITECO bloqueado |
| Impacto en claim principal | NINGUNO — no modifica las 595 celdas existentes |
| Integridad 595 celdas | VERIFICADA (8/8 checks PASS) |
| Sin leakage | CONFIRMADO (7 checks CONFIRMADOS) |
| Config LightGBM pre-registrada | SÍ — `config_snapshot.json` |
| Acción requerida | Propietario debe proveer datos brutos de 14 estaciones |

### 2.2 WP-B: EMS Corpus Gap Audit

| Dimensión | Resultado |
|-----------|-----------|
| Veredicto | EMS_GAP_PARTIALLY_SUPPORTED |
| Papers EMS encontrados | 2 (Williams 2025, Comment on Williams 2026) |
| Papers adyacentes incluidos | 4 (MFM HESS, NatComms 2021, arXiv PM10, GCMeval) |
| Dimensiones sin cobertura EMS | B, C, D, E, F, G, H (7 de 8) |
| Combinación completa B∧C∧D ausente | SÍ (EMS y corpus adyacente) |
| Corpus cubierto | Búsqueda web 23 consultas; no revisión Scopus/WoS sistemática |

---

## 3. Matriz de Decisión

| Componente | Veredicto | ¿Necesario para EMS_READY? | ¿Disponible? |
|------------|-----------|---------------------------|-------------|
| 595 celdas verificadas sin leakage | VERIFIED | SÍ (imprescindible) | SÍ ✓ |
| Gap EMS ≥ PARTIALLY_SUPPORTED | EMS_GAP_PARTIALLY_SUPPORTED | SÍ | SÍ ✓ |
| LightGBM robustez | BLOCKED | NO (deseable, no obligatorio) | NO |
| Revisión sistemática Scopus/WoS | no realizada | NO (deseable) | NO |

**Conclusión lógica:** Los dos requisitos imprescindibles están satisfechos.
Los dos deseables están bloqueados/no realizados. El veredicto EMS_READY aplica.

---

## 4. Especificación del Reencuadre

El reencuadre de Paper A para EMS debe:

### 4.1 Frame principal

**Claim central:** La evaluación de pronóstico de PM10 mediante criterio de error solo
(habilidad de predicción + test DM) resulta en selección errónea de modelos en el 97.1%
de los casos donde los modelos pasarían dicho criterio, cuando se aplica adicionalmente
la retención de varianza (alpha = Var(ŷ)/Var(y)) y el recall de episodios extremos.

### 4.2 Posicionamiento respecto al debate EMS

- Citar Williams 2025 (EMS) como el debate más cercano: EMS está publicando discusiones sobre
  qué métricas usar para evaluación de modelos ambientales.
- Posicionar Paper A como: primer estudio empírico multi-estación multi-horizonte en calidad
  del aire que OPERACIONALIZA una regla conjunta (skill + alpha + recall) y CUANTIFICA su impacto
  en selección de modelos.
- Distinguir alpha de NSE/KGE: alpha no es una métrica de habilidad normalizada sino una ratio
  de varianzas que mide fidelidad dinámica; esto responde al debate Williams de forma constructiva.

### 4.3 Lo que NO debe incluir el reencuadre

- Los 7 gaps heterogéneos previos (ver superprompt: prohibición explícita).
- Claims sobre LightGBM (WP-A bloqueado).
- Claims sobre generalización más allá de PM10 España (sin evidencia).
- Redacción de secciones del manuscrito (acción prohibida en este audit).

---

## 5. Verificaciones de Integración

| Check | Estado |
|-------|--------|
| WP-A veredicto emitido | PASS ✓ |
| WP-B veredicto emitido | PASS ✓ |
| Veredicto final en lista permitida | PASS ✓ (EMS_READY_FOR_TARGETED_REWRITE) |
| Decision matrix completa | PASS ✓ |
| Claims allowed/prohibited documentados | PASS ✓ |
| Evidencia 595 celdas no modificada | PASS ✓ |
| Sin leakage detectado | PASS ✓ |
| LightGBM config pre-registrada antes de ejecución | PASS ✓ |

---

## 6. Próximos Pasos Requeridos

1. **Datos LightGBM [PROPIETARIO]:** Proveer archivos CSV de PM10 diario para las 14
   estaciones faltantes en `data/raw/` (IDs: 08019004, 08019028, 08019043, 08019045,
   08019052, 08019054, 08263007, 22125001, 43004005, 43004006, 44013007, 44216001,
   45153999, 50008001) o habilitar acceso de red a MITECO desde el entorno de ejecución.

2. **Reencuadre del manuscrito [PROPIETARIO + REVISIÓN]:** Reescribir la sección de
   introducción y contribución principal usando el frame del punto 4.2; no realizar esto
   hasta aprobación del propietario. No toca datos ni resultados.

3. **Revisión sistemática opcional [PROPIETARIO]:** Si EMS requiere declarar ausencia de
   antecedentes, considerar búsqueda Scopus/WoS con criterios PRISMA para robustecer C-B1.

---

## 7. Limitaciones

- WP-A: datos no disponibles; experimento no ejecutado.
- WP-B: corpus web-search solamente; paywall EMS limita acceso full-text.
- Integración: el veredicto EMS_READY_FOR_TARGETED_REWRITE no garantiza aceptación;
  solo indica que la evidencia disponible justifica la revisión del manuscrito.
