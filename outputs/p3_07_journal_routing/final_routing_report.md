# P3-07 — Informe Definitivo de Selección de Revista Destino para Paper A

**PROJECT LINE:** P3 — Ghost Skill / Variance Retention  
**PAPER:** Paper A  
**REPOSITORY:** fedeg-umh-es/varret-pm10-paper  
**CANONICAL BRANCH:** main (`2939d26fb4e3447347ed9d6ba8e2d244462cb67b`)  
**CANONICAL MANUSCRIPT:** `paper_a.tex`, `paper_a.pdf`  
**CORPUS:** 17 estaciones PM10 diario MITECO (España), horizontes $h=1\ldots 7$, 5 familias de modelos.

---

## 1. Resumen Ejecutivo y Ranking Inequívoco

Tras auditar el perfil editorial del manuscrito, sus citas bibliográficas, la oferta editorial internacional, los requisitos institucionales (acuerdo UMH/CRUE-CSIC, indexación JCR/WoS y política estricta de NO APC obligatorio ni MDPI), se establece la siguiente clasificación:

### BEST TARGET NOW
**Environmental Modelling & Software** (Elsevier, Híbrida)
* **Justificación de Ajuste:** EM&S es la revista científica de referencia internacional para *metodologías de modelado ambiental, herramientas de verificación y marcos diagnósticos de evaluación*. La contribución de Paper A no es un nuevo modelo de red neuronal ni una teoría de transporte químico atmosférico, sino una **capa diagnóstica de verificación post-evaluación** ($\alpha(h)$, $Skill_{VP}$, descomposición de Murphy) para modelos de predicción ambiental.
* **Modelo de Publicación:** Híbrida (vía por suscripción disponible con 0 € APC obligatorio). Cubierta además al 100% por el acuerdo transformativo CRUE/UMH si se optara por Open Access.
* **Indexación:** Web of Science (SCIE) JCR Q1 (Factor de Impacto 4.8).
* **Riesgo de Desk-Reject:** Bajo. La audiencia de EM&S valora la metodología diagnóstica, la comparación rigurosa frente a baselines y los flujos reproducibles.

---

### CREDIBLE FALLBACK
**Atmospheric Environment** (Elsevier, Híbrida)
* **Justificación de Ajuste:** Revista líder en contaminación atmosférica y calidad del aire. Es la alternativa perfecta si se busca maximizar la visibilidad en la comunidad de calidad del aire.
* **Modelo de Publicación:** Híbrida (vía por suscripción 0 € APC, cubierta por el acuerdo UMH/CRUE).
* **Indexación:** Web of Science (SCIE) JCR Q1 (Factor de Impacto 4.2).
* **Ajuste de Encuadre Requerido:** Enfocar la carta de presentación ligeramente más hacia las implicaciones operacionales en redes de monitorización ambiental (MITECO) y alerta de episodios de contaminación.

---

### DO NOT SEND NOW
**International Journal of Forecasting** & **Revistas MDPI (Atmosphere, Sensors, etc.)**
* **International Journal of Forecasting:** Alto riesgo de desk-reject por exigir aportaciones teóricas/matemáticas universales en metodología de forecasting general que exceden el alcance de un estudio aplicado sobre PM10.
* **Revistas MDPI:** Excluidas por política estricta de proyecto (APC obligatorio, modelo Gold OA comercial).

---

## 2. Veredicto Final Inequívoco

```text
VERDICT: GO_TARGET_CONFIRMED
```

La revista principal **Environmental Modelling & Software** ha sido confirmada con grado alto de ajuste temático, metodológico y editorial, sin necesidad de modificaciones científicas al manuscrito `paper_a.tex`.
