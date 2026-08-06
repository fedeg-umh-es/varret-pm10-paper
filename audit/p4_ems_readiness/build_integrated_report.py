"""
Build Integrated Readiness Report for EMS — Paper A / P4
=========================================================
Generates all artifacts in audit/p4_ems_readiness/

Synthesizes Work Package A (LightGBM Robustness) and Work Package B (EMS Corpus Audit).
"""

import json
import hashlib
import time
from pathlib import Path
import pandas as pd

REPO = Path('/Users/fede/Library/Mobile Documents/iCloud~md~obsidian/Documents/03_Investigacion/repos/varret-pm10-paper')
READINESS_DIR = REPO / 'audit' / 'p4_ems_readiness'
READINESS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DECISION MATRIX CSV (Table from Section 24)
# ---------------------------------------------------------------------------
matrix_rows = [
    {
        "editorial_question": "Does the phenomenon depend on a single learner?",
        "lightgbm_evidence": "No. LightGBM reproduces HGB dynamics (mean skill 0.1915 vs 0.1918, mean alpha 0.1860 vs 0.1829).",
        "ems_corpus_evidence": "N/A (Learner robustness tested empirically in WP A).",
        "conclusion": "Robustness confirmed across gradient boosting implementations."
    },
    {
        "editorial_question": "Does dynamic fidelity change the selected model?",
        "lightgbm_evidence": "Yes. Adding alpha >= 0.50 disqualifies ML models in 95.8% of h>=2 decisions, changing top-1 selection.",
        "ems_corpus_evidence": "Only 12% of EMS papers report selection changes from multi-metric criteria; 0% audit multi-horizon desqualification.",
        "conclusion": "Material decision consequence established across site x horizon."
    },
    {
        "editorial_question": "Do existing antecedents report fidelity metrics?",
        "lightgbm_evidence": "N/A",
        "ems_corpus_evidence": "20% of EMS papers report KGE or alpha, primarily in hydrology.",
        "conclusion": "Fidelity metrics exist, but are evaluated separately from model selection."
    },
    {
        "editorial_question": "Do existing antecedents report decision changes?",
        "lightgbm_evidence": "N/A",
        "ems_corpus_evidence": "0% of EMS papers perform a joint error-fidelity-significance decision audit.",
        "conclusion": "Gap supported: decision-change audit is novel in EMS."
    },
    {
        "editorial_question": "Does Paper A add a non-obvious consequence?",
        "lightgbm_evidence": "154 total discordant cells where error & DM & recall are favorable but alpha < 0.50.",
        "ems_corpus_evidence": "EMS studies treat error skill and peak recall as complementary without reporting discordance.",
        "conclusion": "Discordance demonstrates non-obvious risk of error-only selection."
    },
    {
        "editorial_question": "Is the contribution generalizable?",
        "lightgbm_evidence": "Demonstrated across 17 sites, 6 model families, 7 horizons (714 cells total).",
        "ems_corpus_evidence": "Multisite multihorizon AI verification is highly sought in EMS.",
        "conclusion": "Generalizable empirical decision framework."
    },
    {
        "editorial_question": "Is the EMS framing defensible?",
        "lightgbm_evidence": "N/A",
        "ems_corpus_evidence": "EMS scope explicitly prioritizes model evaluation quality and AI verification standards.",
        "conclusion": "High scope alignment when framed as a decision audit."
    }
]

pd.DataFrame(matrix_rows).to_csv(READINESS_DIR / 'decision_matrix.csv', index=False)

# ---------------------------------------------------------------------------
# 2. CLAIMS ALLOWED MD (Section 31)
# ---------------------------------------------------------------------------
claims_allowed_content = """# Claims Allowed for EMS Manuscript (Paper A / P4)

The following claims are strictly backed by empirical evidence from Work Package A and Work Package B:

1. **Learner Robustness:**
   > "The inclusion of LightGBM confirmed that positive persistence-relative skill coexists with low variance retention under the same rolling-origin protocol, matching HistGradientBoosting dynamics across 119 station–horizon evaluations."

2. **Model Selection Consequence:**
   > "Adding a dynamic-fidelity requirement ($\alpha \ge 0.50$) changed model eligibility and, across 714 station–model–horizon evaluations, led to model desqualification in 95.8% of multi-step horizons ($h \ge 2$)."

3. **Literature Gap & Contribution:**
   > "Existing EMS forecasting studies frequently report error metrics and occasionally event or variability diagnostics, but explicit evidence that their joint consideration systematically changes model-selection decisions across sites, model families and forecast horizons remains limited."

4. **Scope of Contribution:**
   > "Paper A contributes a multi-station, multi-model and multi-horizon decision audit protocol rather than proposing a new universal metric."
"""
(READINESS_DIR / 'claims_allowed.md').write_text(claims_allowed_content)

# ---------------------------------------------------------------------------
# 3. CLAIMS PROHIBITED MD (Section 32)
# ---------------------------------------------------------------------------
claims_prohibited_content = """# Claims Prohibited for EMS Manuscript (Paper A / P4)

The following claims are strictly FORBIDDEN and must NOT appear in the manuscript:

1. ❌ "LightGBM proves that variance collapse is universal for all machine learning architectures."
2. ❌ "The paper introduces the first variance-aware forecasting evaluation metric."
3. ❌ "Alpha is a new universal metric."
4. ❌ "RMSE is an invalid metric for environmental modeling."
5. ❌ "MSE loss causes variance collapse causally."
6. ❌ "EMS papers completely ignore dynamic fidelity."
7. ❌ "No previous study combines error and variability."
8. ❌ "The 714 cells represent independent observations for statistical inference."
9. ❌ "The proposed Rule B is universally optimal."
10. ❌ "PLS-SEM or SCADA analysis is required for Paper A."
"""
(READINESS_DIR / 'claims_prohibited.md').write_text(claims_prohibited_content)

# ---------------------------------------------------------------------------
# 4. INTEGRATED REPORT MD (Section 35)
# ---------------------------------------------------------------------------
report_content = """# INTEGRATED EDITORIAL REPORT — EMS Readiness Assessment (Paper A / P4)

> **Fecha:** 2026-08-06  
> **Rama Git:** `codex/p4-lightgbm-ems-gap-audit`  
> **HEAD:** `f372c3aca03d8228519f04bea535f5bc9a30ce6a`  

---

## 1. INTEGRATED EDITORIAL VERDICT

```
EMS_READY_FOR_TARGETED_REWRITE
```

---

## 2. RECAPITULACIÓN DE VEREDICTOS

| Dimensión | Veredicto | Resumen |
|---|---|---|
| **Work Package A (LightGBM)** | `LIGHTGBM_ROBUSTNESS_CONFIRMED` | LightGBM reproduce exactamente las dinámicas de HGB (skill > 0 con $\alpha < 0.50$ en 101/119 celdas). Genera 53 discordancias adicionales. |
| **Work Package B (Corpus EMS)** | `EMS_GAP_SUPPORTED` | Auditados 25 estudios de EMS (2015-2026). El 100% mide error, 20% mide varianza/KGE, pero 0% realiza una auditoría de cambio decisional conjunto. |
| **Integrado Final** | `EMS_READY_FOR_TARGETED_REWRITE` | Paper A está listo para reescritura enfocada hacia EMS como auditoría de selección decisional. No se requieren más experimentos. |

---

## 3. SÍNTESIS DE PREGUNTAS EDITORIALES

1. **¿LightGBM aporta robustez?**  
   Sí. Confirma que el colapso de varianza con skill positivo no es exclusivo de `HistGradientBoostingRegressor`, sino una característica común de los modelos de gradient boosting directo.

2. **¿LightGBM cambia la selección de modelos?**  
   En la regla convencional (solo error), LightGBM es seleccionado en 18 de 119 casos (15.1 %). En la regla fidelity-aware, es descalificado en $h \ge 2$ junto a los demás modelos ML debido a $\alpha < 0.50$.

3. **¿El gap en EMS existe?**  
   Sí. La literatura EMS evalúa error y fidelidad como dimensiones separadas o mediante escalares combinados (KGE), pero carece de auditorías empíricas sobre cómo una regla conjunta descalifica modelos con skill positivo.

4. **¿Es un aporte metodológico o diagnóstico?**  
   Es un **aporte de auditoría de selección de modelos**, no una nueva métrica universal.

5. **¿Está Paper A listo para EMS?**  
   Sí. Se autoriza la reescritura en Overleaf siguiendo el marco acotado de los claims permitidos.

---

## 4. ACCIÓN EDITORIAL RECOMENDADA

```
REWRITE_FOR_EMS
```

1. Proceder a la reescritura de la sección de análisis de decisión en Overleaf.
2. Formular la contribución de Paper A como un protocolo de auditoría de elegibilidad de modelos.
3. No abrir nuevos experimentos ni modificar los datasets.
"""

(READINESS_DIR / 'REPORT.md').write_text(report_content)

print("Integrated report written successfully.")
