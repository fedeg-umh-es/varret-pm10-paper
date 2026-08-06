# Paper A — Manuscript Editorial Profile

**PROJECT LINE:** P3 — Ghost Skill / Variance Retention  
**PAPER:** Paper A  
**CANONICAL SOURCE:** `paper_a.tex`  
**HEAD COMMIT:** `2939d26fb4e3447347ed9d6ba8e2d244462cb67b` (PR #7 merged to `main`)  
**EVIDENCE BASE:** 17 MITECO PM10 stations, daily frequency, lead times $h \in \{1, \ldots, 7\}$, 5 model families.

---

## 1. Core Profile Attributes

- **Title:** *Variance Retention: A Forecast-Verification Diagnostic for Skilful-but-Smoothed Point Forecasts of Daily PM10*
- **Primary Topic:** Forecast verification methodology and post-evaluation diagnostics for air-quality point forecasts.
- **Secondary Topics:** Rolling-origin multi-horizon evaluation, persistence baseline comparison, amplitude retention ($\alpha$), event exceedance detection, Murphy MSE decomposition.
- **Contribution Type:** Diagnostic verification methodology + empirical reporting evaluation practice.
- **Methodological Level:** Post-evaluation diagnostic layer (formalized variance ratio $\alpha(h)$, auxiliary $Skill_{VP}$, threshold sensitivity analysis).
- **Application Domain:** Environmental air quality (daily PM10 across 17 heterogeneous Spanish monitoring stations from the national MITECO network).
- **Primary Audience:** Environmental modellers, air-quality forecasters, forecast verification researchers, environmental software developers.
- **Novelty Claim:** Exposing amplitude collapse ($\alpha < 0.5$) coexisting with statistically significant RMSE skill gains over persistence in non-naive ML/statistical PM10 point forecasters, supported by a post-evaluation diagnostic workflow.

---

## 2. Scope Boundaries & Rejection Fronts

- **What the paper IS NOT:**
  - NOT a new machine-learning architecture or deep-learning model design.
  - NOT a new atmospheric chemistry or chemical transport model (CTM).
  - NOT a universal mathematical theory of forecast skill.
  - NOT an operational deployment field trial.
  - NOT a Paper B manuscript (0 occurrences of $H^*$, right-censoring, or prequential model selection).

- **Critical Desk-Reject Avoidance Strategy:**
  - **Front 1 (Pure ML/AI journals):** Desk-reject risk if misrouted to computer-science venues expecting novel neural network architectures.
  - **Front 2 (Pure Atmospheric Physics journals):** Desk-reject risk if misrouted to physical meteorology venues expecting atmospheric dynamics or chemical transport physics.
  - **Optimal Target Envelope:** Journals dedicated to **environmental modelling methodology, diagnostic evaluation frameworks, and applied environmental data science**.
