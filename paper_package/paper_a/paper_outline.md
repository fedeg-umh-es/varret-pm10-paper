# Paper A — canonical outline

1. **Provisional title:** *Variance Retention as a Diagnostic Complement to
   Persistence-Relative Skill in Daily PM$_{10}$ Forecasting.*

2. **Dominant question:** Can a daily PM$_{10}$ point forecast achieve positive
   persistence-relative RMSE skill while failing to retain the observed
   day-to-day variance (variance collapse / loss of dynamic fidelity)?

3. **Thesis (one sentence):** Across 17 stations and 7 horizons, the leading
   point forecasters are simultaneously *skilful* under RMSE and *dynamically
   over-smoothed* (variance-collapsed), so RMSE skill alone can mask a
   fidelity failure that a simple variance-retention diagnostic exposes.

4. **Promise to the reader:** A low-cost, post-evaluation diagnostic layer
   (α and Skill$_{VP}$) computed from predictions you already have, that
   separates "accurate" from "dynamically faithful".

5. **Contributions (2–4):**
   - Document a skill-with-collapse pattern (HGB/Ridge 118/119, SARIMA 110/119).
   - Formalise variance retention α(h)=Var(ŷ)/Var(y) and the auxiliary
     Skill$_{VP}$=Skill·α as post-evaluation diagnostics.
   - Show, with controls (seasonal naive, STL+Ridge), that amplitude
     preservation is necessary but not sufficient for useful skill.
   - Threshold-robustness + DM/Murphy/exceedance triangulation.

6. **Section architecture:** Intro → Related work/positioning → Data & sites →
   Methodology (skill, α, Skill$_{VP}$, thresholds) → Results (skill profiles,
   α profiles, skill–α space, sensitivity) → Discussion → Limitations →
   Conclusion.

7. **Main table:** five-model diagnostic summary
   (`model_family_diagnostic_summary.csv`).

8. **Main figure:** α profiles (F4) + skill–α space (F5).

9. **To supplement:** per-cell variance-retention table, bootstrap α CIs,
   PRISMA reporting-gap audit, exceedance and Murphy detail, three-model early
   diagnostic table.

10. **Prohibited claims:**
    - α is a standard-deviation / amplitude ratio, or "α=0.5 = half the
      amplitude" (it is half the *variance*).
    - The α Var/SD correction is a headline result (it is a precision fix; no
      number changed).
    - Variance collapse implies the models are useless (they are RMSE-skilful).
    - Causal or physical-limit claims about why variance collapses.
    - Equating this α with the KGE α component (a separate SD-ratio diagnostic).
