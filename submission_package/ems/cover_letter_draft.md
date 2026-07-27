# Cover Letter for Submission to Environmental Modelling & Software

**Date:** July 27, 2026  
**To:** The Editor-in-Chief and Editorial Board, *Environmental Modelling & Software*  
**Subject:** Submission of Research Paper: "Variance Retention: A Forecast-Verification Diagnostic for Skilful-but-Smoothed Point Forecasts of Daily PM10"

Dear Editors,

We submit our manuscript entitled **"Variance Retention: A Forecast-Verification Diagnostic for Skilful-but-Smoothed Point Forecasts of Daily PM10"** for consideration as a Research Paper in *Environmental Modelling & Software*.

### Context & Novelty
Deterministic point forecasts of atmospheric pollutants are routinely verified using mean-squared error (RMSE) or persistence-relative skill scores. However, a model can achieve positive RMSE skill while severely over-smoothing point trajectories, flattening peak concentrations critical for public health advisories and episode detection.

In this work, we formalise **variance retention** $\alpha(h) = \operatorname{Var}(\hat{y}_h)/\operatorname{Var}(y_h)$ as a lightweight, post-evaluation verification diagnostic. Across a rolling-origin evaluation of 17 heterogeneous monitoring stations from the Spanish MITECO national network, 5 model families, and 7 forecast horizons ($h=1,\dots,7$), we demonstrate that:
1. Direct machine-learning forecasters (HGB and Ridge) attain positive RMSE skill in 100% of station-horizon cells but exhibit near-universal variance collapse ($\alpha < 0.5$) in 99.2% of cells.
2. Classical statistical baselines (SARIMA) reproduce this variance collapse (92.4% of cells), whereas variance-preserving decomposition approaches (STL+Ridge) preserve amplitude at the expense of severe error penalties and high false-alarm rates.
3. Exceedance diagnostics confirm that over-smoothed ML forecasts suffer a drastic drop in episode recall (falling below 10% for $h \ge 3$ at P90).

### Alignment with Journal Scope
Our contribution is specifically framed as a **post-evaluation diagnostic methodology for environmental software and forecast verification practice**. We provide a reproducible, open-source workflow and distil minimum reporting practices for deterministic air-quality model verification, directly aligning with *Environmental Modelling & Software*'s core focus on transparent, reproducible, and operationally relevant environmental software tools.

### Declarations
- This manuscript represents original work and has not been published previously, nor is it under consideration for publication elsewhere.
- All authors have approved the final manuscript and agree with its submission to *Environmental Modelling & Software*.
- The authors declare no competing financial or personal interests.
- Complete code, execution scripts, and evidence packages are publicly available at: \url{https://github.com/fedeg-umh-es/varret-pm10-paper}.

Thank you for considering our manuscript. We look forward to receiving the reviewers' feedback.

Sincerely,

**Federico García Crespi** (Corresponding Author)  
Department of Computer Engineering, Universidad Miguel Hernández, Elche, Spain  
Email: fedeg@umh.es  

**Julio Alberto Ramos Martínez**  
Department of Statistics, Mathematics and Computer Science, Universidad Miguel Hernández, Elche, Spain  
Email: j.ramos@umh.es  
