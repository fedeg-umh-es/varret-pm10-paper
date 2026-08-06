# P2 — Predictability Bound: Final Report

Branch: `claude/p2-predictability-bound-ldvcbi`  
Commit: `4a49b08b041c578ec5981dc1472125b2af0a4d59`


## VERDICT

**P2_ISOLATED_WORKTREE_READY_AND_TABLES_GENERATED**

_Note: This verdict means tables are ready for human scientific review. It does NOT mean P2_READY_FOR_PAPER._


## HECHOS VERIFICADOS

- Three daily PM10 series present and verified (Elche, Valencia Vivers, Zarra EMEP).
- Prediction CSVs verified (hgb_direct, ridge_direct, persistence for all stations;
  sarima, seasonal_naive, stl_ridge_direct for Valencia Vivers and Zarra EMEP).
- Sarima skill for Elche recovered from master_diagnostic_table.csv.
- Skill definition: Skill_RMSE = 1 - RMSE_model / RMSE_persistence.
- Persistence = last observed value (lag-1) for all horizons.
- Yule–Walker reference: AR(p=14) via scipy.linalg.solve, calendar-aligned valid pairs.
- No temporal compression applied (NaN preserved on reindexed calendar).

## ACF REPRODUCIBILITY

- Elche: ρ(1) recalculated = 0.5245, control ≈ 0.511, diff = 0.0135
- Valencia Vivers: ρ(1) recalculated = 0.6057, control ≈ 0.614, diff = 0.0083
- Zarra EMEP: ρ(1) recalculated = 0.6281, control ≈ 0.643, diff = 0.0149

## NUMERICAL DIAGNOSTICS

- OK: 21 (station×horizon) cells

## EMPIRICAL MODEL SELECTION

Best model selected from: hgb_direct, ridge_direct, sarima (only).
seasonal_naive and stl_ridge_direct excluded from best-model selection.

        station  horizon best_model_name  best_model_skill
          Elche        1    ridge_direct          0.106824
          Elche        2    ridge_direct          0.226415
          Elche        3    ridge_direct          0.257885
          Elche        4    ridge_direct          0.271506
          Elche        5    ridge_direct          0.278199
          Elche        6    ridge_direct          0.280038
          Elche        7    ridge_direct          0.271727
Valencia Vivers        1    ridge_direct          0.095462
Valencia Vivers        2      hgb_direct          0.177449
Valencia Vivers        3    ridge_direct          0.203627
Valencia Vivers        4    ridge_direct          0.224281
Valencia Vivers        5    ridge_direct          0.241513
Valencia Vivers        6    ridge_direct          0.243808
Valencia Vivers        7    ridge_direct          0.252876
     Zarra EMEP        1          sarima          0.146116
     Zarra EMEP        2      hgb_direct          0.194401
     Zarra EMEP        3      hgb_direct          0.231315
     Zarra EMEP        4          sarima          0.249554
     Zarra EMEP        5          sarima          0.327036
     Zarra EMEP        6          sarima          0.300306
     Zarra EMEP        7    ridge_direct          0.206461

## REFERENCE COMPARISON SUMMARY

- Best model exceeds YW reference: 2/21 (station×horizon) cells.
- Exceeding YW does NOT automatically imply non-linearity or exogenous information.
  Alternative explanations: sampling uncertainty, non-stationarity, insufficient order p, different samples, metric inconsistency, regularization.

## DISCREPANCIAS

- Elche sarima skill comes from master_diagnostic_table (multi-station run), not predictions.csv.
  The predictions.csv for Elche contains only hgb_direct, ridge_direct, persistence.
  Sarima persistence alignment for Elche cannot be verified at row level.
- P4 snapshot 390685f does not exist in remote execution environment (Mac-local reference).
  This is documented; data integrity verified via existing trazabilidad_tres_estaciones.csv.

## INTERPRETACIONES PERMITIDAS

- The YW reference provides a conditional linear predictability benchmark.
- Comparison is conditional on: stationarity, MSE loss, p=14, existing sample, skill_RMSE.
- ACF-reproduced ρ(1) values agree with control values within tolerance.
- Tables are ready for human scientific review.

## INTERPRETACIONES PROHIBIDAS

- Universal linear predictability ceiling (the reference is conditional).
- Definitive proof of non-linearity when best_model > YW.
- Exogenous causal information implied by skill gap.
- Generalization to all PM10 stations.
- P2 ready for paper (requires human scientific review and GO decision).
- Reopening scientific conclusions of Paper A.

## LIMITACIONES

- Three stations only; no claim of representativeness.
- AR order p=14 fixed; sensitivity to p not evaluated in P2.
- Stationarity assumed; PM10 may exhibit non-stationarity and seasonality.
- Missing data handled by valid-pair estimation; fewer pairs → higher uncertainty.
- Persistence baseline is lag-1 (same value repeated); this may not match h-step naive baselines.
- Sarima results for Elche: different sample than main prediction pipeline.
- Numerical stability verified; one REGULARIZED case would be flagged explicitly.
- P2 derives from same data as P4 snapshot; independent external validation not performed.

## SAFE_NEXT_ACTION

Revisión científica humana de las tres tablas P2 y decisión GO/NO-GO.

## BLOCKERS

None.
