# Paper A numerical traceability

Primary source: `outputs/tables/master_diagnostic_table.csv`.
SHA-256: `6dfb12c5a8a1c2263ecfad71e441cd2af6f451c9b73ba9049b1986eaeee62af6`.
All counts below were recomputed locally from the canonical table; no synthetic arm is used.

| Claim | Value | Subset | Calculation / script | Status |
|---|---:|---|---|---|
| Diagnostic rows | 595 | all rows | `len(df)`; integrity verifier | VERIFIED |
| Stations | 17 | `station_id` | `nunique()` | VERIFIED |
| Models | 5 | `model` | `nunique()`; exact model set check | VERIFIED |
| Horizons | 7 | `horizon` | `nunique()`; values 1--7 | VERIFIED |
| LightGBM rows | 0 | `model` | exact exclusion check | VERIFIED |
| Rule A | 277 | all cells | `skill > 0 & dm_significant` | VERIFIED |
| Rule B | 8 | all cells | Rule A & `alpha >= .50` & `recall_p75 >= .20` | VERIFIED |
| Decision changes | 269 | Rule-A cells | `277 - 8` | VERIFIED |
| Change proportion | 97.1% | Rule-A cells | `269 / 277 * 100` | VERIFIED |
| Spearman rho(alpha, skill) | -0.863 | all cells | `scipy.stats.spearmanr(alpha, skill)` | VERIFIED |
| Discordant cells | 101 | Rule A & recall P75 >= .20 & alpha < .50 | boolean count | VERIFIED |
| HGB Rule A -> B | 111 -> 1 | HGB | grouped boolean counts | VERIFIED |
| Ridge Rule A -> B | 117 -> 1 | Ridge | grouped boolean counts | VERIFIED |
| SARIMA Rule A -> B | 44 -> 1 | SARIMA | grouped boolean counts | VERIFIED |
| Seasonal naive Rule A -> B | 5 -> 5 | seasonal naive | grouped boolean counts | VERIFIED |
| STL+Ridge Rule A -> B | 0 -> 0 | STL+Ridge | grouped boolean counts | VERIFIED |
| HGB collapse | 118/119 | HGB | `alpha < .50` count | VERIFIED |
| Ridge collapse | 118/119 | Ridge | `alpha < .50` count | VERIFIED |
| SARIMA collapse | 110/119 | SARIMA | `alpha < .50` count | VERIFIED |

## Definitions used in the paper

`alpha` is the canonical variance ratio `Var(y_pred)/Var(y_true)` with `ddof=0`; it is not a standard-deviation ratio. Table 2 uses median skill and median alpha, plus counts for collapse, DM significance, Rule A, Rule B, and discordance. Table 3 uses counts and arithmetic changes. Figure 2 uses median skill and alpha by model and horizon.

## Provenance boundary

The canonical integrity verifier returned 16/16 PASS. Historical quarantine and audit documents may mention synthetic material for forensic traceability, but no such rows, statistics, or claims enter the manuscript, tables, figures, supplementary text, or final Overleaf ZIP.
