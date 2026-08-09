# P4 Paper-A traceability

Primary empirical source: `outputs/tables/master_diagnostic_table.csv`. SHA-256: `6dfb12c5a8a1c2263ecfad71e441cd2af6f451c9b73ba9049b1986eaeee62af6`.

| Claim | Value | Source/subset | Calculation | Status |
|---|---:|---|---|---|
| Rows | 595 | canonical table | `len(df)` | VERIFIED |
| Stations | 17 | canonical `station_id` | `nunique()` | VERIFIED |
| Models | 5 | canonical `model` | exact model-set check | VERIFIED |
| Horizons | 7 | canonical `horizon` | unique values 1--7 | VERIFIED |
| Rolling-origin folds | 5 | protocol and run manifests | documented expanding-fold design | VERIFIED |
| LightGBM rows | 0 | canonical `model` | exclusion check | VERIFIED |
| Rule A | 277 | all canonical cells | `skill>0 & dm_significant` | VERIFIED |
| Rule B | 8 | Rule-A cells | Rule A & alpha>=.50 & recall>=.20 | VERIFIED |
| Changes | 269 | Rule-A cells | `277-8` | VERIFIED |
| Change proportion | 97.1% | Rule-A cells | `269/277*100` | VERIFIED |
| Formal discordance | 101 | Rule A & recall>=.20 & alpha<.50 | boolean count | VERIFIED |
| HGB discordance | 54 | HGB subset | grouped boolean count | VERIFIED |
| Ridge discordance | 43 | Ridge subset | grouped boolean count | VERIFIED |
| SARIMA discordance | 4 | SARIMA subset | grouped boolean count | VERIFIED |
| HGB collapse | 118/119 | HGB subset | alpha<.50 | VERIFIED |
| Ridge collapse | 118/119 | Ridge subset | alpha<.50 | VERIFIED |
| SARIMA collapse | 110/119 | SARIMA subset | alpha<.50 | VERIFIED |
| Pooled rho(alpha,skill) | -0.863 | all canonical cells | Spearman correlation | VERIFIED; descriptive only |
| Bootstrap | B=1000, seed 42 | alpha diagnostics | canonical implementation | VERIFIED |
| Sensitivity range | 91.3--98.2% changes | 3x3 neighbourhood | existing `rule_b_sensitivity.csv`, filtered to alpha .40/.50/.60 and recall .10/.20/.30; displayed in Figure 3 and Supplementary Table S3 | VERIFIED |
| Abstract word count | 139 | final abstract | PDF text extraction, lexical-token count; below 150-word limit | VERIFIED |

## Rule definitions

Rule A is positive persistence-relative RMSE skill and significant BH-adjusted DM comparison. Rule B adds alpha >= 0.50 and P75 recall >= 0.20. The thresholds are operational audit criteria, not universal or optimised values.

## Provenance boundaries

The empirical integrity verifier returned 16/16 PASS. The structured cells share stations and time series and are not independent replicates. Historical synthetic material and the separate 486-abstract reporting block are outside the revised manuscript evidence chain. Full model specifications are reported in Supplementary Table S2; exact sensitivity counts are reported in Supplementary Table S3.

`UNTRACEABLE_MAIN_TEXT_NUMBERS = 0`: every quantitative claim retained
in the manuscript maps to the canonical table or verified deterministic
sensitivity artifact above. Unresolved methodological details are
documented separately and are not used as unsupported numerical claims.
