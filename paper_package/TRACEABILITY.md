# TRACEABILITY

Generated: 2026-08-09
Source: outputs/tables/master_diagnostic_table.csv
SHA-256: 6dfb12c5a8a1c2263ecfad71e441cd2af6f451c9b73ba9049b1986eaeee62af6

| Manuscript claim | Value | Source | Calculation |
| --- | --- | --- | --- |
| Median skill HGB direct | 0.205 | master_diagnostic_table.csv | df[model].skill.median() |
| Median alpha HGB direct | 0.151 | master_diagnostic_table.csv | df[model].alpha.median() |
| alpha<0.5 HGB direct | 118/119 | master_diagnostic_table.csv | (df[model].alpha<0.5).sum() |
| DM sig HGB direct | 111/119 | master_diagnostic_table.csv | df[model].dm_significant.sum() |
| Median skill Ridge direct | 0.219 | master_diagnostic_table.csv | df[model].skill.median() |
| Median alpha Ridge direct | 0.087 | master_diagnostic_table.csv | df[model].alpha.median() |
| alpha<0.5 Ridge direct | 118/119 | master_diagnostic_table.csv | (df[model].alpha<0.5).sum() |
| DM sig Ridge direct | 117/119 | master_diagnostic_table.csv | df[model].dm_significant.sum() |
| Median skill SARIMA | 0.208 | master_diagnostic_table.csv | df[model].skill.median() |
| Median alpha SARIMA | 0.095 | master_diagnostic_table.csv | df[model].alpha.median() |
| alpha<0.5 SARIMA | 110/119 | master_diagnostic_table.csv | (df[model].alpha<0.5).sum() |
| DM sig SARIMA | 44/119 | master_diagnostic_table.csv | df[model].dm_significant.sum() |
| Median skill Seasonal naive | -0.026 | master_diagnostic_table.csv | df[model].skill.median() |
| Median alpha Seasonal naive | 1.0 | master_diagnostic_table.csv | df[model].alpha.median() |
| alpha<0.5 Seasonal naive | 0/119 | master_diagnostic_table.csv | (df[model].alpha<0.5).sum() |
| DM sig Seasonal naive | 39/119 | master_diagnostic_table.csv | df[model].dm_significant.sum() |
| Median skill STL+Ridge | -1.107 | master_diagnostic_table.csv | df[model].skill.median() |
| Median alpha STL+Ridge | 1.399 | master_diagnostic_table.csv | df[model].alpha.median() |
| alpha<0.5 STL+Ridge | 0/119 | master_diagnostic_table.csv | (df[model].alpha<0.5).sum() |
| DM sig STL+Ridge | 119/119 | master_diagnostic_table.csv | df[model].dm_significant.sum() |
| Rule A total | 277 | master_diagnostic_table.csv | (skill>0) & dm_significant |
| Rule B total | 8 | master_diagnostic_table.csv | Rule A & alpha>=0.5 & recall_p75>=0.20 |
| Decision changes | 269 | master_diagnostic_table.csv | Rule A - Rule B |
| Decision change proportion | 97.1% | master_diagnostic_table.csv | changes/rule_a*100 |
| Spearman rho(alpha,skill) | -0.863 | master_diagnostic_table.csv | spearmanr(alpha,skill) |
| Discordant cases | 101 | master_diagnostic_table.csv | Rule A & recall>=0.20 & alpha<0.5 |
| Total cells | 595 | master_diagnostic_table.csv | len(df) |
| Models | 5 | master_diagnostic_table.csv | df.model.nunique() |
| Stations | 17 | master_diagnostic_table.csv | df.station_id.nunique() |
| Horizons | 7 | master_diagnostic_table.csv | df.horizon.nunique() |
