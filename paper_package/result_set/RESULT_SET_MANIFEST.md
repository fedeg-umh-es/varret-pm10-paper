# Result-set manifest

The result set was assembled and visually inspected before narrative
rewriting. All numerical elements remain downstream of the frozen
canonical table or the verified deterministic sensitivity artifact.

| ID | Type | Scientific question | Canonical source | Generating script | Rows/subset | Main numerical claim | Narrative role | Current status | Final status |
|---|---|---|---|---|---:|---|---|---|---|
| F1 | Figure | How do skill and retained variance change with horizon? | `master_diagnostic_table.csv` | `generate_figures.py::figure_horizon_profiles` | 595 | Horizon-wise median skill and alpha for five families | Establishes the two-dimensional contrast before cell-level discordance | Existing deterministic plot | KEEP |
| F2A | Figure candidate | Where does discordance occur within each family? | `master_diagnostic_table.csv` | `generate_result_set_figures.py::candidate_a` | 595 | 101 formal discordant cells shown by open contours | Makes family clustering and the formal definition visible | Redesigned faceted candidate | KEEP as final F2 |
| F2B | Figure candidate | How does pooled structure compare with family summaries? | `master_diagnostic_table.csv` | `generate_result_set_figures.py::candidate_b` | 595 | Pooled scatter plus descriptive family rates | Alternative compact diagnostic | Redesign candidate | REMOVE_REDUNDANT |
| F3 | Figure | Does the eligibility consequence depend on cut-offs? | verified `rule_b_sensitivity.csv` | `generate_rule_sensitivity.py` | 3 x 3 threshold grid | 91.3--98.2% of Rule-A cells change | Answers the threshold objection | Existing deterministic heatmap | KEEP |
| T1 | Table | What profiles characterise the five families? | `master_diagnostic_table.csv` | `generate_tables.py` | 595 | Medians, counts, Rule A/B and discordance | Quantitative family summary | Existing table, copied to result set | KEEP |
| T2 | Table | How much does eligibility change? | `master_diagnostic_table.csv` | `generate_tables.py` | 595 | 277 -> 8; 269 changes; 97.1% | Quantitative climax | Existing decision table, copied to result set | KEEP |
| T3 | Table | What retained counts accompany the sensitivity percentages? | verified `rule_b_sensitivity.csv` | `generate_rule_sensitivity.py` | 3 x 3 grid | Retained cells and change percentages | Preserves exact sensitivity counts beside the heatmap | Detailed table | KEEP in main |
| T4 | Table | What was evaluated? | frozen provenance and verified labels | manual LaTeX table | five families | Model inputs and recoverability limits | Reproducibility support | Existing specification table | KEEP |

Obsolete reporting-gap plots, the former eligibility bar chart, the
486-study table, and unused legacy figures have no role in the final
argument and are not included in the v4 archive.
