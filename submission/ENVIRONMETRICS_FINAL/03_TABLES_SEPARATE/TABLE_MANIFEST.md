# 03_TABLES_SEPARATE — Table Manifest & Metadata

This directory provides standalone LaTeX source (`.tex`) and compiled vector PDF (`.pdf`) versions of every table in the final manuscript, prepared for separate presentation per *Environmetrics* author guidelines.

## Table Manifest

| Table # | Title / Content Summary | Standalone Files | Underlying Source Table | Producer Script | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **Table 1** | Continuous error metrics (RMSE, persistence RMSE, RMSE skill) by model and horizon | `Table_1.tex`<br>`Table_1.pdf` | `outputs/publication_tables/pub_table_1_error_metrics.csv` | `scripts/44_generate_publication_source_tables.py` | `PASS` (Exact numbers match manuscript and source tables) |
| **Table 2** | Dynamic-fidelity metrics (variance retention, temporal variability, amplitude ratio, event amplitude retention) | `Table_2.tex`<br>`Table_2.pdf` | `outputs/publication_tables/pub_table_2_dynamic_fidelity.csv` | `scripts/44_generate_publication_source_tables.py` | `PASS` (Exact numbers match manuscript and source tables) |
| **Table 3** | Exceedance event metrics (TP, FP, FN, TN, POD, CSI) at training-partition $p_{75}$ threshold | `Table_3.tex`<br>`Table_3.pdf` | `outputs/publication_tables/pub_table_3_event_metrics.csv` | `scripts/44_generate_publication_source_tables.py` | `PASS` (Exact numbers match manuscript and source tables) |
| **Table 4** | Ghost-skill structure, fold-stability summary, and pairwise rank-reversal indicators | `Table_4.tex`<br>`Table_4.pdf` | `outputs/publication_tables/pub_table_4_ghost_skill_structure.csv` | `scripts/44_generate_publication_source_tables.py` | `PASS` (Exact numbers match manuscript and source tables) |

## Note on Journal Presentation Guidance
Per *Environmetrics* presentation guidelines, tables may be presented on separate pages following the reference list or uploaded separately. These standalone `.tex` and `.pdf` files provide clean, independent representations of each table.
