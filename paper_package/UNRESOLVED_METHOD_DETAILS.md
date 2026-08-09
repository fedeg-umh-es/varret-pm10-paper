# Method-detail audit

Generated 2026-08-09 from local repository inspection. No new experiment was run.

| Detail | Status | Evidence / decision |
|---|---|---|
| Rolling-origin protocol | VERIFIED | `docs/protocol.md`, `config/config.yaml`, `outputs/reproduction/run_manifest_rolling_origin.json`; five expanding folds are stated in the protocol artefacts. |
| Daily frequency and horizons | VERIFIED | `docs/protocol.md` and canonical table: horizons 1--7. |
| Train-only controls and P75 threshold fitting | VERIFIED | `docs/protocol.md`, `config/thresholds.yaml`, and evidence manifest. |
| SARIMA configuration | CONFLICTING | `src/models/sarima_model.py`/`scripts/02_generate_sarima_predictions.py` show (1,0,1)(1,0,1,7), while an older reproduction manifest records a different hourly specification. The manuscript reports only the fixed-order family label and does not use the conflicting hourly detail. |
| Bootstrap for alpha | VERIFIED | `src/diagnostics/variance.py`: percentile bootstrap, `n_boot=1000`, seed 42; canonical table includes CI columns. |
| Zenodo DOI | NOT_FOUND | No verified DOI is used in the manuscript or package. |
| Exact calendar endpoints | NOT_FOUND | The canonical aggregate/provenance package does not establish a single defensible start/end period for this rewrite. |
| Exact per-fold train/test row counts | NOT_FOUND | The aggregate table is cell-level; the manuscript reports the verified five-fold rolling-origin design without inventing row counts. |
| HGB/Ridge lag and hyperparameter details | NOT_FOUND | The canonical table identifies the families, but does not provide a complete auditable configuration contract for this package. |

The unresolved/conflicting details are intentionally omitted or bounded in the manuscript rather than filled by inference.
