# Method-detail audit

Generated for the P4 pre-submission revision. Unresolved details are omitted from the scientific claims rather than inferred.

| Detail | Status | Treatment |
|---|---|---|
| Rolling-origin design | VERIFIED | Five expanding folds in chronological order; exact per-fold row counts are not reported. |
| Train-only controls | VERIFIED | Protocol supports train-only preprocessing and fold-training P75 thresholds. |
| Bootstrap | VERIFIED | Percentile bootstrap, B=1000, seed 42. |
| BH multiplicity family | VERIFIED | Valid model--horizon p-values adjusted within each dataset. |
| SARIMA configuration | CONFLICTING | Historical repository records disagree; manuscript reports only the fixed-order family description and marks the exact order unresolved. |
| HGB/Ridge feature and hyperparameter details | NOT_FOUND | Model specification table states that exact details are not recoverable from frozen provenance. |
| STL decomposition settings | NOT_FOUND | Omitted rather than inferred. |
| Seasonal period implementation details | NOT_FOUND | Omitted rather than inferred. |
| Exact calendar endpoints | NOT_FOUND | The recovered station package does not establish one defensible common period. |
| Exact Zenodo DOI | NOT_FOUND | No DOI is asserted. |

Machine-readable status flag: `SARIMA_CONFIGURATION_CONFLICTING`.
