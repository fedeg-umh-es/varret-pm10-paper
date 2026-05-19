# E1-RR Post-Evaluation Contract

## Current Work Package

This repository is currently used only for the post-evaluation of E1-RR outputs through variance retention, `alpha`, and `Skill_VP` diagnostics.

The immediate objective is to determine whether persistence-relative skill observed in E1-RR is accompanied by sufficient forecast variance retention across horizons, or whether the results show patterns compatible with variance collapse / ghost skill.

## In Scope

- E1-RR outputs only.
- Daily PM10 forecasting outputs.
- Horizons already evaluated in E1-RR, expected range: `h = 1,...,7`.
- Leakage-free rolling-origin predictions produced upstream.
- Persistence-relative skill already computed upstream or recomputed from E1-RR predictions.
- Variance retention diagnostics by dataset, model, and horizon.
- Diagnostic flags for collapse, inflation, and near-ideal variance retention.
- Final paper-ready tabular output.

## Out of Scope

- E2-MET.
- E3-PROB.
- New meteorological ablations.
- Probabilistic or conformal extensions.
- New model families.
- Architecture search.
- Reframing this repository as a general forecasting package.
- Claims that `Skill_VP` is a universal replacement for standard skill scores.

## Required Input Contract

The minimum starting point is an E1-RR predictions table with the following fields:

| Column | Meaning |
|---|---|
| `dataset` | dataset or station identifier |
| `model` | model identifier |
| `fold` | rolling-origin fold identifier |
| `origin_date` | forecast origin date |
| `horizon` | lead time in days |
| `date` | target date |
| `y_true` | observed PM10 value |
| `y_pred` | model prediction |

If the persistence prediction is not present in the same table, a separate skill table must be provided or generated consistently from E1-RR outputs.

## Required Output Contract

The minimum output is:

```text
outputs/tables/variance_retention_summary.csv
```

with these columns:

| Column | Meaning |
|---|---|
| `dataset` | dataset or station identifier |
| `model` | model identifier |
| `horizon` | lead time in days |
| `skill` | persistence-relative skill |
| `alpha` | variance-retention indicator |
| `skill_vp` | diagnostic skill adjusted by variance retention |
| `collapse_flag` | forecast variance is suspiciously low |
| `inflation_flag` | forecast variance is suspiciously high |
| `near_ideal_flag` | variance retention is close to observed variability |

## Methodological Guardrails

- Interpret `Skill_VP` as a diagnostic post-evaluation metric, not as a new universal forecast accuracy score.
- Do not infer physical predictability from positive skill alone.
- Treat high skill with very low `alpha` as a warning pattern, not as proof of invalidity by itself.
- Keep all claims bounded to E1-RR unless new experiments are explicitly added later.
- Preserve the distinction between empirical result, diagnostic interpretation, and manuscript claim.

## Minimal Start Checklist

1. Clone the repository.
2. Create a clean Python environment.
3. Install dependencies from `requirements.txt`.
4. Locate or export the E1-RR predictions table.
5. Confirm that the input columns match this contract.
6. Run the variance-retention table builder.
7. Inspect `outputs/tables/variance_retention_summary.csv` before opening any new experiment front.
