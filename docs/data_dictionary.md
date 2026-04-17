# Data Dictionary

## Canonical Processed Dataset

- `date`: daily timestamp
- `y`: PM10 target value

## Predictions Table

- `dataset`: dataset identifier
- `model`: model identifier
- `fold`: rolling-origin fold identifier
- `origin_date`: forecast origin date
- `horizon`: lead time in days
- `date`: target date
- `y_true`: observed target value
- `y_pred`: predicted target value

## Skill Table

- `dataset`: dataset identifier
- `model`: model identifier
- `horizon`: lead time in days
- `skill`: baseline-relative skill

## Variance Retention Summary

- `dataset`
- `model`
- `horizon`
- `skill`
- `alpha`
- `skill_vp`
- `collapse_flag`
- `inflation_flag`
- `near_ideal_flag`
