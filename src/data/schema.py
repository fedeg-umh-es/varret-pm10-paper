"""Schema definitions for canonical P33 data contracts."""

CANONICAL_DATASET_COLUMNS = ["date", "y"]
PREDICTIONS_COLUMNS = ["dataset", "model", "fold", "origin_date", "horizon", "date", "y_true", "y_pred"]
SKILL_COLUMNS = ["dataset", "model", "horizon", "skill"]
VARIANCE_SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "horizon",
    "skill",
    "alpha",
    "skill_vp",
    "collapse_flag",
    "inflation_flag",
    "near_ideal_flag",
]
