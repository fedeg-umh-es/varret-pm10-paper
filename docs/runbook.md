# Runbook

1. Place the daily PM10 raw dataset in `data/raw/`.
2. Validate the raw file with `python scripts/01_validate_raw_data.py`.
3. Build the canonical processed dataset.
4. Run the baselines, linear model, and boosting model.
5. Build the skill table.
6. Build `outputs/tables/variance_retention_summary.csv`.
7. Build `outputs/reports/run_summary.txt`.
8. Run `pytest` to validate the minimal diagnostics layer.
