# GEMINI.md — Project Context & Developer Guide

## Project: `varret-pm10-paper` (P33)

**Title**: *Variance Retention and Diagnostic Skill Adjustment in Multi-Horizon PM10 Forecasting Under Rolling-Origin Evaluation*  
**Author**: F. García Crespi  
**Repository Role**: Paper-first, audit-first reproducible scientific research codebase.

---

## 1. Scientific Overview & Core Question

This repository investigates whether positive forecasting skill in multi-horizon daily/hourly $\text{PM}_{10}$ forecasting under rolling-origin evaluation reflects genuine operational capability or deceptive **variance collapse** (termed *ghost skill*).

### Key Concepts & Metrics:
- **Nominal Skill ($S$)**: Standard baseline-relative metric (e.g., relative to persistence).
- **Variance Retention Ratio ($\alpha$)**: Ratio of predicted variance to observed variance $\sigma^2(\hat{y}) / \sigma^2(y)$.
  - $\alpha \approx 1.0$: Dynamic fidelity preserved.
  - $\alpha \ll 1.0$: Variance collapse (mean-reversion / ghost skill).
  - $\alpha > 1.0$: Variance inflation.
- **Variance-Penalized Skill ($S_{vp}$ / $\text{skill}_{vp}$)**: Diagnostic metric penalizing variance distortion.
- **Murphy Skill Decomposition**: Resolving error into resolution, reliability, and uncertainty components.
- **Diebold-Mariano (DM) Tests**: Horizon-wise statistical significance testing of paired forecast losses with autocorrelation corrections (Newey-West / auto-lag).
- **Exceedance & Event Metrics**: Accuracy and recall on threshold violations ($\text{PM}_{10} > 50\,\mu\text{g/m}^3$).

---

## 2. Architecture & Directory Layout

```
varret-pm10-paper/
├── src/                    # Core Python library
│   ├── data/               # Data loaders, downloaders, and cleaners
│   ├── diagnostics/        # Variance retention, alpha, Murphy decomposition, KGE
│   ├── evaluation/         # Diebold-Mariano tests, exceedance metrics, scoring rules
│   ├── features/           # Lag features and temporal indicators
│   ├── models/             # Baselines (persistence), linear/autoregressive, boosting
│   ├── plotting/           # Publication-ready matplotlib/seaborn figures
│   ├── reporting/          # Tables, LaTeX renderers, audit generators
│   ├── splits/             # Leakage-free rolling-origin splitter
│   ├── training/           # Model fit/predict orchestration
│   └── utils/              # I/O, hashing, logging, verification
├── scripts/                # Numbered pipeline steps & reproduction scripts
│   ├── 01_validate_raw_data.py
│   ├── 02_build_processed_datasets.py
│   ├── 03_run_baselines.py
│   ├── 04_run_linear_models.py
│   ├── 05_run_boosting_model.py
│   ├── 06_build_skill_tables.py
│   ├── 07_build_variance_retention_table.py
│   ├── 08_build_run_summary.py
│   ├── run_p33_pipeline.py          # Unified pipeline runner
│   ├── run_paper_a_empirical.py     # Paper A reproduction runner
│   └── recover_madrid_pm10.py       # Madrid station data recovery
├── configs/                # Dataset, model, and horizon configurations (YAML/JSON)
├── data/                   # Raw, processed, and manuscript benchmark data
├── outputs/                # Prediction tables, summary CSVs, and reproduction logs
├── docs/                   # Protocols, data dictionary, contracts, reproducibility audits
├── tests/                  # Pytest suite (rolling-origin integrity, causality, metrics)
├── Makefile                # Automated targets: paper, figures, data, reproduce, test
└── *.tex / *.bib           # Manuscript files (paper_a.tex, serra_manuscript.tex, etc.)
```

---

## 3. Data Contracts & Canonical Schemas

### 1. Canonical Processed Dataset (`data/processed/*.csv`)
- `date`: Timestamp / ISO-8601 Date
- `y`: Target concentration ($\mu\text{g/m}^3$)

### 2. Prediction Table (`outputs/predictions/*.csv`)
- `dataset`: Name/ID of station/dataset
- `model`: Model identifier (`persistence`, `sarima`, `ridge`, `lgbm`, etc.)
- `fold`: Rolling split index
- `origin_date`: Forecast origin date
- `horizon`: Forecast step ($h \in [1, H_{\max}]$)
- `date`: Target verification timestamp
- `y_true`: Ground truth observed value
- `y_pred`: Model forecast value

### 3. Diagnostic & Variance Retention Table (`outputs/tables/variance_retention_summary.csv`)
- `dataset`, `model`, `horizon`, `skill`, `alpha`, `skill_vp`, `collapse_flag`, `inflation_flag`, `near_ideal_flag`

---

## 4. Methodological & Coding Guardrails

1. **Strict Temporal Causality (No Lookahead Leakage)**:
   - Rolling-origin splits must never expose future observations to training folds.
   - Any scaler, standardizer, imputation, or feature engineering must be fit **strictly on training data** within each fold.
   - Missing observations in historical series are never backfilled from future timestamps.
2. **Baselines First**:
   - Persistence (and optional seasonal persistence) is the mandatory benchmark.
   - All skill metrics are evaluated relative to the persistence baseline.
3. **Paper-First Code Quality**:
   - Keep scripts modular, deterministic (fixed random seeds), and reproducible.
   - Avoid bloated abstractions or heavy frameworks unless required.
   - Maintain type annotations and clear docstrings on core scientific functions in `src/`.
4. **Auditability**:
   - Results, tables, and figures must have exact traceability back to source scripts and prediction files.

---

## 5. Key Commands & Execution

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# For exact historical reproduction dependencies:
# pip install -r requirements-reproduction.txt
```

### Running the Core Pipeline
```bash
# Run complete P33 pipeline:
python scripts/run_p33_pipeline.py

# Run specific diagnostic tables:
python scripts/07_build_variance_retention_table.py
```

### Reproducing Paper A & Compiling LaTeX
```bash
make data          # Download and verify raw data archives
make reproduce     # Run rolling-origin + holdout evaluations and generate figures
make figures       # Generate publication figures
make paper         # Compile manuscript with latexmk
make test          # Run test suite
```

### Testing
```bash
pytest -q
```
