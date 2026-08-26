# 🧮 Bias Audit Tool

Exploratory bias and fairness diagnostics for tabular datasets using Fairlearn and scikit-learn. The Streamlit app helps researchers inspect representation and group-wise performance disparities. It does not establish that a dataset or model is fair, unbiased, non-discriminatory, or legally compliant.

---

## 🚀 Features

- CSV upload and preprocessing recommendations, with a processed-data preview and CSV download
- Candidate sensitive attributes based on column-name and metadata heuristics — review before use
- User-supplied expected-distribution (benchmark) representation analysis
- Binary-classification modeling with train-only-fitted preprocessing
- Optional inclusion or exclusion of the selected sensitive attribute, including its direct encodings, as a model feature
- Held-out classification metrics, ROC AUC, a confusion matrix, an ROC curve, and permutation feature importance
- Group-wise fairness diagnostics, including Demographic Parity Difference and Equalized Odds Difference
- Count-plot visualizations for selected grouping columns and observed-vs-expected distribution charts

### Scope limits

- Exploratory diagnostic only — not a legal, regulatory, or compliance determination
- Supervised modeling is binary classification only
- One selected sensitive attribute is audited at a time
- Sensitive-attribute recommendations are heuristic and require human review
- Excluding the selected attribute removes its direct encodings, but does not remove proxy variables
- Benchmark-relative representation analysis requires a user-supplied expected distribution

---

## 🛠️ Installation
### 1. Clone the repository
```bash
git clone https://github.com/nabinkim0318/abdi.git
cd abdi
```

### 2. (Optional) Install Poetry if not already installed
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Install all dependencies & set up pre-commit hooks
```bash
make install
```

### 4. Activate the Poetry-managed virtual environment
```bash
make setup
```

### 5. Run the app locally
```bash
make run
```

📁 Project Structure
```bash
app.py                             # Streamlit entry point
bias_audit_tool/
├── data/
│   └── data_loader.py             # CSV loading
├── modeling/
│   ├── fairness.py                # Representation and model fairness metrics
│   ├── model_selector.py          # Baseline classifier training and evaluation
│   └── target_validation.py       # Binary-target checks for the modeling path
├── preprocessing/
│   ├── modeling_pipeline.py       # Leakage-safe train/test modeling pipeline
│   ├── preprocess.py              # Preprocessing recommendations
│   ├── recommend_columns.py       # Heuristic candidate sensitive attributes
│   ├── summary.py                 # Preprocessing recommendation summary table
│   └── transform.py               # Exploratory (full-data) preprocessing
├── sample_data/
│   └── clinical_dataset_breast_cancer.csv
├── utils/
│   └── ui_helpers.py              # Streamlit helpers for preprocessing and modeling
├── visualization/
│   ├── ui_blocks.py               # Representation-audit UI
│   ├── visualization.py           # Live charts
│   └── evaluation_plots.py        # Held-out confusion matrix and ROC figures

tests/
├── test_fairness.py
├── test_model_evaluation_plots.py
├── test_modeling_pipeline.py
├── test_preprocess.py
├── test_product_surface.py
├── test_recommend_columns.py
├── test_target_validation.py

Makefile                           # Common tasks: run, lint, test
pyproject.toml                     # Project dependencies and build settings
requirements.txt                   # Plain dependency list (optional)
README.md                          # This file
LICENSE                            # MIT License
```
