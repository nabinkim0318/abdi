# 🧮 Bias Audit Tool

Exploratory bias and fairness diagnostics for tabular datasets using Fairlearn and scikit-learn. The Streamlit app helps researchers inspect representation and group-wise performance disparities. It does not establish that a dataset or model is fair, unbiased, non-discriminatory, or legally compliant.

The bundled demo dataset is fully synthetic. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md). User-uploaded datasets are supplied by the user and may contain sensitive information. The bundled repository demo itself is synthetic.

---

## 🚀 Features

- CSV upload and preprocessing recommendations, with a processed-data preview and CSV download
- Candidate sensitive attributes based on column-name and metadata heuristics — review before use
- User-supplied expected-distribution (benchmark) representation analysis
- Binary-classification modeling with train-only-fitted preprocessing
- Optional inclusion or exclusion of the selected sensitive attribute, including its direct encodings, as a model feature
- Held-out classification metrics, ROC AUC, a confusion matrix, an ROC curve, and permutation feature importance
- Group-wise fairness diagnostics, including Demographic Parity Difference and Equalized Odds Difference
- Input guardrails detect duplicate CSV headers and non-finite numeric values before modeling.
- The binary modeling path checks minimum dataset/class support and warns on extreme class imbalance.
- Uploaded datasets are tracked by content fingerprint so changed files with reused filenames do not reuse stale analysis state.
- Count-plot visualizations for selected grouping columns and observed-vs-expected distribution charts

### Scope limits

- Exploratory diagnostic only — not a legal, regulatory, or compliance determination
- Supervised modeling is binary classification only
- One selected sensitive attribute is audited at a time
- Sensitive-attribute recommendations are heuristic and require human review
- Excluding the selected attribute removes its direct encodings, but does not remove proxy variables
- Benchmark-relative representation analysis requires a user-supplied expected distribution
- Duplicate headers, non-finite numeric values, and below-minimum class/row support block modeling; they are not auto-repaired
- Extreme class imbalance produces a heuristic warning only — classes are not resampled and thresholds are not changed

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

## Demo walkthrough (synthetic CSV)

Use the committed synthetic demo, not any external clinical extract.

1. Start the app (`make run` or `streamlit run app.py`).
2. Upload `bias_audit_tool/sample_data/demo.csv`.
3. Apply the suggested preprocessing, then choose **`demo_group_mapped`** as the sensitive attribute. Exploratory one-hot encoding plus the existing dummy-merge heuristic reconstructs `demo_group` as `demo_group_mapped`. `age_band_*` dummy columns may also appear as candidates because the name contains `age` — use `demo_group_mapped` for this walkthrough.
4. For representation analysis, paste the synthetic demonstration benchmark from `bias_audit_tool/sample_data/demo_benchmark.json` (proportions that sum to 1.0). This is a software fixture, not a population prevalence.
5. Enable modeling and select **`outcome`** as the binary target.
6. Run model evaluation. Inspect the classification report, ROC-AUC, Confusion Matrix, ROC Curve, permutation importance, group-wise fairness diagnostics, DP Difference, and EO Difference.

Do not interpret demo group differences as real-world demographic findings.

📁 Project Structure
```bash
app.py                             # Streamlit entry point
bias_audit_tool/
├── data/
│   ├── data_loader.py             # CSV loading
│   ├── upload_state.py            # Content-fingerprint session identity
│   └── validation.py              # Header, finite-value, and support checks
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
│   ├── demo.csv                   # Synthetic portfolio demo (see DATA_PROVENANCE.md)
│   └── demo_benchmark.json        # Synthetic demonstration benchmark
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
├── test_synthetic_demo.py
├── test_target_validation.py
├── test_data_validation.py
├── test_upload_state.py

scripts/
└── generate_demo_data.py          # Deterministic synthetic demo generator (seed 42)

Makefile                           # Common tasks: run, lint, test
pyproject.toml                     # Project dependencies and build settings
requirements.txt                   # Plain dependency list (optional)
README.md                          # This file
DATA_PROVENANCE.md                 # Synthetic demo provenance and data dictionary
LICENSE                            # MIT License
```
