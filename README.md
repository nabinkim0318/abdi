# 🧮 Bias Audit Tool

A modular and interactive tool for auditing bias and fairness in machine learning datasets and models. Powered by `Streamlit`, `scikit-learn`, and `fairlearn`, it helps data scientists and researchers evaluate performance disparities across demographic groups.

---

## 🚀 Features

- Easy upload and preprocessing of CSV datasets
- Automatic recommendation of sensitive columns
- Fairness metric computation (e.g., Equalized Odds, Demographic Parity)
- Basic model training and bias evaluation
- Summary statistics, visualizations, and automated reports

---

🛠️ Installation
# 1. Clone the repository
git clone https://github.com/nabinkim0318/abdi.git
cd abdi

# 2. (Optional) Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# 3. Install all dependencies & set up pre-commit hooks
make install

# 4. Activate the Poetry-managed virtual environment
make setup

📁 Project Structure
```bash
bias_audit_tool/
├── app.py                         # Main Streamlit entry point
├── data/
│   └── data_loader.py             # Dataset loading and validation
├── modeling/
│   ├── fairness.py                # Fairness metric computations (e.g., EO, DP)
│   ├── interpretation.py          # Model interpretation (e.g., SHAP)
│   └── model_selector.py          # Baseline model training and evaluation
├── preprocessing/
│   ├── preprocess.py              # Preprocessing workflow manager
│   ├── recommend_columns.py       # Automatic sensitive attribute detection
│   ├── summary.py                 # Feature summary statistics
│   └── transform.py               # Missing value imputation, scaling
├── report/
│   └── report_generator.py        # Report creation (PDF/HTML)
├── sample_data/
│   └── demo.csv                   # Example dataset for demonstration
├── stats/
│   └── stats_analysis.py          # Statistical testing utilities
├── utils/
│   └── ui_helpers.py              # Streamlit helper components and layout logic
├── visualization/
│   ├── ui_blocks.py               # Modular Streamlit UI blocks
│   └── visualization.py           # Charts and visualization functions

tests/
├── test_preprocess.py             # Unit test for preprocessing logic
├── test_recommend_columns.py      # Unit test for column detection
├── test_failure.py                # CI breakage test stub

Makefile                           # Common tasks: run, lint, test
pyproject.toml                     # Project dependencies and build settings
requirements.txt                   # Plain dependency list (optional)
README.md                          # This file
LICENSE                            # MIT License
