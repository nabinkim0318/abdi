# abdi
A lightweight Streamlit-based tool for detecting and reporting statistical and algorithmic bias in biomedical datasets. ABDI offers automated preprocessing suggestions, fairness audits across demographic groups, and interpretable reports (HTML/PDF) tailored for clinical researchers, DEI officers, and regulatory teams.

bias_audit_tool/
├── app.py                         # Main Streamlit app entrypoint
├── data/                          # Data loading utilities
├── modeling/                      # Modeling logic (fairness metrics, model selection, interpretation)
├── preprocessing/                # Data preprocessing: cleaning, transformation, column recommendations
├── report/                        # Report generation and formatting
├── sample_data/                  # Sample input CSVs
├── stats/                         # Statistical analysis tools
├── utils/                         # UI helpers and shared functions
├── visualization/                # Visualization components for Streamlit interface
tests/                             # Unit tests for core modules
Makefile                           # Dev shortcuts (e.g., run app, run tests)
pyproject.toml, requirements.txt   # Package and dependency management
README.md                          # Project overview and usage instructions
