from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bias_audit_tool.modeling.fairness import compute_input_fairness
from bias_audit_tool.modeling.fairness import GROUP_COL
from bias_audit_tool.visualization.visualization import plot_distribution_comparison


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_does_not_declare_unused_shap_or_reportlab():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "shap" not in text
    assert "reportlab" not in text


def test_live_app_does_not_import_removed_report_or_shap_apis():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    helpers = (ROOT / "bias_audit_tool" / "utils" / "ui_helpers.py").read_text(
        encoding="utf-8"
    )
    ui_blocks = (
        ROOT / "bias_audit_tool" / "visualization" / "ui_blocks.py"
    ).read_text(encoding="utf-8")
    combined = "\n".join([app_source, helpers, ui_blocks])

    assert "import shap" not in combined
    assert "from shap" not in combined
    assert "generate_pdf_report" not in combined
    assert "report_generator" not in combined
    assert "bias_audit_tool.report" not in combined


def test_readme_does_not_advertise_removed_product_claims():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "shap" not in readme
    assert "html report" not in readme
    assert "pdf report" not in readme
    assert "automated reports" not in readme
    assert "test_failure.py" not in readme
    assert "confusion matrix" in readme
    assert "roc curve" in readme
    assert "does not remove proxy variables" in readme


def test_live_app_does_not_show_unwired_encoding_checkbox():
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    helpers = (ROOT / "bias_audit_tool" / "utils" / "ui_helpers.py").read_text(
        encoding="utf-8"
    )
    combined = "\n".join([app_source, helpers])
    assert "get_user_preprocessing_options" not in combined
    assert "Encode categorical columns" not in combined
    assert "Apply Scaling to numeric columns" not in combined


def test_plot_distribution_comparison_uses_stable_group_column():
    df = pd.DataFrame({"gender": ["F"] * 40 + ["M"] * 60})
    result = compute_input_fairness(
        df,
        demographic_col="gender",
        benchmark_distribution={"F": 0.5, "M": 0.5},
    )
    assert GROUP_COL in result.columns
    assert "gender" not in result.columns

    fig = plot_distribution_comparison(result, top_n=20)
    assert fig is not None
    labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    assert set(labels) == {"F", "M"}
    plt.close(fig)
