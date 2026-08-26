from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score

from bias_audit_tool.preprocessing.modeling_pipeline import run_modeling_pipeline
from bias_audit_tool.visualization.evaluation_plots import (
    ROC_UNAVAILABLE_ONE_CLASS,
)
from bias_audit_tool.visualization.evaluation_plots import (
    build_confusion_matrix_figure,
)
from bias_audit_tool.visualization.evaluation_plots import build_roc_curve_figure
from bias_audit_tool.visualization.evaluation_plots import (
    confusion_matrix_from_predictions,
)


ROOT = Path(__file__).resolve().parents[1]


def test_confusion_matrix_counts_match_known_predictions():
    y_test = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    matrix, labels = confusion_matrix_from_predictions(y_test, y_pred)

    assert list(labels) == [0, 1]
    # Rows = actual, columns = predicted.
    assert matrix[0, 0] == 2  # true 0 predicted 0
    assert matrix[0, 1] == 1  # true 0 predicted 1
    assert matrix[1, 0] == 1  # true 1 predicted 0
    assert matrix[1, 1] == 2  # true 1 predicted 1

    fig, plotted = build_confusion_matrix_figure(y_test, y_pred)
    assert isinstance(fig, Figure)
    np.testing.assert_array_equal(plotted, matrix)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Predicted"
    assert ax.get_ylabel() == "Actual"
    plt.close(fig)


def test_confusion_matrix_keeps_both_test_labels_when_one_class_unpredicted():
    y_test = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    matrix, labels = confusion_matrix_from_predictions(y_test, y_pred)

    assert list(labels) == [0, 1]
    assert matrix.shape == (2, 2)
    assert matrix[1, 1] == 0
    assert matrix[1, 0] == 2


def test_roc_curve_uses_held_out_scores_and_known_auc():
    y_test = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    expected_auc = roc_auc_score(y_test, y_score)
    assert expected_auc == pytest.approx(1.0)

    fig, message = build_roc_curve_figure(y_test, y_score)
    assert message is None
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate"
    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("AUC = 1.00" in label for label in legend_texts)
    plt.close(fig)


def test_roc_curve_is_unavailable_for_one_class_y_test():
    y_test = np.array([1, 1, 1, 1])
    y_score = np.array([0.2, 0.4, 0.6, 0.8])
    fig, message = build_roc_curve_figure(y_test, y_score)
    assert fig is None
    assert message == ROC_UNAVAILABLE_ONE_CLASS


def test_pipeline_held_out_outputs_feed_evaluation_plots_without_a_second_split():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "age": rng.normal(50, 10, n),
            "gender": np.where(np.arange(n) < n // 2, "F", "M"),
            "outcome": np.where(np.arange(n) % 2 == 0, 0, 1),
        }
    )
    recommendations = {"age": "MinMaxScaler", "gender": "OneHotEncoder"}
    result = run_modeling_pipeline(
        raw_df=df,
        df_proc=df,
        target_col="outcome",
        sensitive_col="gender",
        include_sensitive_in_features=False,
        recommendations=recommendations,
        random_state=42,
    )

    assert result.y_pred is not None
    assert result.y_prob is not None
    assert len(result.y_test) == len(result.y_pred) == len(result.y_prob)
    pd.testing.assert_index_equal(result.y_test.index, result.sensitive_test.index)

    cm_fig, matrix = build_confusion_matrix_figure(result.y_test, result.y_pred)
    assert matrix.sum() == len(result.y_test)
    roc_fig, message = build_roc_curve_figure(result.y_test, result.y_prob)
    assert message is None
    assert isinstance(cm_fig, Figure)
    assert isinstance(roc_fig, Figure)
    plt.close(cm_fig)
    plt.close(roc_fig)


def test_fit_and_evaluate_model_has_no_show_plots_or_plt_show():
    source = (ROOT / "bias_audit_tool" / "modeling" / "model_selector.py").read_text(
        encoding="utf-8"
    )
    assert "show_plots" not in source
    assert "plt.show" not in source
    assert "def run_basic_modeling" not in source


def test_production_code_does_not_call_plt_show():
    roots = [ROOT / "app.py", ROOT / "bias_audit_tool"]
    for path in roots:
        files = [path] if path.is_file() else path.rglob("*.py")
        for py_file in files:
            text = py_file.read_text(encoding="utf-8")
            assert "plt.show(" not in text, py_file
