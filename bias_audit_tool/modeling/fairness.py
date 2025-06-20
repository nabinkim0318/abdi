# utils/fairness.py
import numpy as np
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def compute_fairness_metrics(y_true, y_pred, sensitive_features):
    """
    Compute group-wise performance and fairness metrics for model predictions.

    This function calculates standard classification metrics (accuracy, precision,
    recall, F1) and fairness-specific metrics (selection rate, demographic
    parity difference,
    and equalized odds difference) across groups defined by sensitive features.

    Args:
        y_true (array-like): Ground truth (actual) labels.
        y_pred (array-like): Predicted labels from the classifier.
        sensitive_features (array-like): Sensitive attribute(s) used for fairness
        grouping (e.g., gender, race). Must be 1D and aligned with `y_true`.

    Returns:
        MetricFrame: A Fairlearn MetricFrame object containing
                     metric values per group.
        dict: A dictionary summarizing group disparities for each metric, including:
            - "<Metric> disparity": Maximum absolute difference across groups
            - "Demographic Parity Difference": Difference in selection
               rate between groups
            - "Equalized Odds Difference": Difference in TPR/FPR across groups

    Raises:
        ValueError: If any of the inputs are misaligned or invalid.
        Exception: Captures errors from Fairlearn fairness metrics and logs
        as strings in summary.

    Example:
        >>> compute_fairness_metrics([1, 0, 1], [1, 1, 0], ["M", "F", "F"])
    """

    # 1. define metrics
    metrics = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1": f1_score,
        "Selection Rate": selection_rate,
    }

    # 2. create MetricFrame
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    # 3. calculate maximum disparity between groups
    disparity_summary = {
        f"{metric} disparity": np.abs(
            metric_frame.by_group[metric].max() - metric_frame.by_group[metric].min()
        )
        for metric in metric_frame.by_group.columns
    }

    # 4. calculate fairness metrics
    try:
        dp_diff = demographic_parity_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )
        eo_diff = equalized_odds_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        disparity_summary["Demographic Parity Difference"] = dp_diff
        disparity_summary["Equalized Odds Difference"] = eo_diff

    except Exception as e:
        disparity_summary["Demographic Parity Difference"] = f"Error: {e}"
        disparity_summary["Equalized Odds Difference"] = f"Error: {e}"

    return metric_frame, disparity_summary
