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
    Compute fairness metrics by group (e.g., TPR, FPR, accuracy)

    Args:
        y_true (array-like): actual label
        y_pred (array-like): predicted label
        sensitive_features (array-like): sensitive features (e.g., sex, race)

    Returns:
        metric_frame (MetricFrame): group-wise metric table
        fairness_summary (dict): summary of maximum disparity between groups
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
