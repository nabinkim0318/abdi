"""Held-out model-evaluation figures. Matplotlib figures are returned, not shown."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


ROC_UNAVAILABLE_ONE_CLASS = (
    "ROC curve is unavailable because the held-out test split contains only "
    "one class."
)
ROC_UNAVAILABLE_NO_SCORES = (
    "ROC curve is unavailable because positive-class scores were not produced."
)


def confusion_matrix_from_predictions(y_test, y_pred):
    """
    Compute a confusion matrix on held-out labels.

    When the test split contains both binary classes, those labels define
    the matrix even if one class is absent from predictions.
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    test_labels = np.unique(y_test)
    if test_labels.size >= 2:
        labels = np.sort(test_labels)
    else:
        labels = np.sort(np.unique(np.concatenate([y_test, y_pred])))
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    return matrix, labels


def build_confusion_matrix_figure(y_test, y_pred) -> tuple[Figure, np.ndarray]:
    """Return a labeled confusion-matrix Figure and the underlying counts."""
    matrix, labels = confusion_matrix_from_predictions(y_test, y_pred)
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    tick_positions = np.arange(len(labels))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    fig.tight_layout()
    return fig, matrix


def build_roc_curve_figure(y_test, y_score):
    """
    Return ``(figure, None)`` for a held-out ROC curve, or ``(None, message)``
    when the curve is not defined.
    """
    y_test = np.asarray(y_test)
    if np.unique(y_test).size < 2:
        return None, ROC_UNAVAILABLE_ONE_CLASS
    if y_score is None:
        return None, ROC_UNAVAILABLE_NO_SCORES

    y_score = np.asarray(y_score)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)

    fig, ax = plt.subplots()
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC curve (AUC = {auc:.2f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, None
