import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def suggest_target_candidates(df, min_ratio=0.01, max_unique=10):
    """
    Suggest candidate target columns based on common medical-related keywords
    and column properties like cardinality and missing values.

    Args:
        df (pd.DataFrame): Input dataset.
        min_ratio (float): Minimum ratio of non-null values required.
        max_unique (int): Maximum number of unique values allowed.

    Returns:
        pd.DataFrame: List of column candidates with stats.
    """
    target_keywords = ["status", "diagnosis", "outcome", "recurrence", "death"]
    candidates = []

    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("int"):
            nunique = df[col].nunique()
            if nunique <= max_unique:
                lower_col = col.lower()
                if any(key in lower_col for key in target_keywords):
                    nonnull_ratio = df[col].notnull().mean()
                    if nonnull_ratio >= min_ratio:
                        candidates.append((col, nunique, nonnull_ratio))

    return pd.DataFrame(
        candidates, columns=["Column", "Unique Values", "Non-null Ratio"]
    )


"""
candidates = suggest_target_candidates(df)
if not candidates.empty:
    print("💡 Suggested target label candidates:")
    print(candidates)

# 예시: 가장 위의 타겟을 사용
target_col = candidates.iloc[0]["Column"]
features = df.drop(columns=[target_col])
target = df[target_col]

result = run_basic_modeling(features, target)
print(result["report"])
"""


def select_model(X, y):
    """
    Select a model based on the number of target classes.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.

    Returns:
        sklearn.base.BaseEstimator: A scikit-learn model instance.
    """
    if y.nunique() == 2:
        return LogisticRegression(max_iter=1000)
    else:
        return RandomForestClassifier()


def run_basic_modeling(X, y, show_plots=True):
    """
    Run a simple modeling pipeline including training, evaluation, and visualization.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        show_plots (bool): Whether to display confusion matrix and ROC curve.

    Returns:
        dict: Dictionary containing the model, evaluation report,
              predictions, and optional probability scores.
    """
    # Handle non-numeric targets
    if y.dtype == "object" or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Dummy encoding for categorical vars
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model selection & training
    model = select_model(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    if show_plots:
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        # ROC Curve (if binary)
        if y_prob is not None and len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = roc_auc_score(y_test, y_prob)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
            ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend(loc="lower right")
            plt.tight_layout()
            plt.show()

    return {
        "model": model,
        "report": report_df,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob if y_prob is not None else None,
    }


# Example usage (commented out):
# result = permutation_importance(model, X_test,
#  y_test, n_repeats=10, random_state=42)
# importance_df = pd.DataFrame({
#     'feature': X_test.columns,
#     'importance_mean': result.importances_mean,
#     'importance_std': result.importances_std
# })
