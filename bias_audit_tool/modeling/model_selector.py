import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


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


def fit_and_evaluate_model(X_train, y_train, X_test, y_test):
    """
    Select a model, fit it on already-split/already-preprocessed training
    data, and evaluate it on the held-out test data.
    """
    model = select_model(X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    return {
        "model": model,
        "report": report_df,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob if y_prob is not None else None,
        "feature_importance": compute_feature_importance(model, X_test, y_test),
    }


def compute_feature_importance(model, X_test, y_test):
    """
    Compute permutation importance for model features.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets

    Returns:
        pd.DataFrame: Feature importance scores
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result["importances_mean"],
            "importance_std": result["importances_std"],
        }
    ).sort_values("importance_mean", ascending=False)

    return importance_df
