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


def select_model(X, y):
    if y.nunique() == 2:
        return LogisticRegression(max_iter=1000)
    else:
        return RandomForestClassifier()


def run_basic_modeling(X, y, show_plots=True):
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
