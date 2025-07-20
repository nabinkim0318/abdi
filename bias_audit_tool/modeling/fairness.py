# utils/fairness.py
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

PLAIN_METRIC_NAMES = {
    "Accuracy": "Overall Correctness",
    "Precision": "Correctness of Positive Predictions",
    "Recall": "Coverage of Actual Positives",
    "F1": "Balance between Precision & Recall",
    "Selection Rate": "Group Selection Rate",
}


def compute_standardized_difference(df, group_col, value_col):
    """
    Compute standardized differences between all group
    pairs for a continuous variable.

    Parameters:
    - df: pandas DataFrame
    - group_col: categorical column to group by (e.g., 'Race')
    - value_col: numeric column to compare (e.g., 'Age')

    Returns:
    - result_df: DataFrame of pairwise standardized differences
    """
    groups = df[group_col].dropna().unique()
    results = []

    for g1, g2 in itertools.combinations(groups, 2):
        x1 = df[df[group_col] == g1][value_col].dropna()
        x2 = df[df[group_col] == g2][value_col].dropna()

        mean1, mean2 = x1.mean(), x2.mean()
        std1, std2 = x1.std(), x2.std()

        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        std_diff = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan

        results.append(
            {
                "Group_1": g1,
                "Group_2": g2,
                "Standardized_Diff": std_diff,
                "Mean_1": mean1,
                "Mean_2": mean2,
                "Pooled_SD": pooled_std,
            }
        )

    return pd.DataFrame(results)


def compare_distributions(p, q, method="kl"):
    """
    Compare two distributions using KL divergence or Wasserstein distance.

    Parameters:
    - p: array-like or pd.Series (distribution 1)
    - q: array-like or pd.Series (distribution 2)
    - method: 'kl' or 'wasserstein'

    Returns:
    - divergence score
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # Normalize to make sure it's a valid distribution
    p /= p.sum()
    q /= q.sum()

    if method == "kl":
        return entropy(p, q)  # KL(P || Q)
    elif method == "wasserstein":
        return wasserstein_distance(p, q)
    else:
        raise ValueError("method must be 'kl' or 'wasserstein'")


def compute_input_fairness(
    df: pd.DataFrame,
    demographic_col: str,
    benchmark_distribution: dict,
    threshold_low: float = 0.8,
    threshold_high: float = 1.25,
    sort_by="Observed_%",
) -> pd.DataFrame:
    """
    Detects unfair representation in a categorical
    demographic column by comparing
    observed proportions to expected (benchmark) proportions.

    Parameters:
    - df: pandas DataFrame containing the data
    - demographic_col: column name containing the demographic
        categories (e.g., 'Race')
    - benchmark_distribution: dict with expected % values (e.g., {
            'Black': 0.36, 'White': 0.19})
    - threshold_low: lower bound of fairness threshold for disparity
        ratio (default = 0.8)
    - threshold_high: upper bound of fairness threshold for disparity
        ratio (default = 1.25)

    Returns:
    - result_df: DataFrame with observed counts, observed %, expected %,
    disparity ratio, and fairness label
    """
    # 1. Count each group
    observed_counts = df[demographic_col].value_counts(dropna=False)

    # 2. Compute proportion
    total = len(df)
    observed_percent = observed_counts / total

    # 3. Combine into result DataFrame
    result_df = pd.DataFrame(
        {"Observed_Count": observed_counts, "Observed_%": observed_percent}
    )

    # 4. Map benchmark distribution
    result_df["Expected_%"] = result_df.index.map(benchmark_distribution)
    if result_df["Expected_%"].isnull().any():
        print(
            "Some groups in data are missing from benchmark_distribution. "
            "Filling with 0.0001."
        )
        result_df["Expected_%"] = result_df["Expected_%"].fillna(0.0001)

    # 5. Compute disparity ratio
    result_df["Disparity_Ratio"] = result_df["Observed_%"] / result_df["Expected_%"]

    # 6. Assess fairness
    result_df["Fair?"] = result_df["Disparity_Ratio"].apply(
        lambda x: "Fair" if threshold_low <= x <= threshold_high else "Not Fair"
    )

    obs_dist = result_df["Observed_%"]
    exp_dist = result_df["Expected_%"]

    result_df.attrs["KL Divergence"] = compare_distributions(
        obs_dist, exp_dist, method="kl"
    )
    result_df.attrs["Wasserstein Distance"] = compare_distributions(
        obs_dist, exp_dist, method="wasserstein"
    )

    # 7. Sort by largest group
    return result_df.sort_values(sort_by, ascending=False)


def plot_input_fairness(result_df: pd.DataFrame):
    fig, ax = plt.subplots()
    sns.barplot(
        x=result_df.index,
        y="Disparity_Ratio",
        hue="Fair?",
        data=result_df,
        dodge=False,
        ax=ax,
    )
    ax.axhline(1, linestyle="--", color="black")
    ax.axhline(0.8, linestyle=":", color="gray")
    ax.axhline(1.25, linestyle=":", color="gray")
    ax.set_title("Disparity Ratio by Group")
    ax.set_ylabel("Disparity Ratio")
    ax.set_xlabel("Group")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def display_fairness_summary(result_df: pd.DataFrame, top_n: int = 5):
    """
    Displays a summary of fairness assessment results in Streamlit.

    Parameters:
    - result_df: output from compute_input_fairness()
    - top_n: number of Not Fair groups to highlight
    """
    total_groups = len(result_df)
    num_fair = (result_df["Fair?"] == "Fair").sum()
    num_not_fair = total_groups - num_fair

    kl = result_df.attrs.get("KL Divergence", None)
    wass = result_df.attrs.get("Wasserstein Distance", None)

    # Summary metrics
    st.markdown("### 📊 Fairness Summary")
    st.markdown(f"- **Total groups:** `{total_groups}`")
    st.markdown(f"- ✅ Fair groups: `{num_fair}`")
    st.markdown(f"- ❌ Not Fair groups: `{num_not_fair}`")

    if kl is not None and wass is not None:
        st.markdown("### 📐 Distribution Distance Metrics")
        st.markdown(f"- **KL Divergence:** `{kl:.4f}`")
        st.markdown(f"- **Wasserstein Distance:** `{wass:.4f}`")

    # Top N Not Fair groups
    not_fair_df = result_df[result_df["Fair?"] == "Not Fair"].copy()
    if not not_fair_df.empty:
        st.markdown(
            f"### 🔍 Top {min(top_n, len(not_fair_df))} Most Deviant "
            "'Not Fair' Groups"
        )
        most_deviant = not_fair_df.sort_values(
            by="Disparity_Ratio", key=lambda x: (x - 1).abs(), ascending=False
        ).head(top_n)
        st.dataframe(
            most_deviant.style.format(
                {
                    "Observed_%": "{:.2%}",
                    "Expected_%": "{:.2%}",
                    "Disparity_Ratio": "{:.2f}",
                }
            )
        )
    else:
        st.success("✅ All groups are within the fairness threshold!")

    # Full table
    with st.expander("📋 Full Fairness Table"):
        st.dataframe(
            result_df.style.format(
                {
                    "Observed_%": "{:.2%}",
                    "Expected_%": "{:.2%}",
                    "Disparity_Ratio": "{:.2f}",
                }
            )
        )


def compute_output_fairness(y_true, y_pred, sensitive_features):
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
    if len(y_true) != len(y_pred) or len(y_true) != len(sensitive_features):
        raise ValueError("Input arrays must have the same length.")
    if np.array(sensitive_features).ndim != 1:
        raise ValueError("sensitive_features must be 1-dimensional.")

    metrics = {
        "Accuracy": accuracy_score,
        "Precision": lambda y_true, y_pred: precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": lambda y_true, y_pred: recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
        "Selection Rate": selection_rate,
    }

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    disparity_summary = {
        f"{metric} disparity": np.abs(
            metric_frame.by_group[metric].max() - metric_frame.by_group[metric].min()
        )
        for metric in metric_frame.by_group.columns
    }

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

    # Plain-speak version
    disparity_summary_plain = {
        PLAIN_METRIC_NAMES.get(
            k.replace(" disparity", ""), k.replace(" disparity", "")
        ): v
        for k, v in disparity_summary.items()
    }

    # Sort disparities by magnitude
    disparity_summary_sorted = dict(
        sorted(
            disparity_summary_plain.items(),
            key=lambda x: -abs(x[1]) if isinstance(x[1], (int, float)) else 0,
        )
    )

    return metric_frame, disparity_summary_sorted
