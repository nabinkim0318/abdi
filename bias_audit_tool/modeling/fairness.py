import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, MetricFrame, selection_rate
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# === Constants ===
PLAIN_METRIC_NAMES = {
    "Accuracy": "Overall Correctness",
    "Precision": "Correctness of Positive Predictions", 
    "Recall": "Coverage of Actual Positives",
    "F1": "Balance between Precision & Recall",
    "Selection Rate": "Group Selection Rate",
}

TNBC_RACE_BENCHMARK = {
    'Black': 0.36,
    'White': 0.19,
    'Hispanic': 0.16,
    'AIAN': 0.16,
    'Asian': 0.13
}

# === Validation ===
def validate_inputs(df, demographic_col, benchmark_distribution):
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be None or empty")
    if demographic_col not in df.columns:
        raise ValueError(f"Column '{demographic_col}' not found in DataFrame")
    if not isinstance(benchmark_distribution, dict):
        raise ValueError("benchmark_distribution must be a dictionary")
    total = sum(benchmark_distribution.values())
    if not 0.95 <= total <= 1.05:
        warnings.warn(f"Benchmark distribution sums to {total:.3f}, not 1.0")

# === Disparity Analysis ===
def compute_input_fairness(
    df: pd.DataFrame,
    demographic_col: str,
    benchmark_distribution: dict = None,
    threshold_low: float = 0.8,
    threshold_high: float = 1.25,
    sort_by: str = "Observed_%",
    justifiability_fn=None,
) -> pd.DataFrame:
    if benchmark_distribution is None:
        benchmark_distribution = TNBC_RACE_BENCHMARK
        print("Using default TNBC race benchmark distribution")

    validate_inputs(df, demographic_col, benchmark_distribution)

    observed_counts = df[demographic_col].value_counts(dropna=False)
    total = len(df)
    observed_percent = observed_counts / total

    result_df = pd.DataFrame({
        "Observed_Count": observed_counts,
        "Observed_%": observed_percent,
    })

    result_df["Expected_%"] = result_df.index.map(benchmark_distribution)
    result_df["Benchmark_Valid"] = ~result_df["Expected_%"].isnull()
    result_df["Expected_%"] = result_df["Expected_%"].fillna(0.0001)

    result_df["Disparity_Ratio"] = result_df["Observed_%"] / result_df["Expected_%"]
    result_df["Magnitude_Flag"] = result_df["Disparity_Ratio"].apply(
        lambda x: "Not Fair" if x < threshold_low or x > threshold_high else "Fair"
    )
    result_df["Absolute_Difference"] = abs(result_df["Observed_%"] - result_df["Expected_%"])

    if justifiability_fn:
        result_df["Justifiable?"] = result_df.apply(justifiability_fn, axis=1)
    else:
        result_df["Justifiable?"] = "Unknown"

    result_df["Framework_Flag"] = result_df.apply(
        lambda row: "⚠️ Check" if (
            not row["Benchmark_Valid"] or
            row["Magnitude_Flag"] == "Not Fair" or
            row["Justifiable?"] != "Yes"
        ) else "✅ OK", axis=1
    )

    obs_dist = result_df["Observed_%"].values
    exp_dist = result_df["Expected_%"].values
    result_df.attrs["KL_Divergence"] = compare_distributions(obs_dist, exp_dist, "kl")
    result_df.attrs["Wasserstein_Distance"] = compare_distributions(obs_dist, exp_dist, "wasserstein")
    result_df.attrs["Total_Variation"] = 0.5 * np.sum(np.abs(obs_dist - exp_dist))

    result_df = result_df.reset_index().rename(columns={"index": "Group"})
    result_df = result_df.sort_values(sort_by, ascending=False)
    return result_df

def compare_distributions(p, q, method="kl"):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    if method == "kl":
        epsilon = 1e-8
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
    p_norm = p / p.sum()
    q_norm = q / q.sum()
    if method == "kl":
        return entropy(p_norm, q_norm)
    elif method == "wasserstein":
        return wasserstein_distance(p_norm, q_norm)
    else:
        raise ValueError("method must be 'kl' or 'wasserstein'")

# === Plotting ===
def plot_input_fairness(fairness_result, top_n=20, figsize=(12, 8)):
    if fairness_result is None or fairness_result.empty:
        print("No data to plot")
        return None

    plot_df = fairness_result.sort_values("Disparity_Ratio", ascending=False).head(top_n)
    colors = plot_df["Framework_Flag"].map({
        "✅ OK": "#2E8B57",
        "⚠️ Check": "#DC143C"
    })

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=plot_df,
        y="Group",
        x="Disparity_Ratio",
        palette=colors,
        ax=ax,
    )

    ax.axvline(1.0, color="black", linestyle="--", linewidth=2, label="Perfect Parity")
    ax.axvline(0.8, color="orange", linestyle=":", label="Lower Bound (0.8)")
    ax.axvline(1.25, color="orange", linestyle=":", label="Upper Bound (1.25)")

    for i, (idx, row) in enumerate(plot_df.iterrows()):
        if row["Disparity_Ratio"] > 2.0 or row["Disparity_Ratio"] < 0.5:
            ax.annotate(f'{row["Disparity_Ratio"]:.2f}',
                        xy=(row["Disparity_Ratio"], i),
                        xytext=(5, 0), textcoords='offset points',
                        va='center', fontsize=9)

    ax.set_title("Disparity Ratios by Demographic Group", fontsize=14)
    ax.set_xlabel("Disparity Ratio (Observed / Expected)", fontsize=12)
    ax.set_ylabel("Demographic Group")
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig
