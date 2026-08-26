# utils/fairness.py
import itertools
import json
import warnings

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

# Criterion-based labels for representation ratios vs a user-supplied
# benchmark. These are not fairness verdicts.
THRESHOLD_STATUS_COL = "Within Threshold?"
WITHIN_THRESHOLD = "Yes"
OUTSIDE_THRESHOLD = "No"
NO_BENCHMARK_AVAILABLE = "No benchmark available"

NO_BENCHMARK_SELECTED_MESSAGE = (
    "No benchmark selected. Provide an expected distribution to compute "
    "benchmark-relative representation disparities."
)
ALL_GROUPS_WITHIN_THRESHOLD_MESSAGE = (
    "All benchmarked groups fall within the selected disparity-ratio " "threshold."
)
FAIRNESS_METRIC_CAVEAT = (
    "Different fairness metrics can disagree, especially when outcome "
    "base rates differ by group. No single metric proves a model is fair."
)
CLAIM_SAFETY_CAVEAT = (
    "This is an exploratory diagnostic based on the selected metric and "
    "benchmark. It does not establish that a model or dataset is fair, "
    "unbiased, non-discriminatory, or legally compliant."
)


def interpret_fairness_metrics(kl, wasserstein, tv):
    """
    Interpret KL Divergence, Wasserstein Distance, and Total Variation
    into severity labels for distribution-alignment diagnostics.
    """

    def label_kl(val):
        if val < 0.05:
            return "✅ Excellent alignment"
        elif val < 0.15:
            return "🟢 Good alignment"
        elif val < 0.50:
            return "⚠️ Moderate deviation"
        else:
            return "🚨 Severe deviation"

    def label_wass(val):
        if val < 0.05:
            return "✅ Excellent alignment"
        elif val < 0.15:
            return "🟢 Mild deviation"
        elif val < 0.30:
            return "⚠️ Moderate deviation"
        else:
            return "🚨 Severe deviation"

    def label_tv(val):
        if val < 0.10:
            return "✅ Excellent alignment"
        elif val < 0.25:
            return "🟢 Mild deviation"
        elif val < 0.40:
            return "⚠️ Moderate deviation"
        else:
            return "🚨 Severe deviation"

    return {
        "KL Divergence": label_kl(kl),
        "Wasserstein Distance": label_wass(wasserstein),
        "Total Variation": label_tv(tv),
    }


def is_valid_benchmark(benchmark_distribution) -> bool:
    """True when a non-empty expected-distribution dict was supplied."""
    return (
        isinstance(benchmark_distribution, dict) and len(benchmark_distribution) > 0
    )


def parse_user_benchmark(benchmark_json):
    """
    Parse a user-supplied expected-distribution JSON string.

    Empty input is a missing benchmark, not a silent fallback. Invalid JSON
    is also treated as missing (no fabricated substitute distribution).

    Returns:
        (benchmark_dict_or_None, status) where status is "ok", "missing",
        or "invalid".
    """
    if benchmark_json is None:
        return None, "missing"
    text = str(benchmark_json).strip()
    if not text:
        return None, "missing"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, "invalid"
    if not is_valid_benchmark(parsed):
        return None, "missing"
    return parsed, "ok"


def validate_inputs(df, demographic_col, benchmark_distribution):
    """Validate inputs for fairness analysis."""
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be None or empty")

    if demographic_col not in df.columns:
        raise ValueError(f"Column '{demographic_col}' not found in DataFrame")

    if benchmark_distribution is None:
        return

    if not isinstance(benchmark_distribution, dict):
        raise ValueError("benchmark_distribution must be a dictionary")

    if not benchmark_distribution:
        return

    # Check if benchmark sums to ~1.0
    total_benchmark = sum(benchmark_distribution.values())
    if not 0.95 <= total_benchmark <= 1.05:
        warnings.warn(
            f"Benchmark distribution sums to {total_benchmark:.3f}, not 1.0",
            stacklevel=2,
        )


def compute_standardized_difference(df, group_col, value_col):
    """
    Compute standardized differences between all group pairs for a
    continuous variable.

    Cohen's d interpretation:
    - Small effect: 0.2
    - Medium effect: 0.5
    - Large effect: 0.8+
    """
    validate_inputs(df, group_col, {})  # Basic validation

    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for comparison")

    results = []

    for g1, g2 in itertools.combinations(groups, 2):
        x1 = df[df[group_col] == g1][value_col].dropna()
        x2 = df[df[group_col] == g2][value_col].dropna()

        if len(x1) == 0 or len(x2) == 0:
            continue

        mean1, mean2 = x1.mean(), x2.mean()
        std1, std2 = x1.std(), x2.std()

        # Use Cohen's d formula (pooled standard deviation)
        pooled_std = np.sqrt(
            ((len(x1) - 1) * std1**2 + (len(x2) - 1) * std2**2)
            / (len(x1) + len(x2) - 2)
        )
        std_diff = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan

        results.append(
            {
                "Group_1": g1,
                "Group_2": g2,
                "Standardized_Diff": std_diff,
                "Effect_Size": _interpret_effect_size(abs(std_diff)),
                "Mean_1": mean1,
                "Mean_2": mean2,
                "N_1": len(x1),
                "N_2": len(x2),
                "Pooled_SD": pooled_std,
            }
        )

    return pd.DataFrame(results).sort_values(
        "Standardized_Diff", key=abs, ascending=False
    )


def _interpret_effect_size(cohen_d):
    """Interpret Cohen's d effect size."""
    if pd.isna(cohen_d):
        return "Unknown"
    elif cohen_d < 0.2:
        return "Negligible"
    elif cohen_d < 0.5:
        return "Small"
    elif cohen_d < 0.8:
        return "Medium"
    else:
        return "Large"


def _threshold_status(ratio, threshold_low, threshold_high):
    """Map a disparity ratio to a criterion label, not a fairness verdict."""
    if pd.isna(ratio):
        return NO_BENCHMARK_AVAILABLE
    if threshold_low <= ratio <= threshold_high:
        return WITHIN_THRESHOLD
    return OUTSIDE_THRESHOLD


def compare_distributions(p, q, method="kl"):
    """
    Compare two distributions using KL divergence or Wasserstein distance.

    Returns:
    - divergence score (lower = more similar)
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # Handle zeros for KL divergence
    if method == "kl":
        epsilon = 1e-8
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)

    # Normalize to valid probability distributions
    p_norm = p / p.sum()
    q_norm = q / q.sum()

    if method == "kl":
        return entropy(p_norm, q_norm)
    elif method == "wasserstein":
        return wasserstein_distance(p_norm, q_norm)
    else:
        raise ValueError("method must be 'kl' or 'wasserstein'")


def compute_input_fairness(
    df: pd.DataFrame,
    demographic_col: str,
    benchmark_distribution: dict = None,
    threshold_low: float = 0.8,
    threshold_high: float = 1.25,
    sort_by: str = "Observed_%",
) -> pd.DataFrame:
    """
    Compare observed group shares to an explicit expected distribution.

    No domain-specific benchmark (including any TNBC incidence table) is
    applied automatically. If `benchmark_distribution` is missing or empty,
    observed counts are still returned, but benchmark-relative ratios are
    not computed.

    Returns:
        DataFrame with observed shares and, when a benchmark is supplied,
        disparity ratios plus a `Within Threshold?` criterion column.
    """
    validate_inputs(df, demographic_col, benchmark_distribution)

    observed_counts = df[demographic_col].value_counts(dropna=False)
    total = len(df)
    observed_percent = observed_counts / total

    result_df = pd.DataFrame(
        {
            "Observed_Count": observed_counts,
            "Observed_%": observed_percent,
        }
    )

    has_benchmark = is_valid_benchmark(benchmark_distribution)
    if not has_benchmark:
        result_df["Expected_%"] = np.nan
        result_df["Disparity_Ratio"] = np.nan
        result_df["Absolute_Difference"] = np.nan
        result_df[THRESHOLD_STATUS_COL] = NO_BENCHMARK_AVAILABLE
        result_df["Deviation_Type"] = NO_BENCHMARK_AVAILABLE
        result_df.attrs["benchmark_status"] = "missing"
        result_df.attrs["benchmark_message"] = NO_BENCHMARK_SELECTED_MESSAGE
        result_df.attrs["KL_Divergence"] = "N/A"
        result_df.attrs["Wasserstein_Distance"] = "N/A"
        result_df.attrs["Total_Variation"] = "N/A"
        result_df = result_df.sort_values(sort_by, ascending=False).reset_index()
        result_df = result_df.rename(columns={"index": "Group"})
        return result_df

    result_df.attrs["benchmark_status"] = "ok"
    result_df.attrs["benchmark_message"] = None

    # Groups absent from the benchmark stay NaN; no fabricated value.
    result_df["Expected_%"] = result_df.index.map(benchmark_distribution)
    missing_groups = result_df["Expected_%"].isnull().sum()
    if missing_groups > 0:
        print(
            f"Warning: {missing_groups} group(s) missing from benchmark. "
            "These groups will be reported as 'No benchmark available' "
            "instead of being assigned a fabricated expected value."
        )

    # Groups without a benchmark yield NaN (Observed_% / NaN).
    result_df["Disparity_Ratio"] = result_df["Observed_%"] / result_df["Expected_%"]
    result_df["Absolute_Difference"] = abs(
        result_df["Observed_%"] - result_df["Expected_%"]
    )

    result_df[THRESHOLD_STATUS_COL] = result_df["Disparity_Ratio"].apply(
        lambda x: _threshold_status(x, threshold_low, threshold_high)
    )

    result_df["Deviation_Type"] = result_df["Disparity_Ratio"].apply(
        lambda x: (
            NO_BENCHMARK_AVAILABLE
            if pd.isna(x)
            else (
                "Under-represented"
                if x < threshold_low
                else "Over-represented" if x > threshold_high else "Within bounds"
            )
        )
    )

    # Distances use benchmarked groups only.
    benchmarked = result_df.dropna(subset=["Expected_%"])
    obs_dist = benchmarked["Observed_%"].values
    exp_dist = benchmarked["Expected_%"].values

    if len(exp_dist) > 0:
        result_df.attrs["KL_Divergence"] = compare_distributions(
            obs_dist, exp_dist, "kl"
        )
        result_df.attrs["Wasserstein_Distance"] = compare_distributions(
            obs_dist, exp_dist, "wasserstein"
        )
        result_df.attrs["Total_Variation"] = 0.5 * np.sum(
            np.abs(obs_dist - exp_dist)
        )
    else:
        result_df.attrs["KL_Divergence"] = "N/A"
        result_df.attrs["Wasserstein_Distance"] = "N/A"
        result_df.attrs["Total_Variation"] = "N/A"

    result_df = result_df.sort_values(sort_by, ascending=False).reset_index()
    result_df = result_df.rename(columns={"index": "Group"})

    return result_df


def plot_input_fairness(fairness_result, top_n=20, figsize=(12, 8)):
    """Enhanced plotting with better styling and annotations."""
    if fairness_result is None or fairness_result.empty:
        print("No data to plot")
        return None

    try:
        plot_df = fairness_result.sort_values(
            "Disparity_Ratio", ascending=False
        ).head(top_n)

        colors = plot_df[THRESHOLD_STATUS_COL].map(
            {WITHIN_THRESHOLD: "#2E8B57", OUTSIDE_THRESHOLD: "#DC143C"}
        )

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(
            data=plot_df,
            y="Group",
            x="Disparity_Ratio",
            palette=colors,
            ax=ax,
        )

        # Add reference lines
        ax.axvline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Perfect Parity (1.0)",
        )
        ax.axvline(
            0.8, color="orange", linestyle=":", alpha=0.7, label="Lower Bound (0.8)"
        )
        ax.axvline(
            1.25,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label="Upper Bound (1.25)",
        )

        # Annotations for extreme values
        for i, (_, row) in enumerate(plot_df.iterrows()):
            if row["Disparity_Ratio"] > 2.0 or row["Disparity_Ratio"] < 0.5:
                ax.annotate(
                    f'{row["Disparity_Ratio"]:.2f}',
                    xy=(row["Disparity_Ratio"], i),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    weight="bold",
                )

        ax.set_title(
            "📊 Fairness Audit: Disparity Ratios by Demographic Group",
            fontsize=14,
            weight="bold",
            pad=20,
        )
        ax.set_xlabel("Disparity Ratio (Observed ÷ Expected)", fontsize=12)
        ax.set_ylabel("Demographic Group", fontsize=12)
        ax.legend(loc="best")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"[ERROR] plot_input_fairness failed: {e}")
        return None


def render_fairness_caveats():
    """Standing claim-safety notes for live fairness diagnostics."""
    st.info(FAIRNESS_METRIC_CAVEAT)
    st.caption(CLAIM_SAFETY_CAVEAT)


def display_fairness_summary(result_df: pd.DataFrame, top_n: int = 5):
    """Summary of representation diagnostics against a selected benchmark."""

    if result_df is None or result_df.empty:
        st.error("No fairness data to display")
        return

    render_fairness_caveats()

    if result_df.attrs.get("benchmark_status") != "ok":
        st.info(
            result_df.attrs.get("benchmark_message", NO_BENCHMARK_SELECTED_MESSAGE)
        )
        with st.expander("📋 Observed group shares"):
            st.dataframe(result_df, use_container_width=True)
        return

    total_groups = len(result_df)
    within_groups = (result_df[THRESHOLD_STATUS_COL] == WITHIN_THRESHOLD).sum()
    outside_groups = (result_df[THRESHOLD_STATUS_COL] == OUTSIDE_THRESHOLD).sum()
    unbenchmarked_groups = (
        result_df[THRESHOLD_STATUS_COL] == NO_BENCHMARK_AVAILABLE
    ).sum()

    kl_div = result_df.attrs.get("KL_Divergence", "N/A")
    wass_dist = result_df.attrs.get("Wasserstein_Distance", "N/A")
    total_var = result_df.attrs.get("Total_Variation", "N/A")
    interpretations = interpret_fairness_metrics(kl_div, wass_dist, total_var)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Groups", total_groups)
    with col2:
        st.metric("Within Threshold", within_groups)
    with col3:
        st.metric("Outside Threshold", outside_groups)
    with col4:
        st.metric("No Benchmark", unbenchmarked_groups)

    st.markdown("### 📐 Distribution Distance Metrics")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        st.metric(
            "KL Divergence",
            f"{kl_div:.4f}" if isinstance(kl_div, float) else kl_div,
        )
        st.caption(interpretations["KL Divergence"])
    with metrics_col2:
        st.metric(
            "Wasserstein Distance",
            f"{wass_dist:.4f}" if isinstance(wass_dist, float) else wass_dist,
        )
        st.caption(interpretations["Wasserstein Distance"])
    with metrics_col3:
        st.metric(
            "Total Variation",
            f"{total_var:.4f}" if isinstance(total_var, float) else total_var,
        )
        st.caption(interpretations["Total Variation"])

    if outside_groups == 0 and within_groups > 0:
        st.info(ALL_GROUPS_WITHIN_THRESHOLD_MESSAGE)

    with st.expander("📋 Complete representation analysis"):
        formatted_df = result_df.style.format(
            {
                "Observed_%": "{:.2%}",
                "Expected_%": "{:.2%}",
                "Disparity_Ratio": "{:.2f}",
                "Absolute_Difference": "{:.3f}",
            }
        ).background_gradient(
            subset=["Disparity_Ratio"], cmap="RdYlGn", vmin=0.5, vmax=1.5
        )

        st.dataframe(formatted_df, use_container_width=True)


def _undefined_metric(reason, **details):
    """Marker for a fairness metric that could not be computed as a number."""
    return {"status": "undefined", "reason": reason, **details}


def compute_output_fairness(y_true, y_pred, sensitive_features):
    """
    Enhanced output fairness with better error handling and interpretability.
    """
    # Input validation
    y_true, y_pred, sensitive_features = map(
        np.array, [y_true, y_pred, sensitive_features]
    )

    if not (len(y_true) == len(y_pred) == len(sensitive_features)):
        raise ValueError("All input arrays must have the same length")

    if sensitive_features.ndim != 1:
        raise ValueError("sensitive_features must be 1-dimensional")

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

    try:
        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to compute MetricFrame: {e}") from e

    # Calculate disparities with better interpretation
    disparity_summary = {}
    for metric in metric_frame.by_group.columns:
        values = metric_frame.by_group[metric]
        max_val, min_val = values.max(), values.min()
        disparity = abs(max_val - min_val)
        disparity_summary[f"{metric} disparity"] = disparity

        # Add ratio-based disparity for better interpretation. A zero minimum
        # makes the ratio mathematically undefined (division by zero), so we
        # surface that explicitly instead of a bare, unexplained `inf`.
        if min_val > 0:
            disparity_summary[f"{metric} ratio"] = max_val / min_val
        else:
            disparity_summary[f"{metric} ratio"] = _undefined_metric(
                "Ratio is undefined: the minimum group value is 0 "
                "(division by zero).",
                min_value=min_val,
                max_value=max_val,
            )

    # Fairlearn-specific metrics. `sensitive_features` must be passed as a
    # keyword argument; this Fairlearn version makes it keyword-only.
    try:
        dp_diff = demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        disparity_summary["Demographic Parity Difference"] = dp_diff
    except Exception as e:
        warnings.warn(
            f"Demographic Parity Difference could not be computed: {e}",
            stacklevel=2,
        )
        disparity_summary["Demographic Parity Difference"] = _undefined_metric(
            str(e)
        )

    try:
        eo_diff = equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        disparity_summary["Equalized Odds Difference"] = eo_diff
    except Exception as e:
        warnings.warn(
            f"Equalized Odds Difference could not be computed: {e}",
            stacklevel=2,
        )
        disparity_summary["Equalized Odds Difference"] = _undefined_metric(str(e))

    # Convert to plain language and sort by severity
    disparity_plain = {
        PLAIN_METRIC_NAMES.get(
            k.replace(" disparity", "").replace(" ratio", ""), k
        ): v
        for k, v in disparity_summary.items()
    }

    # Sort by magnitude (excluding error strings)
    disparity_sorted = dict(
        sorted(
            disparity_plain.items(),
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else -1,
            reverse=True,
        )
    )

    return metric_frame, disparity_sorted


# Convenience function for complete fairness audit
def run_complete_fairness_audit(
    df,
    demographic_col,
    y_true=None,
    y_pred=None,
    benchmark_dist=None,
    show_plots=True,
):
    """
    Run a complete fairness audit including both input and output fairness.

    Returns:
    - dict with input_fairness, output_fairness (if provided), and summary
    """
    results = {}

    # Input fairness
    print("🔍 Running input fairness analysis...")
    input_fairness = compute_input_fairness(df, demographic_col, benchmark_dist)
    results["input_fairness"] = input_fairness

    if show_plots:
        fig = plot_input_fairness(input_fairness)
        if fig:
            results["input_plot"] = fig

    # Output fairness (if predictions provided)
    if y_true is not None and y_pred is not None:
        print("🎯 Running output fairness analysis...")
        sensitive_features = df[demographic_col].values
        metric_frame, disparities = compute_output_fairness(
            y_true, y_pred, sensitive_features
        )
        results["output_fairness"] = {
            "metrics": metric_frame,
            "disparities": disparities,
        }

    outside_threshold_groups = (
        input_fairness[THRESHOLD_STATUS_COL] == OUTSIDE_THRESHOLD
    ).sum()
    total_groups = len(input_fairness)

    results["summary"] = {
        "total_groups": total_groups,
        "outside_threshold_groups": outside_threshold_groups,
        "within_threshold_rate": (
            (total_groups - outside_threshold_groups) / total_groups
            if total_groups
            else None
        ),
        "kl_divergence": input_fairness.attrs.get("KL_Divergence"),
        "wasserstein_distance": input_fairness.attrs.get("Wasserstein_Distance"),
        "benchmark_status": input_fairness.attrs.get("benchmark_status"),
    }

    print(
        f"Audit complete. {outside_threshold_groups}/{total_groups} groups "
        "outside the selected disparity-ratio threshold."
    )
    return results
