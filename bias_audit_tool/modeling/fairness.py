# utils/fairness.py
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

# Stable group-column name returned by compute_input_fairness, regardless
# of the original demographic column name.
GROUP_COL = "Group"

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

# Live model-fairness contract: y_true and y_pred are binary 0/1 with
# positive class 1. The modeling path LabelEncodes string targets to 0/1
# before this boundary. Fairlearn's demographic_parity_difference and
# equalized_odds_difference do not accept pos_label and use label 1, so
# compute_output_fairness / bootstrap do not take a custom positive_label.
# compute_group_support may still count an explicit positive_label as a
# pure utility; the live UI always uses 1.
DEFAULT_POSITIVE_LABEL = 1
ALLOWED_BINARY_FAIRNESS_LABELS = frozenset({0, 1})
BINARY_FAIRNESS_LABELS_ERROR = (
    "Model fairness metrics require binary 0/1 labels " "with positive class 1."
)

# Heuristic stability cutoffs for held-out group-support warnings.
# These are not statistical-validity, legal, or fairness criteria.
MIN_GROUP_N_WARNING = 30
MIN_CLASS_SUPPORT_WARNING = 10

SUPPORT_COL_N = "n"
SUPPORT_COL_POSITIVE_LABELS = "Positive Labels"
SUPPORT_COL_NEGATIVE_LABELS = "Negative Labels"
SUPPORT_COL_PREDICTED_POSITIVES = "Predicted Positives"
SUPPORT_COL_PREDICTED_NEGATIVES = "Predicted Negatives"
SUPPORT_COL_SELECTION_DENOMINATOR = "Selection Rate Denominator"
SUPPORT_TABLE_COLUMNS = [
    GROUP_COL,
    SUPPORT_COL_N,
    SUPPORT_COL_POSITIVE_LABELS,
    SUPPORT_COL_NEGATIVE_LABELS,
    SUPPORT_COL_PREDICTED_POSITIVES,
    SUPPORT_COL_PREDICTED_NEGATIVES,
    SUPPORT_COL_SELECTION_DENOMINATOR,
]

WARNING_SMALL_GROUP_N = "small_group_n"
WARNING_FEW_POSITIVE_LABELS = "few_positive_labels"
WARNING_FEW_NEGATIVE_LABELS = "few_negative_labels"
WARNING_ZERO_POSITIVE_LABELS = "zero_positive_labels"
WARNING_ZERO_NEGATIVE_LABELS = "zero_negative_labels"
WARNING_MISSING_SENSITIVE_EXCLUDED = "missing_sensitive_values_excluded"
WARNING_SPARSE_BOOTSTRAP = "sparse_support_bootstrap"
WARNING_HIGH_INVALID_BOOTSTRAP = "high_invalid_bootstrap_fraction"

GROUP_SUPPORT_CAPTION = (
    "Counts below describe the held-out evaluation rows used to compute "
    "these group metrics. They are held-out evaluation-set support, not "
    "population sizes."
)
GROUP_SUPPORT_SECTION_TITLE = "Group support on held-out evaluation set"

BOOTSTRAP_METHOD = "percentile bootstrap over held-out rows"
DEFAULT_N_BOOTSTRAP = 500
DEFAULT_BOOTSTRAP_RANDOM_STATE = 42
DEFAULT_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_HIGH_INVALID_FRACTION = 0.20
INSUFFICIENT_VALID_BOOTSTRAP_REASON = "insufficient valid bootstrap replicates"
BOOTSTRAP_CI_CAVEAT = (
    "Bootstrap intervals reflect variability from resampling the held-out "
    "evaluation rows with the fitted model fixed. They do not establish "
    "that a model is fair or compliant."
)


def _distance_alignment_label(val, cuts, severe_label):
    """Map a numeric distance to a caption; non-numeric values stay N/A."""
    if not isinstance(val, (int, float)) or isinstance(val, bool) or pd.isna(val):
        return "N/A"
    if val < cuts[0]:
        return "✅ Excellent alignment"
    if val < cuts[1]:
        return "🟢 Good alignment" if severe_label == "kl" else "🟢 Mild deviation"
    if val < cuts[2]:
        return "⚠️ Moderate deviation"
    return "🚨 Severe deviation"


def interpret_fairness_metrics(kl, wasserstein, tv):
    """
    Interpret KL Divergence, Wasserstein Distance, and Total Variation
    into severity labels for distribution-alignment diagnostics.
    """
    return {
        "KL Divergence": _distance_alignment_label(kl, (0.05, 0.15, 0.50), "kl"),
        "Wasserstein Distance": _distance_alignment_label(
            wasserstein, (0.05, 0.15, 0.30), "wass"
        ),
        "Total Variation": _distance_alignment_label(tv, (0.10, 0.25, 0.40), "tv"),
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
        DataFrame with a `Group` column (independent of `demographic_col`),
        observed shares, and, when a benchmark is supplied, disparity
        ratios plus a `Within Threshold?` criterion column.
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
        return _with_group_column(result_df, sort_by)

    result_df.attrs["benchmark_status"] = "ok"
    result_df.attrs["benchmark_message"] = None

    # Groups absent from the benchmark stay NaN; no fabricated value.
    result_df["Expected_%"] = result_df.index.map(benchmark_distribution)
    missing_groups = result_df["Expected_%"].isnull().sum()
    if missing_groups > 0:
        warnings.warn(
            f"{missing_groups} group(s) missing from benchmark. "
            "These groups will be reported as 'No benchmark available' "
            "instead of being assigned a fabricated expected value.",
            stacklevel=2,
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

    return _with_group_column(result_df, sort_by)


def _with_group_column(result_df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    """Reset the group index to a stable `Group` column name."""
    result_df = result_df.sort_values(sort_by, ascending=False)
    result_df.index.name = GROUP_COL
    return result_df.reset_index()


def plot_input_fairness(fairness_result, top_n=20, figsize=(12, 8)):
    """Enhanced plotting with better styling and annotations."""
    if fairness_result is None or fairness_result.empty:
        return None
    if GROUP_COL not in fairness_result.columns:
        return None

    plot_df = fairness_result.sort_values("Disparity_Ratio", ascending=False).head(
        top_n
    )

    colors = plot_df[THRESHOLD_STATUS_COL].map(
        {
            WITHIN_THRESHOLD: "#2E8B57",
            OUTSIDE_THRESHOLD: "#DC143C",
            NO_BENCHMARK_AVAILABLE: "#808080",
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=plot_df,
        y=GROUP_COL,
        x="Disparity_Ratio",
        hue=GROUP_COL,
        palette=list(colors),
        legend=False,
        ax=ax,
    )

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


def _preserve_group_label(value):
    """Convert numpy scalars to Python scalars without stringifying labels."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepare_held_out_fairness_arrays(y_true, y_pred, sensitive_features):
    """
    Align held-out arrays and drop rows with missing sensitive values.

    Fairlearn's MetricFrame grouping omits missing sensitive labels
    (pandas-style dropna). The same rows are excluded from MetricFrame,
    group support, and bootstrap so those views describe the same
    observations. Missing group labels are never dropped from only one
    of them.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_features = np.asarray(sensitive_features)

    if y_true.ndim != 1:
        y_true = np.ravel(y_true)
    if y_pred.ndim != 1:
        y_pred = np.ravel(y_pred)
    if sensitive_features.ndim != 1:
        raise ValueError("sensitive_features must be 1-dimensional")

    if not (len(y_true) == len(y_pred) == len(sensitive_features)):
        raise ValueError("All input arrays must have the same length")

    missing = pd.isna(sensitive_features)
    n_missing = int(np.sum(missing))
    if n_missing:
        keep = ~missing
        y_true = y_true[keep]
        y_pred = y_pred[keep]
        sensitive_features = sensitive_features[keep]

    if len(y_true) == 0:
        raise ValueError(
            "No held-out rows remain after excluding missing sensitive " "values"
        )

    return y_true, y_pred, sensitive_features, n_missing


def _normalize_fairness_class_label(value):
    """Map numpy/bool/integral-float labels to a Python 0/1 when possible."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _observed_class_labels(values):
    observed = set()
    for v in np.asarray(values).ravel():
        if pd.isna(v):
            observed.add(None)
            continue
        observed.add(_normalize_fairness_class_label(v))
    return observed


def _validate_binary_fairness_labels(y_true, y_pred):
    """Reject labels that are not binary 0/1 with positive class 1."""
    for arr in (y_true, y_pred):
        if not _observed_class_labels(arr) <= ALLOWED_BINARY_FAIRNESS_LABELS:
            raise ValueError(BINARY_FAIRNESS_LABELS_ERROR)


def _observed_group_labels(sensitive_features):
    """Unique held-out group labels, first-appearance order, missing omitted."""
    labels = []
    seen = set()
    for raw in sensitive_features:
        if pd.isna(raw):
            continue
        label = _preserve_group_label(raw)
        key = (type(label), label)
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def _group_key(value):
    label = _preserve_group_label(value)
    return (type(label), label)


def compute_group_support(
    y_true,
    y_pred,
    sensitive_features,
    positive_label=DEFAULT_POSITIVE_LABEL,
):
    """
    Held-out per-group support for model fairness metrics.

    Counts describe the same evaluation-set rows used by
    ``compute_output_fairness`` (after excluding missing sensitive
    values). They are not population sizes.

    ``positive_label`` is only used for these support counts. The live
    model-fairness path (MetricFrame, DP, EO, bootstrap) requires binary
    0/1 labels with positive class 1 and does not accept a custom
    positive class.

    Returns:
        DataFrame with one row per observed group and the columns in
        ``SUPPORT_TABLE_COLUMNS``. ``attrs['n_missing_sensitive']`` is
        the number of held-out rows excluded for a missing group label.
        ``attrs['positive_label']`` records the positive class used.
    """
    y_true, y_pred, sensitive_features, n_missing = (
        _prepare_held_out_fairness_arrays(y_true, y_pred, sensitive_features)
    )

    frame = pd.DataFrame(
        {
            "_y_true": y_true,
            "_y_pred": y_pred,
            GROUP_COL: [_preserve_group_label(v) for v in sensitive_features],
        }
    )
    rows = []
    for group, sub in frame.groupby(GROUP_COL, sort=False, dropna=True):
        n = int(len(sub))
        n_positive = int((sub["_y_true"] == positive_label).sum())
        n_predicted_positive = int((sub["_y_pred"] == positive_label).sum())
        rows.append(
            {
                GROUP_COL: group,
                SUPPORT_COL_N: n,
                SUPPORT_COL_POSITIVE_LABELS: n_positive,
                SUPPORT_COL_NEGATIVE_LABELS: n - n_positive,
                SUPPORT_COL_PREDICTED_POSITIVES: n_predicted_positive,
                SUPPORT_COL_PREDICTED_NEGATIVES: n - n_predicted_positive,
                SUPPORT_COL_SELECTION_DENOMINATOR: n,
            }
        )

    result = pd.DataFrame(rows, columns=SUPPORT_TABLE_COLUMNS)
    result.attrs["n_missing_sensitive"] = n_missing
    result.attrs["positive_label"] = positive_label
    return result


def _support_warning(group, code, message, **details):
    payload = {"group": group, "code": code, "message": message}
    payload.update(details)
    return payload


def assess_group_support(
    support_df,
    min_group_n=MIN_GROUP_N_WARNING,
    min_class_support=MIN_CLASS_SUPPORT_WARNING,
):
    """
    Structured heuristic warnings for held-out group support.

    Thresholds such as ``min_group_n=30`` and ``min_class_support=10``
    are stability heuristics, not statistical-validity cutoffs and not
    fairness verdicts.
    """
    warnings_out = []
    if support_df is None:
        return warnings_out

    n_missing = int(support_df.attrs.get("n_missing_sensitive") or 0)
    if n_missing:
        warnings_out.append(
            _support_warning(
                None,
                WARNING_MISSING_SENSITIVE_EXCLUDED,
                "Held-out rows with a missing sensitive-attribute value "
                "were excluded from group metrics, matching Fairlearn "
                "grouping. They are not shown as a separate group.",
                n_missing=n_missing,
            )
        )

    if support_df.empty:
        return warnings_out

    for _, row in support_df.iterrows():
        group = row[GROUP_COL]
        n = int(row[SUPPORT_COL_N])
        n_positive = int(row[SUPPORT_COL_POSITIVE_LABELS])
        n_negative = int(row[SUPPORT_COL_NEGATIVE_LABELS])

        if n < min_group_n:
            warnings_out.append(
                _support_warning(
                    group,
                    WARNING_SMALL_GROUP_N,
                    f"Small held-out group (n={n} < {min_group_n}); "
                    "group metrics may be unstable. This is a heuristic "
                    "stability warning, not a validity cutoff.",
                    n=n,
                    threshold=min_group_n,
                )
            )

        if n_positive == 0:
            warnings_out.append(
                _support_warning(
                    group,
                    WARNING_ZERO_POSITIVE_LABELS,
                    "This group has 0 positive true labels on the "
                    "held-out set, so true-positive-rate / recall-style "
                    "Equalized Odds components are mathematically "
                    "unsupported. Any numeric EO value should be read "
                    "with that support context.",
                    n_positive=n_positive,
                )
            )
        elif n_positive < min_class_support:
            warnings_out.append(
                _support_warning(
                    group,
                    WARNING_FEW_POSITIVE_LABELS,
                    "True-positive-based metrics for this group may be "
                    f"unstable (positive labels={n_positive} < "
                    f"{min_class_support}). Heuristic stability warning, "
                    "not a validity cutoff.",
                    n_positive=n_positive,
                    threshold=min_class_support,
                )
            )

        if n_negative == 0:
            warnings_out.append(
                _support_warning(
                    group,
                    WARNING_ZERO_NEGATIVE_LABELS,
                    "This group has 0 negative true labels on the "
                    "held-out set, so false-positive-rate-style "
                    "Equalized Odds components are mathematically "
                    "unsupported. Any numeric EO value should be read "
                    "with that support context.",
                    n_negative=n_negative,
                )
            )
        elif n_negative < min_class_support:
            warnings_out.append(
                _support_warning(
                    group,
                    WARNING_FEW_NEGATIVE_LABELS,
                    "False-positive-based metrics for this group may be "
                    f"unstable (negative labels={n_negative} < "
                    f"{min_class_support}). Heuristic stability warning, "
                    "not a validity cutoff.",
                    n_negative=n_negative,
                    threshold=min_class_support,
                )
            )

    return warnings_out


def minimum_valid_bootstrap_replicates(n_requested):
    """Require at least 100 valid draws and at least 50% of requested draws.

    Under this default rule, ``n_bootstrap < 100`` cannot produce a CI
    unless the caller overrides ``min_valid_replicates``. The live UI
    uses ``DEFAULT_N_BOOTSTRAP = 500``.
    """
    return max(100, int(np.ceil(0.5 * n_requested)))


def _is_finite_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(
        value
    )


def _fairlearn_metric_or_undefined(metric_fn, y_true, y_pred, sensitive_features):
    try:
        value = metric_fn(y_true, y_pred, sensitive_features=sensitive_features)
    except Exception as e:
        return _undefined_metric(str(e))
    if not _is_finite_number(value):
        return _undefined_metric(
            "Metric result was not a finite number.",
            value=value,
        )
    return float(value)


def _replicate_groups_complete(sensitive_sample, required_groups):
    present = {_group_key(g) for g in _observed_group_labels(sensitive_sample)}
    required = {_group_key(g) for g in required_groups}
    return required.issubset(present)


def _replicate_has_eo_class_support(y_true_sample, sensitive_sample):
    frame = pd.DataFrame(
        {
            "_y_true": y_true_sample,
            GROUP_COL: [_preserve_group_label(v) for v in sensitive_sample],
        }
    )
    for _, sub in frame.groupby(GROUP_COL, sort=False, dropna=True):
        n_positive = int((sub["_y_true"] == DEFAULT_POSITIVE_LABEL).sum())
        n_negative = int(len(sub) - n_positive)
        if n_positive == 0 or n_negative == 0:
            return False
    return True


def _ok_bootstrap_result(
    estimate,
    ci_lower,
    ci_upper,
    confidence_level,
    n_requested,
    n_valid,
    warnings_out=None,
):
    return {
        "status": "ok",
        "estimate": estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "n_requested": n_requested,
        "n_valid": n_valid,
        "method": BOOTSTRAP_METHOD,
        "warnings": list(warnings_out or []),
    }


def _unavailable_bootstrap_result(
    reason,
    n_requested,
    n_valid,
    confidence_level,
    estimate=None,
    warnings_out=None,
    **details,
):
    payload = _undefined_metric(
        reason,
        estimate=estimate,
        ci_lower=None,
        ci_upper=None,
        confidence_level=confidence_level,
        n_requested=n_requested,
        n_valid=n_valid,
        method=BOOTSTRAP_METHOD,
        valid_bootstrap_replicates=n_valid,
        requested_bootstrap_replicates=n_requested,
        warnings=list(warnings_out or []),
        **details,
    )
    return payload


def _bootstrap_support_warnings(support_df, n_valid, n_requested):
    warnings_out = []
    support_warnings = assess_group_support(support_df)
    sparse_codes = {
        WARNING_SMALL_GROUP_N,
        WARNING_FEW_POSITIVE_LABELS,
        WARNING_FEW_NEGATIVE_LABELS,
        WARNING_ZERO_POSITIVE_LABELS,
        WARNING_ZERO_NEGATIVE_LABELS,
    }
    if any(item["code"] in sparse_codes for item in support_warnings):
        warnings_out.append(
            _support_warning(
                None,
                WARNING_SPARSE_BOOTSTRAP,
                "Held-out groups or class counts are sparse, so the "
                "bootstrap interval can be unstable. This is descriptive "
                "support context, not a pass/fail cutoff.",
            )
        )
    if n_requested > 0 and (1.0 - (n_valid / n_requested)) > (
        BOOTSTRAP_HIGH_INVALID_FRACTION
    ):
        warnings_out.append(
            _support_warning(
                None,
                WARNING_HIGH_INVALID_BOOTSTRAP,
                "A large share of bootstrap replicates were invalid "
                f"({n_valid} valid of {n_requested} requested) because "
                "required group or class support disappeared. This is "
                "descriptive, not a pass/fail cutoff.",
                n_valid=n_valid,
                n_requested=n_requested,
            )
        )
    return warnings_out


def bootstrap_fairness_metric(
    y_true,
    y_pred,
    sensitive_features,
    metric_fn,
    *,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    random_state=DEFAULT_BOOTSTRAP_RANDOM_STATE,
    min_valid_replicates=None,
    require_eo_class_support=False,
):
    """
    Percentile bootstrap CI for one held-out fairness statistic.

    Resamples the existing ``(y_true, y_pred, sensitive_features)``
    tuples with replacement. The fitted model is not retrained. The
    interval estimates sampling variability of the statistic on the
    held-out evaluation rows, conditional on the already-trained model
    and its predictions. It is not training uncertainty, a population
    causal interval, a fairness guarantee, or a regulatory CI.

    Labels must be binary 0/1 with positive class 1, matching Fairlearn
    DP/EO. Replicates that lose a required group (or, for Equalized
    Odds, a required positive/negative class inside a group) are marked
    invalid and excluded from the percentile calculation. They are not
    replaced with 0, inf, or any other fabricated value.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    y_true, y_pred, sensitive_features, _n_missing = (
        _prepare_held_out_fairness_arrays(y_true, y_pred, sensitive_features)
    )
    _validate_binary_fairness_labels(y_true, y_pred)
    required_groups = _observed_group_labels(sensitive_features)
    if min_valid_replicates is None:
        min_valid_replicates = minimum_valid_bootstrap_replicates(n_bootstrap)

    point = _fairlearn_metric_or_undefined(
        metric_fn, y_true, y_pred, sensitive_features
    )
    estimate = None if isinstance(point, dict) else point
    support_df = compute_group_support(
        y_true,
        y_pred,
        sensitive_features,
        positive_label=DEFAULT_POSITIVE_LABEL,
    )

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    valid_values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        pred_b = y_pred[idx]
        sens_b = sensitive_features[idx]
        if not _replicate_groups_complete(sens_b, required_groups):
            continue
        if require_eo_class_support and not _replicate_has_eo_class_support(
            y_b, sens_b
        ):
            continue
        value = _fairlearn_metric_or_undefined(metric_fn, y_b, pred_b, sens_b)
        if isinstance(value, dict):
            continue
        valid_values.append(value)

    n_valid = len(valid_values)
    extra_warnings = _bootstrap_support_warnings(support_df, n_valid, n_bootstrap)

    if n_valid < min_valid_replicates:
        return _unavailable_bootstrap_result(
            INSUFFICIENT_VALID_BOOTSTRAP_REASON,
            n_requested=n_bootstrap,
            n_valid=n_valid,
            confidence_level=confidence_level,
            estimate=estimate,
            warnings_out=extra_warnings,
            min_valid_replicates=min_valid_replicates,
        )

    alpha = 1.0 - confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    ci_lower, ci_upper = np.percentile(valid_values, [lower_q, upper_q])
    if estimate is None:
        reason = (
            point.get("reason", "undefined")
            if isinstance(point, dict)
            else "undefined"
        )
        return _unavailable_bootstrap_result(
            reason,
            n_requested=n_bootstrap,
            n_valid=n_valid,
            confidence_level=confidence_level,
            estimate=None,
            warnings_out=extra_warnings,
        )
    return _ok_bootstrap_result(
        estimate=estimate,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence_level=confidence_level,
        n_requested=n_bootstrap,
        n_valid=n_valid,
        warnings_out=extra_warnings,
    )


def bootstrap_output_fairness(
    y_true,
    y_pred,
    sensitive_features,
    *,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    random_state=DEFAULT_BOOTSTRAP_RANDOM_STATE,
    min_valid_replicates=None,
):
    """
    Percentile bootstrap CIs for Demographic Parity and Equalized Odds.

    Both intervals resample the same held-out ``(y_true, y_pred,
    sensitive_features)`` tuples. The model is not retrained. Reported
    ``estimate`` values are the ordinary Fairlearn point estimates on
    the original held-out arrays, not bootstrap means.

    Labels must be binary 0/1 with positive class 1.
    """
    dp = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive_features,
        demographic_parity_difference,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
        min_valid_replicates=min_valid_replicates,
        require_eo_class_support=False,
    )
    eo = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive_features,
        equalized_odds_difference,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
        min_valid_replicates=min_valid_replicates,
        require_eo_class_support=True,
    )
    return {
        "Demographic Parity Difference": dp,
        "Equalized Odds Difference": eo,
    }


def compute_output_fairness(y_true, y_pred, sensitive_features):
    """
    Group-wise held-out fairness metrics for a binary classifier.

    Inputs must be the same held-out ``y_test`` / ``y_pred`` /
    ``sensitive_test`` arrays used for model evaluation. Missing
    sensitive values are excluded before MetricFrame so group metrics
    and group-support counts describe the same rows.

    Labels must be binary 0/1 with positive class 1. That matches
    Fairlearn ``demographic_parity_difference`` /
    ``equalized_odds_difference``, which do not accept ``pos_label``.
    The live modeling path supplies 0/1 labels at this boundary
    (string targets are LabelEncoded earlier). Precision, recall, F1,
    selection rate, DP, and EO all use this same positive class.
    """
    y_true, y_pred, sensitive_features, _n_missing = (
        _prepare_held_out_fairness_arrays(y_true, y_pred, sensitive_features)
    )
    _validate_binary_fairness_labels(y_true, y_pred)

    metrics = {
        "Accuracy": accuracy_score,
        "Precision": lambda yt, yp: precision_score(
            yt, yp, zero_division=0, pos_label=DEFAULT_POSITIVE_LABEL
        ),
        "Recall": lambda yt, yp: recall_score(
            yt, yp, zero_division=0, pos_label=DEFAULT_POSITIVE_LABEL
        ),
        "F1": lambda yt, yp: f1_score(
            yt, yp, zero_division=0, pos_label=DEFAULT_POSITIVE_LABEL
        ),
        "Selection Rate": lambda yt, yp: selection_rate(
            yt, yp, pos_label=DEFAULT_POSITIVE_LABEL
        ),
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

    # Keep difference and ratio as distinct keys. Mapping both through a
    # shared display name would silently overwrite one of the values.
    disparity_plain = {
        _output_fairness_display_key(k): v for k, v in disparity_summary.items()
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


def _output_fairness_display_key(internal_key: str) -> str:
    """Map internal disparity/ratio keys to unique display names."""
    if internal_key.endswith(" disparity"):
        return f"{internal_key[: -len(' disparity')]} Difference"
    if internal_key.endswith(" ratio"):
        return f"{internal_key[: -len(' ratio')]} Ratio"
    return internal_key
