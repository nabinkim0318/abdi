# interpretation.py
"""
interpretation.py

Provides interpretation messages and textual explanations for statistical results,
data preprocessing decisions, fairness assessments, and model behavior in the
Bias Audit Tool.

This module contains human-readable message generators for:
- Encoding and normalization guidance
- Missing data handling recommendations
- Statistical test interpretations (e.g., ANOVA, chi-square)
- Fairness disparity explanations (e.g., demographic parity, TPR gaps)
- SHAP and permutation importance interpretations

Used in report generation and UI explanations to enhance interpretability
for non-technical stakeholders.

Functions:
    interpret_encoding(col)
    interpret_normalization(col)
    interpret_missing(col, missing_rate)
    interpret_distribution_difference(col, group, pval)
    interpret_anova(col, pval)
    interpret_missing_bias(col, group, rate)
    interpret_fairness_gap(metric_name, group1, group1_score, group2, group2_score)
    interpret_demographic_parity(group1, group1_score, group2, group2_score)
    interpret_fairness_warning(metric, gap, threshold)
    interpret_shap_group_diff(col, group)
    interpret_permutation_importance(col)
    interpret_groupwise_missing(col, group, rate)
    interpret_summary(n_columns, n_flagged, col_1, col_2, group_pair, col_shap_top)
    generate_interpretation(stat_result)
"""


def interpret_encoding(col: str) -> str:
    """
    Generate a message suggesting encoding for a categorical column.

    Args:
        col (str): Name of the column.

    Returns:
        str: Explanation string.
    """
    return (
        f"The `{col}` column is recognized as categorical "
        "and is suitable for OneHot or Label encoding. "
        "This typically applies to variables like gender or race "
        "that represent distinct groups."
    )


def interpret_normalization(col: str) -> str:
    """
    Generate a message recommending normalization for a numerical column.

    Args:
        col (str): Name of the column.

    Returns:
        str: Explanation string.
    """
    return (
        f"The `{col}` column is a continuous numerical feature. "
        "Normalization (e.g., MinMax scaling) is recommended to "
        "stabilize model performance. "
        "Examples include age, tumor size, or lab values."
    )


def interpret_missing(col: str, missing_rate: float) -> str:
    """
    Recommend handling strategies based on the missing rate of a column.

    Args:
        col (str): Column name.
        missing_rate (float): Proportion of missing values (0–1).

    Returns:
        str: Recommendation message.
    """
    if missing_rate > 0.3:
        return (
            f"The `{col}` column has a high missing rate of "
            f"{missing_rate:.1%}, which may impact analysis reliability. "
            "Removing this column is recommended."
        )
    else:
        return (
            f"The `{col}` column contains about {missing_rate:.1%} "
            "missing values. Options include replacing with the mean "
            "or dropping the column."
        )


def interpret_distribution_difference(col: str, group: str, pval: float) -> str:
    """
    Summarize statistical significance of distribution differences across groups.

    Args:
        col (str): Column name.
        group (str): Grouping feature name.
        pval (float): p-value from test.

    Returns:
        str: Explanation message.
    """
    return (
        f"The `{col}` column shows a statistically significant "
        f"difference in value distribution across {group} groups "
        f"(p = {pval:.3f}). This suggests the variable may affect "
        "group-wise interpretation or model behavior."
    )


def interpret_anova(col: str, pval: float) -> str:
    """
    Interpret ANOVA result for mean difference across groups.

    Args:
        col (str): Column name.
        pval (float): p-value from ANOVA test.

    Returns:
        str: Interpretation message.
    """
    return (
        f"The `{col}` column exhibits significant mean differences "
        f"across groups and may be associated with the outcome variable "
        f"(ANOVA p = {pval:.3f})."
    )


def interpret_missing_bias(col: str, group: str, rate: float) -> str:
    """
    Highlight bias due to missing data in a specific group.

    Args:
        col (str): Column name.
        group (str): Group name.
        rate (float): Missing rate (0–1) for the group.

    Returns:
        str: Explanation message.
    """
    return (
        f"The `{col}` column has a higher missing rate in the "
        f"`{group}` group (e.g., {rate:.1%}), which could lead to "
        "biased predictions or analysis outcomes for that group."
    )


def interpret_fairness_gap(
    metric_name: str,
    group1: str,
    group1_score: float,
    group2: str,
    group2_score: float,
) -> str:
    """
    Explain disparity in fairness metric between two groups.

    Args:
        metric_name (str): Fairness metric name (e.g., TPR).
        group1 (str): Name of first group.
        group1_score (float): Score for first group.
        group2 (str): Name of second group.
        group2_score (float): Score for second group.

    Returns:
        str: Explanation message.
    """
    return (
        f"The model has an overall {metric_name} score, "
        f"but a gap exists between the `{group1}` group "
        f"({group1_score:.2f}) and the `{group2}` group "
        f"({group2_score:.2f}). This suggests the model may be "
        f"less sensitive to positive outcomes in the `{group2}` group."
    )


def interpret_demographic_parity(
    group1: str, group1_score: float, group2: str, group2_score: float
) -> str:
    """
    Describe demographic parity difference between two groups.

    Args:
        group1 (str): First group name.
        group1_score (float): Rate for group1.
        group2 (str): Second group name.
        group2_score (float): Rate for group2.

    Returns:
        str: Explanation message.
    """
    return (
        f"The `{group1}` group has a positive prediction rate of "
        f"{group1_score:.2f}, while the `{group2}` group has "
        f"{group2_score:.2f}. This indicates that the model may "
        "not be offering equal prediction opportunities."
    )


def interpret_fairness_warning(
    metric: str, gap: float, threshold: float = 0.1
) -> str:
    """
    Warn if fairness metric gap exceeds a threshold.

    Args:
        metric (str): Fairness metric name.
        gap (float): Observed disparity.
        threshold (float, optional): Acceptable gap. Default is 0.1.

    Returns:
        str: Warning message.
    """
    return (
        f"The fairness metric `{metric}` exceeds the acceptable "
        f"gap threshold (gap = {gap:.2f} > {threshold}). "
        "Adjustment may be necessary."
    )


def interpret_shap_group_diff(col: str, group: str) -> str:
    """
    Highlight SHAP-based feature importance variation across group.

    Args:
        col (str): Feature name.
        group (str): Group with higher SHAP contribution.

    Returns:
        str: Explanation message.
    """
    return (
        f"The `{col}` feature contributes more strongly to "
        f"predictions for the `{group}` group, indicating the model may "
        "rely more heavily on it for that subgroup."
    )


def interpret_permutation_importance(col: str) -> str:
    """
    Indicate that a feature is highly important via permutation.

    Args:
        col (str): Feature name.

    Returns:
        str: Importance message.
    """
    return (
        f"The `{col}` feature is among the most important "
        "variables in determining model predictions, "
        "suggesting it plays a key explanatory role."
    )


def interpret_groupwise_missing(col, group, rate):
    """
    Indicate group-specific missing value rate.

    Args:
        col (str): Column name.
        group (str): Group name.
        rate (float): Missing rate (0–1) for the group.

    Returns:
        str: Bias warning message.
    """
    return (
        f"The `{col}` column has a missing rate of "
        f"{rate:.1%} in the `{group}` group. "
        "This could bias model performance."
    )


def interpret_summary(
    n_columns: int,
    n_flagged: int,
    col_1: str,
    col_2: str,
    group_pair: str,
    col_shap_top: str,
) -> str:
    return (
        f"📊 **Summary**\n"
        f"- Out of {n_columns} total variables, {n_flagged} showed "
        "potential bias in missingness or group-wise differences.\n"
        f"- Key features showing distribution differences by group: "
        f"`{col_1}`, `{col_2}`\n"
        f"- Fairness gaps in outcome prediction were detected "
        f"between: {group_pair}\n"
        f"- Top SHAP-attributed features for group-specific "
        f"prediction include: `{col_shap_top}`\n\n"
        f"💡 **Suggestions**\n"
        f"- Consider revising inclusion criteria or model treatment "
        f"for `{col_1}`.\n"
        f"- `{group_pair}` shows low TPR, indicating a need to "
        "improve model sensitivity for that group."
    )


def generate_interpretation(stat_result: dict) -> str:
    """
    Generate a textual interpretation of statistical test results.

    Supports interpretation of group-wise outcome disparities based on
    chi-square, ANOVA, or other statistical tests, and summarizes
    which groups show the highest or lowest values for the metric.

    Args:
        stat_result (dict): Dictionary with statistical test results. Example:
            {
                "feature": "race",
                "test": "chi-square",
                "p_value": 0.0123,
                "groups": ["White", "Black", "Asian"],
                "metric": "positive_rate",
                "group_values": {
                    "White": 0.34,
                    "Black": 0.62,
                    "Asian": 0.45
                }
            }

    Returns:
        str: Human-readable message describing the statistical result,
             indicating significance and the magnitude of group differences.

    Raises:
        KeyError: If required keys are missing from `stat_result`.
    """
    feature = stat_result["feature"]
    test = stat_result["test"]
    p_val = stat_result["p_value"]
    metric = stat_result.get("metric", "outcome rate")
    # groups = stat_result["groups"]
    values = stat_result["group_values"]

    top_group = max(values, key=values.get)
    bottom_group = min(values, key=values.get)

    if p_val < 0.05:
        return (
            f"🧪 The {test} test for `{feature}` shows a "
            "**statistically significant difference** "
            f"(p = {p_val:.4f}) in `{metric}` across groups. "
            f"For example, `{top_group}` has the highest rate "
            f"({values[top_group]:.2f}), while `{bottom_group}` "
            f"has the lowest ({values[bottom_group]:.2f})."
        )
    else:
        return (
            f"✅ The {test} test for `{feature}` suggests "
            "**no significant difference** across groups "
            f"(p = {p_val:.4f})."
        )
