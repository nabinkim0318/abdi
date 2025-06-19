# interpretation.py


def interpret_encoding(col: str) -> str:
    return (
        f"The `{col}` column is recognized as categorical "
        "and is suitable for OneHot or Label encoding. "
        "This typically applies to variables like gender or race "
        "that represent distinct groups."
    )


def interpret_normalization(col: str) -> str:
    return (
        f"The `{col}` column is a continuous numerical feature. "
        "Normalization (e.g., MinMax scaling) is recommended to "
        "stabilize model performance. "
        "Examples include age, tumor size, or lab values."
    )


def interpret_missing(col: str, missing_rate: float) -> str:
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
    return (
        f"The `{col}` column shows a statistically significant "
        f"difference in value distribution across {group} groups "
        f"(p = {pval:.3f}). This suggests the variable may affect "
        "group-wise interpretation or model behavior."
    )


def interpret_anova(col: str, pval: float) -> str:
    return (
        f"The `{col}` column exhibits significant mean differences "
        f"across groups and may be associated with the outcome variable "
        f"(ANOVA p = {pval:.3f})."
    )


def interpret_missing_bias(col: str, group: str, rate: float) -> str:
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
    return (
        f"The `{group1}` group has a positive prediction rate of "
        f"{group1_score:.2f}, while the `{group2}` group has "
        f"{group2_score:.2f}. This indicates that the model may "
        "not be offering equal prediction opportunities."
    )


def interpret_fairness_warning(
    metric: str, gap: float, threshold: float = 0.1
) -> str:
    return (
        f"The fairness metric `{metric}` exceeds the acceptable "
        f"gap threshold (gap = {gap:.2f} > {threshold}). "
        "Adjustment may be necessary."
    )


def interpret_shap_group_diff(col: str, group: str) -> str:
    return (
        f"The `{col}` feature contributes more strongly to "
        f"predictions for the `{group}` group, indicating the model may "
        "rely more heavily on it for that subgroup."
    )


def interpret_permutation_importance(col: str) -> str:
    return (
        f"The `{col}` feature is among the most important "
        "variables in determining model predictions, "
        "suggesting it plays a key explanatory role."
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
    stat_result: {
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
