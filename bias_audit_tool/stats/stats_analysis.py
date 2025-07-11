import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import ks_2samp


def run_chi_square(df, col, target, group_col):
    """
    Perform a chi-square test between a categorical feature and a group column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Categorical column to test.
        target (str): Not used, included for compatibility.
        group_col (str): Grouping column (e.g., 'race', 'gender').

    Returns:
        dict: Chi-square test result including p-value, groups,
              and positive rates.
    """
    contingency = pd.crosstab(df[col], df[group_col])
    chi2, p, _, _ = chi2_contingency(contingency)

    group_values = df.groupby(group_col)[col].mean().to_dict()
    return {
        "feature": col,
        "test": "chi-square",
        "p_value": p,
        "groups": list(group_values.keys()),
        "metric": "positive_rate",
        "group_values": group_values,
    }


def run_anova(df, col, group_col):
    """
    Perform a one-way ANOVA test for differences in means across groups.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Numeric column to test.
        group_col (str): Grouping column for comparison.

    Returns:
        dict: ANOVA test result including p-value and group names.
    """
    groups = [group[col].dropna() for name, group in df.groupby(group_col)]
    fval, p = f_oneway(*groups)

    return {
        "feature": col,
        "test": "anova",
        "p_value": p,
        "groups": df[group_col].unique().tolist(),
    }


def run_ks_test(df, col, group_col):
    """
    Perform pairwise Kolmogorov-Smirnov tests across all group pairs.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Numeric column to test.
        group_col (str): Column to define group membership.

    Returns:
        list[dict]: List of test results for each group pair,
                    including statistic and p-value.
    """
    groups = df[group_col].dropna().unique()
    results = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1 = df[df[group_col] == groups[i]][col].dropna()
            g2 = df[df[group_col] == groups[j]][col].dropna()
            stat, pval = ks_2samp(g1, g2)
            results.append(
                {
                    "feature": col,
                    "group1": groups[i],
                    "group2": groups[j],
                    "stat": stat,
                    "p_value": pval,
                }
            )
    return results


def low_variance_filter(df, threshold=0.01):
    """
    Identify columns with variance below a given threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Variance threshold below which columns are flagged.

    Returns:
        list[str]: List of low-variance column names.
    """
    low_var_cols = df.var()[df.var() < threshold].index.tolist()
    return low_var_cols


def groupwise_missing_rate(df, target_col, group_col):
    """
    Compute missing value ratio for a column, broken down by group.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Column to evaluate for missingness.
        group_col (str): Column to group by (e.g., 'race', 'gender').

    Returns:
        dict: Dictionary mapping each group to its missing value ratio.
    """
    return (
        df.groupby(group_col)[target_col].apply(lambda x: x.isna().mean()).to_dict()
    )
