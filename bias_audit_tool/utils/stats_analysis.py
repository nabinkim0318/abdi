import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway


def run_chi_square(df, col, target, group_col):
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
    groups = [group[col].dropna() for name, group in df.groupby(group_col)]
    fval, p = f_oneway(*groups)

    return {
        "feature": col,
        "test": "anova",
        "p_value": p,
        "groups": df[group_col].unique().tolist(),
    }
