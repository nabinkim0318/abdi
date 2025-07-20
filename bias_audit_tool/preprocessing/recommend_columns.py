from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd

DEMOGRAPHIC_KEYWORDS = [
    r"gender|sex",
    r"age",
    r"race|ethnicity|ethnic",
    r"income|salary|wage",
    r"education|degree|school",
    r"employment|job|occupation",
    r"disability|impairment",
    r"language|lang",
    r"region|zip|location|state|city",
    r"religion|faith",
    r"orientation|sexual",
    r"marital|relationship",
    r"children|kids|dependents",
    r"family",
]

VALUE_PATTERNS = {
    "gender": ["male", "female", "nonbinary"],
    "race": ["white", "black", "asian", "hispanic"],
    "marital": ["married", "single", "divorced"],
}

DEMOGRAPHIC_CATEGORIES = [
    "gender",
    "sex",
    "age",
    "race",
    "ethnicity",
    "income",
    "education",
    "employment",
    "disability",
    "language",
    "region",
    "religion",
    "orientation",
    "marital",
    "children",
    "family",
]


def identify_by_hierarchy(df: pd.DataFrame) -> list[str]:
    matched = [
        col
        for col in df.columns
        if col.startswith("demographics.")
        or any(k in col.lower() for k in DEMOGRAPHIC_CATEGORIES)
    ]
    print(f"[DEBUG] Hierarchy-based identified columns: {matched}")
    return matched


def merge_dummy_columns(
    df: pd.DataFrame, prefix_keywords: list[str]
) -> pd.DataFrame:
    """
    Merge one-hot encoded dummy columns (e.g., gender_female,
    gender_male) into single categorical columns.
    """
    df_new = df.copy()
    for keyword in prefix_keywords:
        dummy_cols = [col for col in df.columns if col.startswith(keyword + "_")]
        if len(dummy_cols) >= 2:

            def get_category(row, cols=dummy_cols, kw=keyword):
                for col in cols:
                    if row.get(col) == 1:
                        return col.replace(kw + "_", "")
                return "unknown"

            df_new[keyword] = df_new[dummy_cols].apply(get_category, axis=1)
    return df_new


def recommend_demographic_columns(
    df: pd.DataFrame, demographic_cols: Optional[List[str]] = None
) -> Optional[List[Tuple[str, int, float]]]:
    """
    Recommend demographic columns for grouping.

    Returns:
        A tuple containing:
            - A list of (column_name, n_unique, missing_ratio)
            - The default recommended column name
        or None if no suitable column is found.
    """
    # Step 1: Merge dummy columns (e.g., gender_male, gender_female → gender)
    df = merge_dummy_columns(
        df,
        prefix_keywords=[
            "demographic.gender",
            "demographic.race",
            "demographic.ethnicity",
        ],
    )

    if demographic_cols is None:
        demographic_cols = identify_by_hierarchy(df)

    print(f"[DEBUG] Evaluating candidate columns from: {demographic_cols}")
    candidate_cols = []

    for col in demographic_cols:
        if col not in df.columns:
            print(f"[WARNING] Column '{col}' not in DataFrame.")
            continue

        n_unique = df[col].nunique(dropna=True)
        missing_ratio = df[col].isna().mean()
        dtype = df[col].dtype

        col_values = df[col].dropna().unique()
        is_binary_values = set(col_values) <= {0, 1, True, False, 0.0, 1.0}
        is_bool_or_dummy = len(col_values) == 2 and is_binary_values

        is_categorical = (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or is_bool_or_dummy
            or (n_unique <= 15 and pd.api.types.is_integer_dtype(dtype))
        )

        print(
            f"[DEBUG] Column '{col}': unique={n_unique}, "
            f"missing={missing_ratio:.2f}, "
            f"categorical={is_categorical}"
        )

        if 2 <= n_unique <= 15 and is_categorical and missing_ratio <= 0.5:
            candidate_cols.append(col)

    if candidate_cols:
        print(f"[DEBUG] Recommended group column(s): {candidate_cols}")
        return candidate_cols
    else:
        print("[WARNING] No suitable group column found.")
        return None
