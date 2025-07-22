from typing import List
from typing import Optional

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


def validate_age_column(df: pd.DataFrame, age_col: str) -> pd.DataFrame:
    if age_col in df.columns:
        invalid = df[df[age_col] < 0]
        if not invalid.empty:
            print(
                f"[WARNING] ⚠️ Dropping {len(invalid)} "
                f"rows with negative age in '{age_col}'"
            )
            df = df[df[age_col] >= 0]

        too_old = df[df[age_col] > 120]
        if not too_old.empty:
            print(
                f"[WARNING] ⚠️ Dropping {len(too_old)} "
                f"rows with age > 120 in '{age_col}'"
            )
            df = df[df[age_col] <= 120]
    return df


def bin_age_column(
    df: pd.DataFrame, age_col: str = "age", new_col: str = "age_group"
) -> pd.DataFrame:
    """
    Bin the age column into groups like 'child', 'teen', 'young adult', etc.
    """
    if age_col not in df.columns:
        print(f"[INFO] No '{age_col}' column to bin.")
        return df

    bins = [0, 13, 18, 30, 45, 60, 75, 121]
    labels = [
        "child",  # 0–12
        "teen",  # 13–17
        "young adult",  # 18–29
        "adult",  # 30–44
        "mid-age",  # 45–59
        "senior",  # 60–74
        "elder",  # 75–120
    ]
    if age_col in df.columns:
        df[new_col] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
        print(f"[INFO] ✅ Binned '{age_col}' into '{new_col}'")
    return df


def process_age_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and process age-related columns:
    - Removes negative or unrealistic age values (<0 or >120)
    - Bins age into categorical groups

    Returns:
        pd.DataFrame: Updated DataFrame with new age group columns.
    """
    # Step 1: identify age columns
    age_cols = [col for col in identify_by_hierarchy(df) if "age" in col.lower()]
    print(f"[INFO] 🔍 Identified age columns: {age_cols}")

    # Step 2: apply validation and binning
    for age_col in age_cols:
        if age_col not in df.columns:
            print(f"[WARNING] Column '{age_col}' not found in DataFrame.")
            continue

        # Validate (remove invalid rows)
        df = validate_age_column(df, age_col=age_col)

        # Bin age into groups
        age_group_col = f"{age_col}_group"
        df = bin_age_column(df, age_col=age_col, new_col=age_group_col)
    return df


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
) -> Optional[List[str]]:
    """
    Recommend demographic columns for grouping.

    Returns:
        A list of column names that are suitable for demographic grouping,
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
