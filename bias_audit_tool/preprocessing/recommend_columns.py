import logging
from typing import List
from typing import Optional

import pandas as pd

# Live recommendation is a column-name substring heuristic plus simple
# metadata filters (cardinality, missingness, dtype). It does not infer
# protected characteristics from values, is English-vocabulary only, and
# is not jurisdictionally exhaustive. Unused regex/value matchers were
# removed so the implementation matches what the UI actually does.
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

SENSITIVE_ATTRIBUTE_CANDIDATE_CAPTION = (
    "Candidate sensitive attributes based on column-name and metadata "
    "heuristics — review before use."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def identify_by_hierarchy(df: pd.DataFrame) -> list[str]:
    """
    List columns whose names look demographic under a name heuristic.

    A column is a name-level candidate if it starts with ``demographics.``
    or if any token in ``DEMOGRAPHIC_CATEGORIES`` is a case-insensitive
    substring of the column name.

    This is not protected-class detection. Known limitations, documented
    by tests:
    - False-positive risk: names such as ``region_id``,
      ``family_history_of_cancer``, and ``device_orientation`` match
      because they contain ``region``, ``family``, or ``orientation``.
    - False-negative / unsupported vocabulary: names such as
      ``nationality``, ``citizenship``, and non-English demographic
      terms are not guaranteed to match.
    """
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


def get_category(row, cols, kw):
    active = [col for col in cols if row[col] == 1]
    if len(active) > 1:
        logging.warning(f"[⚠️ Collision] Multiple 1's for {kw}: {active}")
    if active:
        return active[0].replace(f"{kw}_", "")
    return "unknown"


def merge_dummy_columns_and_get_mapping(
    df: pd.DataFrame, prefix_keywords: list[str], drop: bool = True
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Merge one-hot encoded dummy columns (e.g., gender_female, gender_male)
    into a single categorical column with '_mapped' suffix, and return
    the mapping from original nested name to merged column name.

    Args:
        df (pd.DataFrame): The input DataFrame.
        prefix_keywords (list[str]): List of column prefixes to merge.
        drop (bool): Whether to drop original dummy columns after merging.

    Returns:
        Tuple[pd.DataFrame, dict[str, str]]: Modified DataFrame and mapping dict.
    """
    df_new = df.copy()
    mapping = {}

    for keyword in prefix_keywords:
        dummy_cols = [col for col in df_new.columns if col.startswith(keyword + "_")]
        if not dummy_cols:
            continue

        mapped_col = f"{keyword}_mapped"

        if len(dummy_cols) >= 2:
            df_new[mapped_col] = df_new.apply(
                lambda row, dcols=dummy_cols, key=keyword: get_category(
                    row, dcols, key
                ),
                axis=1,
            )
            logging.info(f"[✅ Merged] {len(dummy_cols)} columns → {mapped_col}")

        elif len(dummy_cols) == 1:
            single_col = dummy_cols[0]
            df_new[mapped_col] = df_new[single_col].apply(
                lambda x: str(x) if pd.notnull(x) else "unknown"
            )
            logging.info(
                f"[ℹ️ Single dummy] Converted '{single_col}' to '{mapped_col}'"
            )

        # Register mapping
        mapping[keyword] = mapped_col

        if drop:
            if mapped_col in df_new.columns and df_new[mapped_col].nunique() > 1:
                df_new.drop(columns=dummy_cols, inplace=True)
                logging.info(
                    f"[🧹 Dropped] {dummy_cols} after merge into '{mapped_col}'"
                )
            else:
                logging.warning(
                    f"[⚠️ Retained] Skipped dropping {dummy_cols} "
                    "due to insufficient category info."
                )

    return df_new, mapping


def is_categorical_column(series: pd.Series) -> bool:
    """Check if a series is suitable as a categorical group variable."""
    dtype = series.dtype
    values = series.dropna().unique()
    n_unique = len(values)
    is_binary = set(values) <= {0, 1, True, False, 0.0, 1.0}
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or (n_unique == 2 and is_binary)
        or (n_unique <= 15 and pd.api.types.is_integer_dtype(dtype))
    )


def recommend_demographic_columns(
    df: pd.DataFrame, demographic_cols: Optional[List[str]] = None
) -> tuple[pd.DataFrame, List[str]]:
    """
    Return candidate grouping columns from name + metadata heuristics.

    Algorithm:
    1. Optionally merge known one-hot demographic dummy prefixes.
    2. Name filter via ``identify_by_hierarchy`` (substring of
       ``DEMOGRAPHIC_CATEGORIES``, or ``demographics.`` prefix).
    3. Keep columns with 2–15 unique non-null values, categorical-like
       dtype (object/string, binary, or integer with ≤15 levels), and
       missingness ≤ 50%.

    Candidates require human review. Non-recommended columns may still
    be sensitive; recommended columns may be false positives.
    """
    # Step 1: Merge dummy columns and get mapping to _mapped names
    df, mapping = merge_dummy_columns_and_get_mapping(
        df,
        prefix_keywords=[
            "demographic.gender",
            "demographic.race",
            "demographic.ethnicity",
            "demographic.age",
        ],
        drop=True,
    )
    print(f"[DEBUG] df.columns after merge: {df.columns}")
    print(f"[DEBUG] merge mapping: {mapping}")

    # Step 2: Identify candidate columns
    if demographic_cols is None:
        demographic_cols = identify_by_hierarchy(df)
    print(f"[DEBUG] Evaluating candidate columns from: {demographic_cols}")

    candidate_cols = []
    for col in demographic_cols:
        mapped_col = mapping.get(col, col)

        if mapped_col not in df.columns:
            print(f"[⚠️ WARNING] Column '{mapped_col}' not found in df.")
            continue

        series = df[mapped_col]
        n_unique = series.nunique(dropna=True)
        missing_ratio = series.isna().mean()
        is_categorical = is_categorical_column(series)

        print(
            f"[DEBUG] Column '{mapped_col}': unique={n_unique}, "
            f"missing={missing_ratio:.2f}, categorical={is_categorical}"
        )

        if 2 <= n_unique <= 15 and is_categorical and missing_ratio <= 0.5:
            candidate_cols.append(mapped_col)

    if candidate_cols:
        print(f"[DEBUG] Recommended group column(s): {candidate_cols}")
        debug_check_mapped_columns(df)
        return df, candidate_cols

    print("[⚠️ WARNING] No suitable group column found.")
    return df, []


def debug_check_mapped_columns(df, expected_suffix="_mapped"):
    mapped_cols = [col for col in df.columns if col.endswith(expected_suffix)]
    if mapped_cols:
        print(f"[✅] Found {len(mapped_cols)} mapped columns:")
        for col in mapped_cols:
            print(f"   • {col}")
    else:
        print("[❌] No mapped columns found.")
