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
    "demo_group",
]

SENSITIVE_ATTRIBUTE_CANDIDATE_CAPTION = (
    "Candidate sensitive attributes based on column-name and metadata "
    "heuristics — review before use."
)

# Prefixes whose `{prefix}_*` columns are treated as *direct encodings* of
# the same logical sensitive attribute (one-hot dummies, mapped labels).
# Unrelated correlated variables (zip code, income, etc.) are not listed.
DIRECT_ENCODING_PREFIXES = (
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.age",
    "demographics.gender",
    "demographics.race",
    "demographics.ethnicity",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "demo_group",
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
    substring of the column name. ``demo_group`` is included so the
    bundled synthetic demo's grouping column is a name-level candidate.

    This is not protected-class detection. Known limitations, documented
    by tests:
    - False-positive risk: names such as ``region_id``,
      ``family_history_of_cancer``, and ``device_orientation`` match
      because they contain ``region``, ``family``, or ``orientation``.
    - False-negative / unsupported vocabulary: names such as
      ``nationality``, ``citizenship``, and non-English demographic
      terms are not guaranteed to match.
    """
    return [
        col
        for col in df.columns
        if col.startswith("demographics.")
        or any(k in col.lower() for k in DEMOGRAPHIC_CATEGORIES)
    ]


def get_category(row, cols, kw):
    active = [col for col in cols if row[col] == 1]
    if len(active) > 1:
        logging.warning("Multiple active dummy columns for a single prefix.")
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

        elif len(dummy_cols) == 1:
            single_col = dummy_cols[0]
            df_new[mapped_col] = df_new[single_col].apply(
                lambda x: str(x) if pd.notnull(x) else "unknown"
            )

        mapping[keyword] = mapped_col

        if drop:
            if mapped_col in df_new.columns and df_new[mapped_col].nunique() > 1:
                df_new.drop(columns=dummy_cols, inplace=True)
            else:
                logging.warning(
                    "Skipped dropping dummy columns due to insufficient "
                    "category info."
                )

    return df_new, mapping


def encoding_prefixes_for_sensitive_column(sensitive_col: str) -> tuple[str, ...]:
    """Return dummy/source prefixes that encode the same logical attribute."""
    prefixes: list[str] = []
    if sensitive_col.endswith("_mapped"):
        prefixes.append(sensitive_col[: -len("_mapped")])
    for prefix in DIRECT_ENCODING_PREFIXES:
        if (
            sensitive_col == prefix
            or sensitive_col == f"{prefix}_mapped"
            or sensitive_col.startswith(prefix + "_")
        ):
            prefixes.append(prefix)
    unique: list[str] = []
    for prefix in prefixes:
        if prefix not in unique:
            unique.append(prefix)
    unique.sort(key=len, reverse=True)
    return tuple(unique)


def direct_columns_for_sensitive_attribute(sensitive_col: str, columns) -> list[str]:
    """
    Columns that *are* the selected sensitive attribute or a direct
    encoding of it (mapped label or one-hot source columns).

    Does not include unrelated proxy variables.
    """
    columns = list(columns)
    related = set()
    if sensitive_col in columns:
        related.add(sensitive_col)
    for prefix in encoding_prefixes_for_sensitive_column(sensitive_col):
        if prefix in columns:
            related.add(prefix)
        mapped = f"{prefix}_mapped"
        if mapped in columns:
            related.add(mapped)
        for col in columns:
            if col.startswith(prefix + "_"):
                related.add(col)
    return [col for col in columns if col in related]


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
    df, mapping = merge_dummy_columns_and_get_mapping(
        df,
        prefix_keywords=list(DIRECT_ENCODING_PREFIXES),
        drop=True,
    )

    if demographic_cols is None:
        demographic_cols = identify_by_hierarchy(df)

    candidate_cols = []
    for col in demographic_cols:
        mapped_col = mapping.get(col, col)

        if mapped_col not in df.columns:
            continue

        series = df[mapped_col]
        n_unique = series.nunique(dropna=True)
        missing_ratio = series.isna().mean()
        is_categorical = is_categorical_column(series)

        if 2 <= n_unique <= 15 and is_categorical and missing_ratio <= 0.5:
            candidate_cols.append(mapped_col)

    return df, candidate_cols
