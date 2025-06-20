import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from pandas.api.types import is_integer_dtype
from pandas.api.types import is_object_dtype
from scipy.stats import skew


def recommend_preprocessing(df):
    """
    Recommend preprocessing strategies for each column in a DataFrame.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: A dictionary mapping column names to recommended
              preprocessing steps.
              Recommendations include encoders, scalers, transforms,
              and missing value handling.
    """
    recommendations = {}

    for col in df.columns:
        series = df[col]
        recs = []

        # 🚫 Drop if empty or no variation
        if series.dropna().empty or series.nunique(dropna=True) == 0:
            recommendations[col] = "DropColumn"
            continue

        null_ratio = series.isnull().mean()
        nunique = series.nunique()

        # 🔠 Categorical or object columns
        if is_object_dtype(series):
            if nunique <= 10:
                recs.append("OneHotEncoder")
            elif nunique <= 50:
                recs.append("LabelEncoder")
            else:
                recs.append("LabelEncoder + ⚠️ HighCardinality")

        # 🔢 Numeric columns
        elif is_integer_dtype(series) or is_float_dtype(series):
            if nunique <= 5:
                recs.append("LabelEncoder")
            else:
                try:
                    skewness = skew(series.dropna())
                except Exception:
                    skewness = 0
                if abs(skewness) > 1:
                    recs.append("Log1pTransform")
                else:
                    recs.append("MinMaxScaler")
        else:
            recs.append("UnknownType")

        # 🧼 Missing value handling
        if null_ratio >= 0.95:
            recs.append("DropHighNaNs")
        elif null_ratio > 0.5:
            recs.append("ImputeMissing + ⚠️ ConsiderDropping")
        elif null_ratio > 0.3:
            recs.append("ImputeMissing + ⚠️ HighMissingRate")
        elif null_ratio > 0:
            recs.append("ImputeMissing")

        # Finalize
        recommendations[col] = " + ".join(recs)

    return recommendations


def summarize_categories(df: pd.DataFrame, recommendations: dict) -> pd.DataFrame:
    """
    Summarize preprocessing recommendations by column category prefix.

    Args:
        df (pd.DataFrame): Input dataset.
        recommendations (dict): Column-wise preprocessing recommendations
                                 returned by `recommend_preprocessing`.

    Returns:
        pd.DataFrame: A summary table containing:
                      - Category name (based on prefix before ".")
                      - Column count
                      - Number of columns recommended for dropping
                      - Average NaN percentage
                      - One-hot encoding recommendation count
                      - Numeric transformation recommendation count
    """
    summary = []

    for prefix in sorted(set(col.split(".")[0] for col in df.columns)):
        cat_cols = [col for col in df.columns if col.startswith(prefix)]

        count = len(cat_cols)
        drop_count = sum(
            1 for col in cat_cols if "DropColumn" in recommendations.get(col, "")
        )
        nan_ratios = [df[col].isna().mean() for col in cat_cols]
        avg_nan = round(np.mean(nan_ratios) * 100, 1) if nan_ratios else 0.0
        onehot_count = sum(
            1 for col in cat_cols if "OneHot" in recommendations.get(col, "")
        )
        numeric_count = sum(
            1
            for col in cat_cols
            if any(
                method in recommendations.get(col, "")
                for method in ["MinMaxScaler", "Log1p"]
            )
        )

        summary.append(
            {
                "Category": prefix,
                "Columns": count,
                "Drop Recommended": drop_count,
                "Avg. NaN %": f"{avg_nan}%",
                "OneHot Recommended": onehot_count,
                "Numeric Transform": numeric_count,
            }
        )

    return pd.DataFrame(summary)
