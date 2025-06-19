import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from pandas.api.types import is_integer_dtype
from pandas.api.types import is_object_dtype
from scipy.stats import skew


def recommend_preprocessing(df: pd.DataFrame) -> dict:
    """
    Recommend preprocessing for each column based on type,
    uniqueness, skew, and NaN rate.
    """
    recommendations = {}

    for col in df.columns:
        series = df[col]
        recs = []

        # ⚠️ Empty column
        if series.dropna().empty:
            recommendations[col] = "⚠️ Empty column - consider removing"
            continue

        null_ratio = series.isnull().mean()
        nunique = series.nunique()

        # 📊 Type-based strategy
        if is_object_dtype(series):
            if nunique <= 10:
                recs.append("OneHotEncoder")
            else:
                recs.append("LabelEncoder")

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
            recs.append("Unknown")

        # ❗ NaN Handling
        if null_ratio > 0.3:
            recs.append("DropHighNaNs")
        elif 0 < null_ratio <= 0.3:
            recs.append("ImputeMissing")

        recommendations[col] = " + ".join(recs)

    return recommendations


def summarize_categories(df: pd.DataFrame, recommendations: dict) -> pd.DataFrame:
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
