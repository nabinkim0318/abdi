from typing import Dict

import pandas as pd


def basic_df_summary(df: pd.DataFrame) -> None:
    """
    Print basic summary statistics of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be summarized.

    Displays:
        - DataFrame info (column types, non-null counts)
        - Descriptive statistics for all columns
        - Top 30 columns with the highest missing value ratios
    """
    print("🔎 df.info():")
    print(df.info())

    print("\n📊 df.describe(include='all'):")
    print(df.describe(include="all"))

    print("\n🔥 Top 30 columns with highest missing rate:")
    print(df.isnull().mean().sort_values(ascending=False).head(30))


def summarize_categories(
    df: pd.DataFrame, recommendations: Dict[str, str]
) -> pd.DataFrame:
    """
    Summarize preprocessing recommendations by logical category
        (e.g., 'cases', 'diagnoses').

    Args:
        df (pd.DataFrame): The original dataset.
        recommendations (Dict[str, str]): A dictionary mapping column
            names to preprocessing recommendations.

    Returns:
        pd.DataFrame: A summary table containing the following columns:
            - Category: Top-level group derived from column name prefix.
            - Columns: Total number of columns in that category.
            - Drop Recommended: Count of columns flagged for removal.
            - Avg. NaN %: Average percentage of missing values.
            - OneHot Recommended: Count of columns recommended for one-hot encoding.
            - Numeric Transform: Count of columns needing numeric transformation.
    """
    PLAIN_COLUMN_NAMES = {
        "Category": "Data Type (e.g., Diagnoses, Tests)",
        "Columns": "Total Columns",
        "Drop Recommended": "Columns Suggested for Removal (⚠️ Unused or Empty)",
        "Avg. NaN %": "Average Missing Data (%)",
        "OneHot Recommended": "Columns Suggested for Grouping (Group by Category)",
        "Numeric Transform": "Columns Suggested for Numeric Adjustment "
        "(Adjust Numbers)",
    }

    summary = {}

    for col, rec in recommendations.items():
        category = col.split(".")[0] if "." in col else "project"

        # Initialize stats if category not seen yet
        if category not in summary:
            summary[category] = {
                "Columns": 0,
                "Drop Recommended": 0,
                "NaN Ratios": [],
                "OneHot Recommended": 0,
                "Numeric Transform": 0,
            }

        stats = summary[category]
        stats["Columns"] += 1

        # NaN ratio
        nan_ratio = df[col].isnull().mean() if col in df.columns else 0.0
        stats["NaN Ratios"].append(nan_ratio)

        # Drop recommendation
        if "DropColumn" in rec or "⚠️ Empty column" in rec:
            stats["Drop Recommended"] += 1

        # Encoding recommendations
        if "OneHotEncoder" in rec:
            stats["OneHot Recommended"] += 1

        # Numeric transformations
        if "MinMaxScaler" in rec or "Log1pTransform" in rec:
            stats["Numeric Transform"] += 1

    # Compile final summary table
    summary_rows = []
    for category, stats in summary.items():
        avg_nan = (
            (sum(stats["NaN Ratios"]) / len(stats["NaN Ratios"])) * 100
            if stats["NaN Ratios"]
            else 0.0
        )
        summary_rows.append(
            {
                "Category": category,
                "Columns": stats["Columns"],
                "Drop Recommended": stats["Drop Recommended"],
                "Avg. NaN %": f"{avg_nan:.1f}%",
                "OneHot Recommended": stats["OneHot Recommended"],
                "Numeric Transform": stats["Numeric Transform"],
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.rename(columns=PLAIN_COLUMN_NAMES, inplace=True)
    return df_summary
