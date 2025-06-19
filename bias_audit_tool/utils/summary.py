from typing import Dict

import pandas as pd


def summarize_categories(
    df: pd.DataFrame, recommendations: Dict[str, str]
) -> pd.DataFrame:
    """
    Summarize preprocessing recommendations by category (e.g., 'cases', 'diagnoses').

    Returns a summary DataFrame with:
    - Category name
    - Number of columns
    - Columns recommended for removal
    - Average NaN %
    - OneHotEncoder recommended count
    - Numeric transformation recommended count
    """

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

    return pd.DataFrame(summary_rows)
