import pandas as pd


def identify_demographic_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify demographic-related columns based on common
    keyword patterns.

    Args:
        df (pd.DataFrame): Input dataframe whose column names
        will be scanned.

    Returns:
        list[str]: A list of column names likely related to demographics
                   (e.g., age, gender, race, income, education).
    """
    demographic_keywords = [
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
        "zip",
        "location",
        "religion",
        "orientation",
        "marital",
        "children",
        "family",
    ]

    return [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in demographic_keywords)
    ]
