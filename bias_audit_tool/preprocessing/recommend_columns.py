import pandas as pd


def identify_demographic_columns(df: pd.DataFrame) -> list[str]:
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
