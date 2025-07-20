import pandas as pd

from bias_audit_tool.preprocessing.recommend_columns import (
    identify_by_hierarchy,
)


def test_identify_demographic_columns():
    df = pd.DataFrame(columns=["gender", "age", "zipcode", "score", "likes_cats"])
    result = identify_by_hierarchy(df)
    # Only gender and age are in DEMOGRAPHIC_CATEGORIES, zipcode is not
    assert set(result) == {"gender", "age"}
