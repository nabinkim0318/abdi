import pandas as pd

from bias_audit_tool.utils.recommend_columns import identify_demographic_columns


def test_identify_demographic_columns():
    df = pd.DataFrame(columns=["gender", "age", "zipcode", "score", "likes_cats"])
    result = identify_demographic_columns(df)
    assert set(result) == {"gender", "age", "zipcode"}
