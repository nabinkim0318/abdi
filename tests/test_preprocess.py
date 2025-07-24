import pandas as pd

from bias_audit_tool.preprocessing.preprocess import recommend_preprocessing


def test_recommend_preprocessing_object_numeric_nan():
    df = pd.DataFrame(
        {
            "category": [
                "cat",
                "dog",
                "cat",
                "dog",
                "mouse",
                None,
                "dog",
                "cat",
                "dog",
                "cat",
            ],
            "numeric": list(range(10)),
            "mostly_nan": [None, None, None, 1, 2, None, None, None, None, None],
        }
    )

    recommendations = recommend_preprocessing(df)

    assert recommendations["category"] == "OneHotEncoder + ImputeMissing"
    assert recommendations["numeric"] == "MinMaxScaler"
    assert recommendations["mostly_nan"] == "LabelEncoder + DropHighNaNs"
