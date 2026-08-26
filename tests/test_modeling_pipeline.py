import numpy as np
import pandas as pd
import pytest

from bias_audit_tool.modeling.target_validation import UnsupportedTargetError
from bias_audit_tool.preprocessing.modeling_pipeline import (
    build_feature_preprocessor,
)
from bias_audit_tool.preprocessing.modeling_pipeline import run_modeling_pipeline
from bias_audit_tool.preprocessing.modeling_pipeline import split_modeling_frame


# ---------------------------------------------------------------------------
# ABDI-PREP-001: preprocessing must be fit on train rows only
# ---------------------------------------------------------------------------


def test_scaler_is_fit_on_train_rows_only_not_leaked_by_test_extreme_value():
    # If MinMaxScaler were fit on train+test combined, the test-only value
    # of 1000 would become the new max, and the largest TRAIN value (8)
    # would transform to something well below 1.0.
    X_train = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    X_test = pd.DataFrame({"num": [1000.0]})
    recommendations = {"num": "MinMaxScaler"}

    preprocessor, buckets = build_feature_preprocessor(X_train, recommendations)
    assert buckets["scale"] == ["num"]

    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Train's own max still maps to exactly 1.0 -> the fitted range came
    # from train only, unaffected by the test row.
    assert X_train_t.iloc[-1, 0] == pytest.approx(1.0)
    # The unseen extreme test value falls far outside the learned [0, 1]
    # range instead of being folded in as the new maximum.
    assert X_test_t.iloc[0, 0] > 5.0


def test_onehot_categories_are_fit_on_train_rows_only():
    X_train = pd.DataFrame({"cat": ["a", "b", "a", "b", "a", "b"]})
    X_test = pd.DataFrame({"cat": ["c"]})  # category never seen in train
    recommendations = {"cat": "OneHotEncoder"}

    preprocessor, buckets = build_feature_preprocessor(X_train, recommendations)
    assert buckets["onehot"] == ["cat"]

    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Only categories observed in train produce dummy columns.
    feature_names = set(preprocessor.get_feature_names_out())
    assert feature_names == {"onehot__cat_a", "onehot__cat_b"}
    assert set(X_train_t.columns) == feature_names

    # The unseen test-only category maps to all-zero rather than minting a
    # new column or being folded into the fitted category set.
    assert (X_test_t.iloc[0] == 0).all()


def test_median_imputer_is_fit_on_train_rows_only():
    # Median of the non-null train values is 3.0. If a test-only extreme
    # value (never passed to `.fit`) leaked into the imputer's statistic,
    # the median used to fill train's own missing value would shift away
    # from 3.0.
    X_train = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]})
    recommendations = {"num": "MinMaxScaler + ImputeMissing"}

    preprocessor, _ = build_feature_preprocessor(X_train, recommendations)
    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)

    # median(1,2,3,4,5) = 3.0 -> scaled with train's own min/max (1..5)
    # gives (3-1)/(5-1) = 0.5, regardless of the extreme test value.
    assert X_train_t.iloc[-1, 0] == pytest.approx(0.5)


def test_modeling_pipeline_end_to_end_does_not_leak_into_test_features():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n).astype(float),
            "gender": pd.Series(rng.choice(["male", "female"], n), dtype="object"),
            "score": rng.normal(50, 10, n),
            "diagnosis": pd.Series(rng.choice(["yes", "no"], n), dtype="object"),
        }
    )
    recommendations = {
        "age": "MinMaxScaler",
        "gender": "OneHotEncoder",
        "score": "MinMaxScaler",
    }

    result = run_modeling_pipeline(
        raw_df=df,
        df_proc=df,
        target_col="diagnosis",
        sensitive_col="gender",
        include_sensitive_in_features=False,
        recommendations=recommendations,
    )

    # The scaler inside the fitted preprocessor must have learned its
    # min/max from the training rows only -- not from `df["age"]` as a
    # whole, which would be the case if it had been fit before the split
    # (as in the original `apply_preprocessing` exploratory pipeline).
    scale_step = result.fitted_preprocessor.named_transformers_["scale"]
    fitted_scaler = scale_step.named_steps["scale"]
    train_age = df.loc[result.X_train.index, "age"]
    assert fitted_scaler.data_max_[0] == pytest.approx(train_age.max())
    assert fitted_scaler.data_min_[0] == pytest.approx(train_age.min())


# ---------------------------------------------------------------------------
# ABDI-SENS-001: explicit, configurable sensitive-feature inclusion
# ---------------------------------------------------------------------------


def _demo_frame(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, n).astype(float),
            "gender": pd.Series(rng.choice(["male", "female"], n), dtype="object"),
            "outcome": pd.Series(rng.choice(["yes", "no"], n), dtype="object"),
        }
    )


def test_excluded_sensitive_column_does_not_enter_feature_matrix():
    df = _demo_frame()
    recommendations = {"age": "MinMaxScaler", "gender": "OneHotEncoder"}

    result = run_modeling_pipeline(
        raw_df=df,
        df_proc=df,
        target_col="outcome",
        sensitive_col="gender",
        include_sensitive_in_features=False,
        recommendations=recommendations,
    )

    assert not any("gender" in col for col in result.X_train.columns)
    assert not any("gender" in col for col in result.X_test.columns)


def _onehot_race_frame(n=120, seed=3):
    rng = np.random.default_rng(seed)
    race = rng.choice(["Black", "White", "Asian"], n)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, n).astype(float),
            "race_Black": (race == "Black").astype(int),
            "race_White": (race == "White").astype(int),
            "race_Asian": (race == "Asian").astype(int),
            "zipcode": rng.integers(10000, 99999, n),
            "income": rng.integers(20, 120, n).astype(float),
            "outcome": pd.Series(rng.choice([0, 1], n), dtype="int64"),
        }
    )


def test_exclude_drops_direct_onehot_source_columns_of_mapped_race():
    raw_df = _onehot_race_frame()
    df_proc = raw_df.copy()
    df_proc["race_mapped"] = np.select(
        [raw_df["race_Black"] == 1, raw_df["race_White"] == 1],
        ["Black", "White"],
        default="Asian",
    )
    recommendations = {
        "age": "MinMaxScaler",
        "zipcode": "MinMaxScaler",
        "income": "MinMaxScaler",
    }

    result = run_modeling_pipeline(
        raw_df=raw_df,
        df_proc=df_proc,
        target_col="outcome",
        sensitive_col="race_mapped",
        include_sensitive_in_features=False,
        recommendations=recommendations,
    )

    feature_names = list(result.X_train.columns)
    assert not any("race_Black" in col for col in feature_names)
    assert not any("race_White" in col for col in feature_names)
    assert not any("race_Asian" in col for col in feature_names)
    assert not any("race_mapped" in col for col in feature_names)
    assert any("zipcode" in col for col in feature_names)
    assert any("income" in col for col in feature_names)
    expected = df_proc.loc[result.y_test.index, "race_mapped"]
    pd.testing.assert_series_equal(
        result.sensitive_test.sort_index(), expected.sort_index(), check_names=False
    )


def test_include_keeps_direct_onehot_source_columns_of_mapped_race():
    raw_df = _onehot_race_frame()
    df_proc = raw_df.copy()
    df_proc["race_mapped"] = np.select(
        [raw_df["race_Black"] == 1, raw_df["race_White"] == 1],
        ["Black", "White"],
        default="Asian",
    )
    recommendations = {
        "age": "MinMaxScaler",
        "zipcode": "MinMaxScaler",
        "income": "MinMaxScaler",
    }

    result = run_modeling_pipeline(
        raw_df=raw_df,
        df_proc=df_proc,
        target_col="outcome",
        sensitive_col="race_mapped",
        include_sensitive_in_features=True,
        recommendations=recommendations,
    )

    feature_names = list(result.X_train.columns)
    assert any("race_Black" in col for col in feature_names)
    assert any("race_White" in col for col in feature_names)
    assert any("race_Asian" in col for col in feature_names)
    assert any("zipcode" in col for col in feature_names)
    expected = df_proc.loc[result.y_test.index, "race_mapped"]
    pd.testing.assert_series_equal(
        result.sensitive_test.sort_index(), expected.sort_index(), check_names=False
    )


def test_included_sensitive_column_enters_feature_matrix():
    df = _demo_frame()
    recommendations = {"age": "MinMaxScaler", "gender": "OneHotEncoder"}

    result = run_modeling_pipeline(
        raw_df=df,
        df_proc=df,
        target_col="outcome",
        sensitive_col="gender",
        include_sensitive_in_features=True,
        recommendations=recommendations,
    )

    assert any("gender" in col for col in result.X_train.columns)
    assert any("gender" in col for col in result.X_test.columns)


@pytest.mark.parametrize("include_sensitive", [False, True])
def test_fairness_grouping_uses_correct_held_out_sensitive_values(
    include_sensitive,
):
    df = _demo_frame()
    recommendations = {"age": "MinMaxScaler", "gender": "OneHotEncoder"}

    result = run_modeling_pipeline(
        raw_df=df,
        df_proc=df,
        target_col="outcome",
        sensitive_col="gender",
        include_sensitive_in_features=include_sensitive,
        recommendations=recommendations,
    )

    # The sensitive vector handed back for fairness grouping must be the
    # ORIGINAL, human-readable held-out values for exactly the test rows,
    # regardless of whether gender was also used as a feature.
    expected = df.loc[result.y_test.index, "gender"]
    pd.testing.assert_series_equal(
        result.sensitive_test.sort_index(),
        expected.sort_index(),
        check_names=False,
    )


def test_sensitive_column_merged_only_in_df_proc_is_sourced_correctly():
    # Mirrors the real app: `group_col` (e.g. a demographic column merged
    # from one-hot dummies by `recommend_demographic_columns`) may not
    # exist under that name in the raw upload at all, only in `df_proc`.
    # `df_proc` may also have fewer rows (duplicates dropped) with a
    # preserved (non-reset) index into the raw data.
    rng = np.random.default_rng(5)
    n = 200
    raw_df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n).astype(float),
            "demographic.gender_male": rng.integers(0, 2, n),
            "demographic.gender_female": 0,
            "outcome": pd.Series(rng.choice(["yes", "no"], n), dtype="object"),
        }
    )
    raw_df["demographic.gender_female"] = 1 - raw_df["demographic.gender_male"]

    # df_proc: same rows minus a dropped-duplicate tail, plus a merged
    # "demographic.gender_mapped" column that doesn't exist in raw_df.
    df_proc = raw_df.iloc[:-10].copy()
    df_proc["demographic.gender_mapped"] = df_proc["demographic.gender_male"].map(
        {1: "male", 0: "female"}
    )

    recommendations = {"age": "MinMaxScaler"}

    result = run_modeling_pipeline(
        raw_df=raw_df,
        df_proc=df_proc,
        target_col="outcome",
        sensitive_col="demographic.gender_mapped",
        include_sensitive_in_features=False,
        recommendations=recommendations,
    )

    # Grouping values must be the human-readable df_proc-sourced labels,
    # not raw 0/1 dummy columns, and must line up with the held-out rows.
    assert set(result.sensitive_test.unique()) <= {"male", "female"}
    expected = df_proc.loc[result.y_test.index, "demographic.gender_mapped"]
    pd.testing.assert_series_equal(
        result.sensitive_test.sort_index(), expected.sort_index(), check_names=False
    )
    # The excluded logical attribute includes the mapped grouping column
    # and its direct one-hot source columns in the raw upload.
    feature_names = list(result.X_train.columns) + list(result.X_test.columns)
    assert not any("gender_mapped" in col for col in feature_names)
    assert not any("gender_male" in col for col in feature_names)
    assert not any("gender_female" in col for col in feature_names)


def test_include_keeps_direct_source_columns_of_mapped_demographic_gender():
    rng = np.random.default_rng(6)
    n = 200
    raw_df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n).astype(float),
            "demographic.gender_male": rng.integers(0, 2, n),
            "outcome": pd.Series(rng.choice(["yes", "no"], n), dtype="object"),
        }
    )
    raw_df["demographic.gender_female"] = 1 - raw_df["demographic.gender_male"]
    df_proc = raw_df.copy()
    df_proc["demographic.gender_mapped"] = df_proc["demographic.gender_male"].map(
        {1: "male", 0: "female"}
    )

    result = run_modeling_pipeline(
        raw_df=raw_df,
        df_proc=df_proc,
        target_col="outcome",
        sensitive_col="demographic.gender_mapped",
        include_sensitive_in_features=True,
        recommendations={"age": "MinMaxScaler"},
    )

    feature_names = list(result.X_train.columns)
    assert any("gender_male" in col for col in feature_names)
    assert any("gender_female" in col for col in feature_names)
    expected = df_proc.loc[result.y_test.index, "demographic.gender_mapped"]
    pd.testing.assert_series_equal(
        result.sensitive_test.sort_index(), expected.sort_index(), check_names=False
    )


# ---------------------------------------------------------------------------
# Train/test/sensitive alignment guarantees
# ---------------------------------------------------------------------------


def test_split_modeling_frame_keeps_features_target_and_sensitive_aligned():
    n = 100
    X = pd.DataFrame({"f": range(n)}, index=[f"row_{i}" for i in range(n)])
    y = pd.Series([i % 2 for i in range(n)], index=X.index)
    sensitive = pd.Series([f"group_{i % 3}" for i in range(n)], index=X.index)

    X_train, X_test, y_train, y_test, sens_train, sens_test = split_modeling_frame(
        X, y, sensitive, test_size=0.25, random_state=7
    )

    # Same rows, same order, across every returned piece of the test split.
    assert list(X_test.index) == list(y_test.index) == list(sens_test.index)
    assert list(X_train.index) == list(y_train.index) == list(sens_train.index)

    # Each sensitive value still corresponds to the row it was assigned to
    # originally -- this would fail if the sensitive vector were split with
    # an independent, unshared random call (a differently-shuffled split
    # would misalign group labels with roughly 2/3 probability per row for
    # a 3-group label set).
    for idx in X_test.index:
        assert sens_test.loc[idx] == sensitive.loc[idx]
        assert y_test.loc[idx] == y.loc[idx]


def test_pipeline_result_predictions_align_with_sensitive_test_for_fairness():
    df = _demo_frame(n=300, seed=3)
    recommendations = {"age": "MinMaxScaler", "gender": "OneHotEncoder"}

    result = run_modeling_pipeline(
        raw_df=df,
        df_proc=df,
        target_col="outcome",
        sensitive_col="gender",
        include_sensitive_in_features=False,
        recommendations=recommendations,
    )

    assert len(result.y_test) == len(result.y_pred) == len(result.sensitive_test)
    assert list(result.sensitive_test.index) == list(result.y_test.index)


# ---------------------------------------------------------------------------
# ABDI-MODEL-001: target validation is wired into the modeling entry point
# ---------------------------------------------------------------------------


def test_pipeline_rejects_continuous_target_before_training():
    rng = np.random.default_rng(4)
    n = 150
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n).astype(float),
            "gender": pd.Series(rng.choice(["male", "female"], n), dtype="object"),
            "lab_value": rng.normal(size=n),
        }
    )
    recommendations = {"age": "MinMaxScaler", "gender": "OneHotEncoder"}

    with pytest.raises(UnsupportedTargetError) as exc_info:
        run_modeling_pipeline(
            raw_df=df,
            df_proc=df,
            target_col="lab_value",
            sensitive_col="gender",
            include_sensitive_in_features=False,
            recommendations=recommendations,
        )
    assert exc_info.value.reason == "continuous_target"


def test_pipeline_rejects_near_unique_id_target_before_training():
    n = 150
    df = pd.DataFrame(
        {
            "age": range(n),
            "gender": pd.Series(["male", "female"] * (n // 2), dtype="object"),
            "patient_id": [f"P{i}" for i in range(n)],
        }
    )
    recommendations = {"age": "MinMaxScaler", "gender": "OneHotEncoder"}

    with pytest.raises(UnsupportedTargetError) as exc_info:
        run_modeling_pipeline(
            raw_df=df,
            df_proc=df,
            target_col="patient_id",
            sensitive_col="gender",
            include_sensitive_in_features=False,
            recommendations=recommendations,
        )
    assert exc_info.value.reason == "near_unique_target"
