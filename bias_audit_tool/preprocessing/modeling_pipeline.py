# Leakage-safe preprocessing + modeling pipeline.
#
# The exploratory preprocessing in `bias_audit_tool.preprocessing.transform`
# (`apply_preprocessing`) fits its imputers/scalers/encoders on the *entire*
# uploaded dataframe, before any train/test split exists. That is fine for
# the UI's data-exploration/visualization workflow, but it must never be the
# source of the features used to train and evaluate a model: statistics and
# category vocabularies learned from rows that end up in the test split
# would then leak into the training representation.
#
# This module implements the required sequence for the modeling path
# instead:
#
#     raw selected modeling data
#     -> split train/test (single shared split, so X/y/sensitive stay
#        aligned)
#     -> fit preprocessing artifacts on TRAIN only
#     -> transform TRAIN
#     -> transform TEST using the already-fit artifacts
#     -> fit model on transformed TRAIN
#     -> evaluate on transformed TEST
#
# Column-level preprocessing choices reuse the same recommendation strings
# produced by `recommend_preprocessing` (e.g. "OneHotEncoder +
# ImputeMissing"), but the actual fitting/transforming is delegated to a
# scikit-learn `ColumnTransformer`, whose `fit`/`transform` split already
# guarantees that no learned state ever comes from data passed only to
# `transform`.
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

from bias_audit_tool.modeling.model_selector import fit_and_evaluate_model
from bias_audit_tool.modeling.target_validation import validate_classification_target
from bias_audit_tool.preprocessing.preprocess import recommend_preprocessing
from bias_audit_tool.preprocessing.recommend_columns import (
    direct_columns_for_sensitive_attribute,
)


def _safe_log1p(values):
    arr = np.asarray(values, dtype=float)
    return np.where(arr <= -1, arr, np.log1p(arr))


def _bucket_columns(X: pd.DataFrame, recommendations: dict) -> dict:
    """
    Sort feature columns into preprocessing buckets based on their
    recommendation string. Columns with no recommendation (e.g. a sensitive
    attribute added to the feature set after Step 1's recommendations were
    computed) get one derived on the fly from their own train-only values.
    """
    buckets = {
        "onehot": [],
        "ordinal": [],
        "scale": [],
        "log1p": [],
        "numeric_passthrough": [],
    }

    for col in X.columns:
        rec = recommendations.get(col)
        if rec is None:
            rec = recommend_preprocessing(X[[col]])[col]
        methods = rec.split(" + ")

        if "DropColumn" in methods or "DropHighNaNs" in methods:
            continue
        elif "OneHotEncoder" in methods:
            buckets["onehot"].append(col)
        elif "LabelEncoder" in methods:
            buckets["ordinal"].append(col)
        elif "MinMaxScaler" in methods:
            buckets["scale"].append(col)
        elif "Log1pTransform" in methods:
            buckets["log1p"].append(col)
        elif "PowerTransform" in methods:
            buckets["numeric_passthrough"].append(col)
        elif pd.api.types.is_numeric_dtype(X[col]):
            # Unmatched recommendation (e.g. "UnknownType") on an already
            # numeric column: just impute, no encoding needed.
            buckets["numeric_passthrough"].append(col)
        else:
            # Unmatched recommendation on a non-numeric column: still route
            # it through an encoder so it never reaches the model as a raw
            # string, using the same safe-fallback categorical encoding as
            # "LabelEncoder"-recommended columns.
            buckets["ordinal"].append(col)

    return buckets


def build_feature_preprocessor(
    X_train: pd.DataFrame, recommendations: dict
) -> tuple[ColumnTransformer, dict]:
    """
    Build (but do not fit) a ColumnTransformer for the modeling feature
    matrix. Bucketing is computed from `X_train` only, so an on-the-fly
    fallback recommendation (see `_bucket_columns`) never looks at test rows.
    """
    buckets = _bucket_columns(X_train, recommendations)
    transformers = []

    if buckets["onehot"]:
        transformers.append(
            (
                "onehot",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                buckets["onehot"],
            )
        )
    if buckets["ordinal"]:
        transformers.append(
            (
                "ordinal",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                buckets["ordinal"],
            )
        )
    if buckets["scale"]:
        transformers.append(
            (
                "scale",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", MinMaxScaler()),
                    ]
                ),
                buckets["scale"],
            )
        )
    if buckets["log1p"]:
        transformers.append(
            (
                "log1p",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("log1p", FunctionTransformer(_safe_log1p)),
                    ]
                ),
                buckets["log1p"],
            )
        )
    if buckets["numeric_passthrough"]:
        transformers.append(
            (
                "numeric_passthrough",
                SimpleImputer(strategy="median"),
                buckets["numeric_passthrough"],
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.set_output(transform="pandas")
    return preprocessor, buckets


def split_modeling_frame(
    X_raw: pd.DataFrame,
    y: pd.Series,
    sensitive: pd.Series = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Single index-based train/test split shared across features, target, and
    the sensitive-attribute vector, so the three stay aligned by
    construction instead of relying on independent random splits.
    """
    stratify = y if y.value_counts().min() >= 2 else None
    idx_train, idx_test = train_test_split(
        X_raw.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    sens_train = sensitive.loc[idx_train] if sensitive is not None else None
    sens_test = sensitive.loc[idx_test] if sensitive is not None else None

    return (
        X_raw.loc[idx_train],
        X_raw.loc[idx_test],
        y.loc[idx_train],
        y.loc[idx_test],
        sens_train,
        sens_test,
    )


@dataclass
class ModelingPipelineResult:
    model: object
    report: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_prob: np.ndarray
    sensitive_test: pd.Series
    feature_importance: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    fitted_preprocessor: ColumnTransformer


def run_modeling_pipeline(
    raw_df: pd.DataFrame,
    df_proc: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    include_sensitive_in_features: bool,
    recommendations: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelingPipelineResult:
    """
    Run the leakage-safe modeling path end to end.

    `raw_df` is the original uploaded dataframe (before any globally-fit
    exploratory preprocessing). `df_proc` supplies the already-cleaned row
    set (duplicates dropped) and, when the sensitive attribute is a merged
    demographic column (e.g. `demographic.gender_mapped`), its human-
    readable per-row values for fairness grouping. Neither the imputation,
    scaling, nor encoding used to build the model's feature matrix comes
    from `df_proc` — those are fit fresh on the training split only.
    """
    if target_col not in raw_df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found in the uploaded dataset."
        )

    aligned_index = (
        df_proc.index.intersection(raw_df.index)
        if df_proc is not None
        else raw_df.index
    )
    working_df = raw_df.loc[aligned_index]

    y = working_df[target_col]
    non_null_mask = y.notna()
    working_df = working_df.loc[non_null_mask]
    y = y.loc[non_null_mask]

    validate_classification_target(y, target_name=target_col)

    X_raw = working_df.drop(columns=[target_col])

    sensitive_series = None
    if sensitive_col is not None:
        if df_proc is not None and sensitive_col in df_proc.columns:
            sensitive_series = df_proc.loc[working_df.index, sensitive_col]
        elif sensitive_col in working_df.columns:
            sensitive_series = working_df[sensitive_col]
        else:
            raise ValueError(
                f"Sensitive column '{sensitive_col}' was not found in either "
                "the uploaded dataset or the processed dataframe."
            )

        if include_sensitive_in_features:
            X_raw = X_raw.copy()
            if sensitive_col not in X_raw.columns:
                X_raw[sensitive_col] = sensitive_series.reindex(X_raw.index)
        else:
            drop_cols = direct_columns_for_sensitive_attribute(
                sensitive_col, X_raw.columns
            )
            if drop_cols:
                X_raw = X_raw.drop(columns=drop_cols)

    X_train_raw, X_test_raw, y_train, y_test, sens_train, sens_test = (
        split_modeling_frame(
            X_raw,
            y,
            sensitive_series,
            test_size=test_size,
            random_state=random_state,
        )
    )

    if y_train.dtype == "object" or isinstance(y_train.iloc[0], str):
        target_encoder = LabelEncoder().fit(y_train)
        y_train = pd.Series(
            target_encoder.transform(y_train), index=y_train.index, name=y_train.name
        )
        y_test = pd.Series(
            target_encoder.transform(y_test), index=y_test.index, name=y_test.name
        )

    preprocessor, _ = build_feature_preprocessor(X_train_raw, recommendations)
    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    results = fit_and_evaluate_model(X_train, y_train, X_test, y_test)

    return ModelingPipelineResult(
        model=results["model"],
        report=results["report"],
        y_test=results["y_test"],
        y_pred=results["y_pred"],
        y_prob=results["y_prob"],
        sensitive_test=sens_test,
        feature_importance=results["feature_importance"],
        X_train=X_train,
        X_test=X_test,
        fitted_preprocessor=preprocessor,
    )
