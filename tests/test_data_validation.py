import io

import numpy as np
import pandas as pd
import pytest

from bias_audit_tool.data.data_loader import load_uploaded_dataset
from bias_audit_tool.data.validation import assess_binary_class_support
from bias_audit_tool.data.validation import assess_csv_headers
from bias_audit_tool.data.validation import assess_dataset_size
from bias_audit_tool.data.validation import CODE_BLANK_HEADER
from bias_audit_tool.data.validation import CODE_DATASET_TOO_SMALL
from bias_audit_tool.data.validation import CODE_DUPLICATE_HEADERS
from bias_audit_tool.data.validation import CODE_EXTREME_CLASS_IMBALANCE
from bias_audit_tool.data.validation import CODE_INSUFFICIENT_CLASS_SUPPORT
from bias_audit_tool.data.validation import CODE_NON_FINITE_VALUES
from bias_audit_tool.data.validation import CODE_SMALL_DATASET_WARNING
from bias_audit_tool.data.validation import collect_modeling_guardrails
from bias_audit_tool.data.validation import DataValidationError
from bias_audit_tool.data.validation import describe_class_distribution
from bias_audit_tool.data.validation import EXTREME_IMBALANCE_MINORITY_FRACTION
from bias_audit_tool.data.validation import fingerprint_upload
from bias_audit_tool.data.validation import inspect_csv_header
from bias_audit_tool.data.validation import MIN_CLASS_COUNT_FOR_MODELING
from bias_audit_tool.data.validation import MIN_MODELING_ROWS
from bias_audit_tool.data.validation import SEVERITY_ERROR
from bias_audit_tool.data.validation import SEVERITY_WARNING
from bias_audit_tool.data.validation import SMALL_DATASET_WARNING_ROWS
from bias_audit_tool.data.validation import validate_finite_values
from bias_audit_tool.preprocessing.modeling_pipeline import run_modeling_pipeline


def _codes(issues):
    return [issue.code for issue in issues]


def test_header_abc_is_valid():
    header = inspect_csv_header(io.BytesIO(b"a,b,c\n1,2,3\n"))
    assert header == ["a", "b", "c"]
    assert assess_csv_headers(header) == []


def test_duplicate_header_is_rejected_before_pandas_mangling():
    raw = b"a,a,c\n1,2,3\n"
    buffer = io.BytesIO(raw)
    header = inspect_csv_header(buffer)
    assert header == ["a", "a", "c"]
    issues = assess_csv_headers(header)
    assert _codes(issues) == [CODE_DUPLICATE_HEADERS]
    assert issues[0].details["duplicate_names"] == ["a"]

    mangled = pd.read_csv(io.BytesIO(raw))
    assert list(mangled.columns) == ["a", "a.1", "c"]

    df, load_issues = load_uploaded_dataset(io.BytesIO(raw))
    assert df is None
    assert _codes(load_issues) == [CODE_DUPLICATE_HEADERS]


def test_quoted_comma_in_header_is_valid():
    raw = b'"a,b",c,d\n1,2,3\n'
    header = inspect_csv_header(io.BytesIO(raw))
    assert header == ["a,b", "c", "d"]
    assert assess_csv_headers(header) == []


def test_blank_header_is_structural_error():
    raw = b'"",x,y\n1,2,3\n'
    header = inspect_csv_header(io.BytesIO(raw))
    assert header[0] == ""
    issues = assess_csv_headers(header)
    assert CODE_BLANK_HEADER in _codes(issues)
    assert issues[0].details["blank_positions"] == [1]


def test_utf8_bom_header_parses():
    raw = b"\xef\xbb\xbfa,b,c\n1,2,3\n"
    header = inspect_csv_header(io.BytesIO(raw))
    assert header == ["a", "b", "c"]
    assert assess_csv_headers(header) == []


def test_duplicate_quoted_headers_are_rejected():
    raw = b'"age","age",outcome\n1,2,0\n'
    header = inspect_csv_header(io.BytesIO(raw))
    assert header == ["age", "age", "outcome"]
    issues = assess_csv_headers(header)
    assert _codes(issues) == [CODE_DUPLICATE_HEADERS]
    assert "age" in issues[0].details["duplicate_names"]
    assert "1,2,0" not in issues[0].message


def test_header_inspection_restores_cursor_for_pandas():
    raw = b"a,b,c\n1,2,3\n4,5,6\n"
    buffer = io.BytesIO(raw)
    inspect_csv_header(buffer)
    df = pd.read_csv(buffer)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.iloc[1].tolist() == [4, 5, 6]


def test_finite_frame_passes():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan], "y": [0, 1, 0]})
    assert validate_finite_values(df) is None


def test_positive_and_negative_infinity_are_errors():
    df = pd.DataFrame(
        {
            "lab_value": [1.0, np.inf, np.inf, 4.0],
            "score": [0.0, -np.inf, 1.0, 2.0],
        }
    )
    issue = validate_finite_values(df)
    assert issue is not None
    assert issue.code == CODE_NON_FINITE_VALUES
    assert issue.severity == SEVERITY_ERROR
    assert issue.details["columns"]["lab_value"] == 2
    assert issue.details["columns"]["score"] == 1
    assert "`lab_value`" in issue.message
    assert "`score`" in issue.message


def test_nan_is_not_confused_with_infinity():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    assert validate_finite_values(df) is None


def test_object_string_inf_is_not_flagged_unless_numeric():
    df = pd.DataFrame({"note": ["inf", "missing", "ok"]})
    assert df["note"].dtype == object
    assert validate_finite_values(df) is None

    parsed = pd.read_csv(io.StringIO("x\ninf\n1\n2\n"))
    if pd.api.types.is_numeric_dtype(parsed["x"]):
        issue = validate_finite_values(parsed)
        assert issue is not None
        assert issue.code == CODE_NON_FINITE_VALUES
    else:
        assert validate_finite_values(parsed) is None


def test_dataset_size_boundaries():
    too_small = assess_dataset_size(MIN_MODELING_ROWS - 1)
    assert _codes(too_small) == [CODE_DATASET_TOO_SMALL]
    assert too_small[0].severity == SEVERITY_ERROR

    at_min = assess_dataset_size(MIN_MODELING_ROWS)
    assert _codes(at_min) == [CODE_SMALL_DATASET_WARNING]
    assert at_min[0].severity == SEVERITY_WARNING

    just_below_warn = assess_dataset_size(SMALL_DATASET_WARNING_ROWS - 1)
    assert _codes(just_below_warn) == [CODE_SMALL_DATASET_WARNING]

    comfortable = assess_dataset_size(SMALL_DATASET_WARNING_ROWS)
    assert comfortable == []
    comfortable_large = assess_dataset_size(400)
    assert comfortable_large == []


def test_class_support_split_matrix():
    # 49/1, 48/2, 47/3 fall below MIN_CLASS_COUNT_FOR_MODELING.
    for majority, minority in ((49, 1), (48, 2), (47, 3)):
        y = pd.Series([0] * majority + [1] * minority)
        issues = assess_binary_class_support(y, target_name="outcome")
        assert CODE_INSUFFICIENT_CLASS_SUPPORT in _codes(issues)
        support = next(
            issue
            for issue in issues
            if issue.code == CODE_INSUFFICIENT_CLASS_SUPPORT
        )
        assert support.details["class_counts"]["1"] == minority

    # 95/5 meets the hard class floor when n is large enough; imbalance warns.
    y_95_5 = pd.Series([0] * 95 + [1] * 5)
    issues_95_5 = assess_binary_class_support(y_95_5)
    assert CODE_INSUFFICIENT_CLASS_SUPPORT not in _codes(issues_95_5)
    assert CODE_EXTREME_CLASS_IMBALANCE in _codes(issues_95_5)

    y_90_10 = pd.Series([0] * 90 + [1] * 10)
    issues_90_10 = assess_binary_class_support(y_90_10)
    assert CODE_INSUFFICIENT_CLASS_SUPPORT not in _codes(issues_90_10)
    assert CODE_EXTREME_CLASS_IMBALANCE in _codes(issues_90_10)


def test_imbalance_warning_thresholds():
    balanced = assess_binary_class_support(pd.Series([0] * 50 + [1] * 50))
    assert CODE_EXTREME_CLASS_IMBALANCE not in _codes(balanced)

    eighty_twenty = assess_binary_class_support(pd.Series([0] * 80 + [1] * 20))
    assert CODE_EXTREME_CLASS_IMBALANCE not in _codes(eighty_twenty)

    ninety_ten = assess_binary_class_support(pd.Series([0] * 90 + [1] * 10))
    assert CODE_EXTREME_CLASS_IMBALANCE in _codes(ninety_ten)
    assert ninety_ten[0].severity == SEVERITY_WARNING

    ninety_nine_one = assess_binary_class_support(pd.Series([0] * 99 + [1] * 1))
    assert CODE_INSUFFICIENT_CLASS_SUPPORT in _codes(ninety_nine_one)
    assert CODE_EXTREME_CLASS_IMBALANCE in _codes(ninety_nine_one)


def test_effective_rows_ignore_null_targets():
    y = pd.Series([0] * 10 + [1] * 10 + [np.nan] * 50)
    issues = collect_modeling_guardrails(y, target_name="outcome")
    assert CODE_DATASET_TOO_SMALL not in _codes(issues)
    assert CODE_SMALL_DATASET_WARNING in _codes(issues)
    size = next(
        issue for issue in issues if issue.code == CODE_SMALL_DATASET_WARNING
    )
    assert size.details["n_effective_rows"] == 20


def test_tiny_dataset_and_tiny_class_report_both_errors():
    y = pd.Series([0] * (MIN_MODELING_ROWS - 2) + [1])
    issues = collect_modeling_guardrails(y)
    assert CODE_DATASET_TOO_SMALL in _codes(issues)
    assert CODE_INSUFFICIENT_CLASS_SUPPORT in _codes(issues)


def test_class_distribution_keeps_original_labels():
    y = pd.Series(["yes"] * 12 + ["no"] * 12)
    distribution = describe_class_distribution(y)
    labels = {item["label"] for item in distribution["classes"]}
    assert labels == {"yes", "no"}
    assert distribution["n_effective_rows"] == 24


def test_pipeline_rejects_insufficient_class_support_before_sklearn(monkeypatch):
    n0, n1 = 49, 1
    df = pd.DataFrame(
        {
            "age": np.arange(n0 + n1, dtype=float),
            "gender": ["A", "B"] * ((n0 + n1) // 2) + ["A"] * ((n0 + n1) % 2),
            "outcome": [0] * n0 + [1] * n1,
        }
    )
    called = {"split": False}

    def _boom(*args, **kwargs):
        called["split"] = True
        raise AssertionError("sklearn split should not run")

    monkeypatch.setattr(
        "bias_audit_tool.preprocessing.modeling_pipeline.train_test_split",
        _boom,
    )

    with pytest.raises(DataValidationError) as exc_info:
        run_modeling_pipeline(
            raw_df=df,
            df_proc=df,
            target_col="outcome",
            sensitive_col="gender",
            include_sensitive_in_features=False,
            recommendations={"age": "MinMaxScaler", "gender": "OneHotEncoder"},
        )
    assert not called["split"]
    assert CODE_INSUFFICIENT_CLASS_SUPPORT in _codes(exc_info.value.issues)
    assert "least populated class" not in str(exc_info.value).lower()


def test_fingerprint_is_sha256_and_restores_cursor():
    raw = b"a,b\n1,2\n"
    buffer = io.BytesIO(raw)
    digest = fingerprint_upload(buffer)
    assert len(digest) == 64
    assert digest == fingerprint_upload(io.BytesIO(raw))
    df = pd.read_csv(buffer)
    assert df.shape == (1, 2)


def test_imbalance_constant_matches_documented_rule():
    assert EXTREME_IMBALANCE_MINORITY_FRACTION == 0.10
    assert MIN_CLASS_COUNT_FOR_MODELING == 5
    assert MIN_MODELING_ROWS == 20
    assert SMALL_DATASET_WARNING_ROWS == 100
