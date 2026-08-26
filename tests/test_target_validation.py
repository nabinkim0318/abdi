import numpy as np
import pandas as pd
import pytest

from bias_audit_tool.modeling.target_validation import UnsupportedTargetError
from bias_audit_tool.modeling.target_validation import validate_classification_target


def test_binary_integer_target_is_accepted():
    y = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
    result = validate_classification_target(y, target_name="outcome")
    assert result.kind == "binary"
    assert result.n_classes == 2


def test_binary_string_target_is_accepted():
    y = pd.Series(["yes", "no", "yes", "no", "no", "yes"])
    result = validate_classification_target(y, target_name="outcome")
    assert result.kind == "binary"
    assert result.n_classes == 2


def test_continuous_float_target_is_rejected():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=200))
    with pytest.raises(UnsupportedTargetError) as exc_info:
        validate_classification_target(y, target_name="lab_value")
    assert exc_info.value.reason == "continuous_target"


def test_near_unique_string_id_target_is_rejected():
    y = pd.Series([f"patient_{i}" for i in range(200)])
    with pytest.raises(UnsupportedTargetError) as exc_info:
        validate_classification_target(y, target_name="patient_id")
    assert exc_info.value.reason == "near_unique_target"


def test_near_unique_integer_id_target_is_rejected():
    y = pd.Series(range(1, 501))
    with pytest.raises(UnsupportedTargetError) as exc_info:
        validate_classification_target(y, target_name="patient_id")
    assert exc_info.value.reason == "near_unique_target"


def test_multiclass_target_is_explicitly_rejected():
    y = pd.Series(["A", "B", "C"] * 30)
    with pytest.raises(UnsupportedTargetError) as exc_info:
        validate_classification_target(y, target_name="diagnosis")
    assert exc_info.value.reason == "multiclass_unsupported"
    assert "binary" in str(exc_info.value).lower()


def test_single_class_target_is_rejected():
    y = pd.Series([1, 1, 1, 1])
    with pytest.raises(UnsupportedTargetError) as exc_info:
        validate_classification_target(y, target_name="constant")
    assert exc_info.value.reason == "too_few_classes"


def test_binary_target_with_missing_values_is_accepted():
    y = pd.Series([0, 1, np.nan, 1, 0, np.nan, 1, 0])
    result = validate_classification_target(y, target_name="outcome")
    assert result.kind == "binary"
    assert result.n_samples == 6
