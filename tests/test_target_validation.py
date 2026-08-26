import numpy as np
import pandas as pd
import pytest

from bias_audit_tool.modeling.target_validation import preferred_target_column
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


def test_preferred_target_keeps_current_selection():
    df = pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )
    assert preferred_target_column(df, current_selection="score") == "score"


def test_preferred_target_picks_first_binary_column_not_first_column():
    df = pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "flag": [0, 1, 0, 1, 0, 1],
            "other": [1, 0, 1, 0, 1, 0],
        }
    )
    assert preferred_target_column(df) == "flag"


def test_preferred_target_skips_deprioritized_binary_columns():
    df = pd.DataFrame(
        {
            "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "group": ["A", "B", "A", "B", "A", "B"],
            "label": [0, 1, 0, 1, 0, 1],
        }
    )
    assert preferred_target_column(df, deprioritized=["group"]) == "label"


def test_preferred_target_falls_back_to_first_column_when_none_are_binary():
    df = pd.DataFrame(
        {
            "a": [0.1, 0.2, 0.3, 0.4],
            "b": [1.1, 1.2, 1.3, 1.4],
        }
    )
    assert preferred_target_column(df) == "a"


def test_unsupported_targets_do_not_call_class_distribution_renderer():
    from bias_audit_tool.utils.ui_helpers import present_supported_class_distribution

    continuous = pd.Series(np.random.default_rng(0).normal(size=200))
    near_unique = pd.Series([f"patient_{i}" for i in range(200)])
    for y, name, reason in (
        (continuous, "lab_value", "continuous_target"),
        (near_unique, "patient_id", "near_unique_target"),
    ):
        called = []

        def _renderer(series, target_name, _called=called):
            _called.append((list(series.head(3)), target_name))

        with pytest.raises(UnsupportedTargetError) as exc_info:
            present_supported_class_distribution(y, name, renderer=_renderer)
        assert exc_info.value.reason == reason
        assert called == []


def test_binary_target_calls_class_distribution_renderer_after_validation():
    from bias_audit_tool.utils.ui_helpers import present_supported_class_distribution

    y = pd.Series([0, 1, 0, 1, 1, 0])
    called = []

    def _renderer(series, target_name):
        called.append(target_name)
        assert set(series.unique()) <= {0, 1}

    present_supported_class_distribution(y, "outcome", renderer=_renderer)
    assert called == ["outcome"]
