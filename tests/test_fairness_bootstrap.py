import inspect

import numpy as np
import pytest
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference

from bias_audit_tool.modeling.fairness import assess_group_support
from bias_audit_tool.modeling.fairness import bootstrap_fairness_metric
from bias_audit_tool.modeling.fairness import bootstrap_output_fairness
from bias_audit_tool.modeling.fairness import compute_group_support
from bias_audit_tool.modeling.fairness import compute_output_fairness
from bias_audit_tool.modeling.fairness import DEFAULT_BOOTSTRAP_RANDOM_STATE
from bias_audit_tool.modeling.fairness import DEFAULT_CONFIDENCE_LEVEL
from bias_audit_tool.modeling.fairness import INSUFFICIENT_VALID_BOOTSTRAP_REASON
from bias_audit_tool.modeling.fairness import WARNING_HIGH_INVALID_BOOTSTRAP
from bias_audit_tool.modeling.fairness import WARNING_SPARSE_BOOTSTRAP
from bias_audit_tool.modeling.fairness import WARNING_ZERO_POSITIVE_LABELS


def _all_selected_equal_groups(n_per_group=80):
    sensitive = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    y_true = np.tile([1, 0], n_per_group)
    y_pred = np.ones(len(y_true), dtype=int)
    return y_true, y_pred, sensitive


def _clear_dp_disparity(n_per_group=60):
    # A is always selected; B is never selected. Both classes present.
    sensitive = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    y_true = np.tile([1, 0], n_per_group)
    y_pred = np.array([1] * n_per_group + [0] * n_per_group)
    return y_true, y_pred, sensitive


def _variable_selection_rates():
    # A: 75% selected, B: 25% selected, both classes present in labels.
    sensitive = np.array(["A"] * 40 + ["B"] * 40)
    y_true = np.array([1, 1, 0, 0] * 20)
    y_pred = np.array([1, 1, 1, 0] * 10 + [1, 0, 0, 0] * 10)
    return y_true, y_pred, sensitive


def test_bootstrap_is_deterministic_for_the_same_seed():
    y_true, y_pred, sensitive = _variable_selection_rates()
    kwargs = dict(
        n_bootstrap=200,
        random_state=DEFAULT_BOOTSTRAP_RANDOM_STATE,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    )
    first = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        demographic_parity_difference,
        **kwargs,
    )
    second = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        demographic_parity_difference,
        **kwargs,
    )
    assert first["status"] == "ok"
    assert first["n_valid"] == second["n_valid"]
    assert first["ci_lower"] == pytest.approx(second["ci_lower"], rel=0, abs=1e-12)
    assert first["ci_upper"] == pytest.approx(second["ci_upper"], rel=0, abs=1e-12)


def test_equal_predictions_have_zero_dp_and_degenerate_or_near_zero_ci():
    y_true, y_pred, sensitive = _all_selected_equal_groups()
    result = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        demographic_parity_difference,
        n_bootstrap=200,
        random_state=0,
    )
    expected = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive
    )
    assert expected == pytest.approx(0.0)
    assert result["status"] == "ok"
    assert result["estimate"] == pytest.approx(expected)
    assert result["ci_lower"] == pytest.approx(0.0, abs=1e-12)
    assert result["ci_upper"] == pytest.approx(0.0, abs=1e-12)
    assert result["n_valid"] == 200
    assert result["n_requested"] == 200


def test_clear_dp_disparity_point_estimate_and_ci_are_numeric():
    y_true, y_pred, sensitive = _clear_dp_disparity()
    _, summary = compute_output_fairness(y_true, y_pred, sensitive)
    result = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        demographic_parity_difference,
        n_bootstrap=200,
        random_state=42,
    )
    expected = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive
    )
    assert expected == pytest.approx(1.0)
    assert summary["Demographic Parity Difference"] == pytest.approx(expected)
    assert result["status"] == "ok"
    assert result["estimate"] == pytest.approx(expected)
    assert result["ci_lower"] == pytest.approx(1.0, abs=1e-12)
    assert result["ci_upper"] == pytest.approx(1.0, abs=1e-12)
    assert result["ci_lower"] <= result["ci_upper"]


def test_variable_dp_ci_is_finite_and_ordered():
    y_true, y_pred, sensitive = _variable_selection_rates()
    result = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        demographic_parity_difference,
        n_bootstrap=200,
        random_state=42,
    )
    expected = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive
    )
    assert result["status"] == "ok"
    assert result["estimate"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert np.isfinite(result["ci_lower"])
    assert np.isfinite(result["ci_upper"])
    assert result["ci_lower"] <= result["ci_upper"]
    assert 0.0 <= result["ci_lower"] <= 1.0
    assert 0.0 <= result["ci_upper"] <= 1.0


def test_bootstrap_supports_three_or_more_groups():
    sensitive = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
    y_true = np.tile([1, 0], 60)
    y_pred = np.array([1] * 40 + [0] * 40 + [1, 0] * 20)
    result = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        demographic_parity_difference,
        n_bootstrap=200,
        random_state=7,
    )
    expected = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive
    )
    assert expected == pytest.approx(1.0)
    assert result["status"] == "ok"
    assert result["estimate"] == pytest.approx(expected)
    assert result["n_valid"] == result["n_requested"]


def test_sparse_groups_do_not_crash_and_report_valid_replicate_count():
    # One tiny group; resampling can drop it without crashing the audit.
    sensitive = np.array(["tiny"] * 2 + ["large"] * 80)
    y_true = np.array([1, 0] + [1, 0] * 40)
    y_pred = np.array([1, 0] + [1, 1, 0, 0] * 20)
    result = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        demographic_parity_difference,
        n_bootstrap=200,
        random_state=42,
    )
    assert result["status"] in {"ok", "undefined"}
    assert 0 <= result["n_valid"] <= result["n_requested"]
    assert result["n_requested"] == 200
    codes = [item["code"] for item in result.get("warnings") or []]
    assert WARNING_SPARSE_BOOTSTRAP in codes


def test_insufficient_valid_replicates_returns_structured_unavailable_state():
    y_true, y_pred, sensitive = _all_selected_equal_groups(n_per_group=20)
    result = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        demographic_parity_difference,
        n_bootstrap=50,
        random_state=42,
    )
    # Default rule: max(100, 50% of 50) = 100, so 50 valid draws are not enough.
    assert result["status"] == "undefined"
    assert result["reason"] == INSUFFICIENT_VALID_BOOTSTRAP_REASON
    assert result["valid_bootstrap_replicates"] == 50
    assert result["requested_bootstrap_replicates"] == 50
    assert result["n_valid"] == 50
    assert result["n_requested"] == 50
    assert result["ci_lower"] is None
    assert result["ci_upper"] is None
    assert result["estimate"] == pytest.approx(0.0)


def test_eo_zero_positive_labels_are_handled_explicitly():
    sensitive = np.array(["A"] * 40 + ["B"] * 40)
    y_true = np.array([0] * 40 + [1, 0] * 20)
    y_pred = np.array([1, 0] * 40)
    result = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        equalized_odds_difference,
        n_bootstrap=200,
        random_state=42,
        require_eo_class_support=True,
    )
    # Group A has no positive labels in the original held-out arrays, so
    # every replicate is invalid for Equalized Odds.
    assert result["n_valid"] == 0
    assert result["status"] == "undefined"
    assert result["reason"] == INSUFFICIENT_VALID_BOOTSTRAP_REASON
    assert result["ci_lower"] is None
    codes = [item["code"] for item in result.get("warnings") or []]
    assert WARNING_SPARSE_BOOTSTRAP in codes
    assert WARNING_HIGH_INVALID_BOOTSTRAP in codes
    support_codes = [
        item["code"]
        for item in assess_group_support(
            compute_group_support(y_true, y_pred, sensitive)
        )
    ]
    assert WARNING_ZERO_POSITIVE_LABELS in support_codes


def test_eo_zero_negative_labels_are_handled_explicitly():
    sensitive = np.array(["A"] * 40 + ["B"] * 40)
    y_true = np.array([1] * 40 + [1, 0] * 20)
    y_pred = np.array([1, 0] * 40)
    result = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        equalized_odds_difference,
        n_bootstrap=200,
        random_state=42,
        require_eo_class_support=True,
    )
    assert result["n_valid"] == 0
    assert result["status"] == "undefined"
    assert result["reason"] == INSUFFICIENT_VALID_BOOTSTRAP_REASON


def test_eo_point_estimate_matches_fairlearn_when_ci_is_available():
    n_per_group = 80
    sensitive = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    y_true = np.tile([1, 0], n_per_group)
    y_pred = y_true.copy()
    result = bootstrap_fairness_metric(
        y_true,
        y_pred,
        sensitive,
        equalized_odds_difference,
        n_bootstrap=200,
        random_state=1,
        require_eo_class_support=True,
    )
    expected = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive
    )
    assert expected == pytest.approx(0.0)
    assert result["status"] == "ok"
    assert result["estimate"] == pytest.approx(expected)
    assert result["ci_lower"] == pytest.approx(0.0, abs=1e-12)
    assert result["ci_upper"] == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_output_fairness_estimates_match_compute_output_fairness():
    y_true, y_pred, sensitive = _variable_selection_rates()
    _, summary = compute_output_fairness(y_true, y_pred, sensitive)
    bundled = bootstrap_output_fairness(
        y_true,
        y_pred,
        sensitive,
        n_bootstrap=200,
        random_state=42,
    )
    dp = bundled["Demographic Parity Difference"]
    eo = bundled["Equalized Odds Difference"]
    assert dp["status"] == "ok"
    assert eo["status"] == "ok"
    assert dp["estimate"] == pytest.approx(summary["Demographic Parity Difference"])
    assert eo["estimate"] == pytest.approx(summary["Equalized Odds Difference"])
    assert dp["estimate"] == pytest.approx(
        demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    )
    assert eo["estimate"] == pytest.approx(
        equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)
    )


def test_bootstrap_does_not_accept_or_require_a_model_object():
    params = inspect.signature(bootstrap_fairness_metric).parameters
    assert "model" not in params
    assert "estimator" not in params
    assert "y_pred" in params
    assert "positive_label" not in params
    assert (
        "positive_label"
        not in inspect.signature(bootstrap_output_fairness).parameters
    )
    assert (
        "positive_label" not in inspect.signature(compute_output_fairness).parameters
    )


def test_bootstrap_requires_zero_one_labels():
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array(["yes", "no", "yes", "no"])
    y_pred = np.array(["yes", "yes", "no", "no"])
    with pytest.raises(ValueError, match="binary 0/1 labels"):
        bootstrap_output_fairness(y_true, y_pred, sensitive, n_bootstrap=50)


def test_bootstrap_wording_is_not_a_fairness_verdict():
    from bias_audit_tool.modeling.fairness import BOOTSTRAP_CI_CAVEAT

    text = BOOTSTRAP_CI_CAVEAT.lower()
    assert "do not establish that a model is fair" in text
    assert "compliant" in text
    assert "95% confidence that the model is fair" not in text
    assert "trustworthy" not in text
