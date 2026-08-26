import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference

from bias_audit_tool.modeling.fairness import compute_input_fairness
from bias_audit_tool.modeling.fairness import compute_output_fairness


# ---------------------------------------------------------------------------
# Output / model fairness (compute_output_fairness)
# ---------------------------------------------------------------------------


def test_output_fairness_equal_groups_has_zero_dp_and_eo():
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])

    _, summary = compute_output_fairness(y_true, y_pred, sensitive)

    assert summary["Demographic Parity Difference"] == pytest.approx(0.0)
    assert summary["Equalized Odds Difference"] == pytest.approx(0.0)


def test_output_fairness_dp_disparity_matches_fairlearn():
    sensitive = np.array(["A"] * 4 + ["B"] * 4)
    y_true = np.array([0] * 8)
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])  # A: 3/4 selected, B: 0/4 selected

    expected_dp = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive
    )
    _, summary = compute_output_fairness(y_true, y_pred, sensitive)

    assert expected_dp == pytest.approx(0.75)
    assert summary["Demographic Parity Difference"] == pytest.approx(expected_dp)


def test_output_fairness_eo_disparity_matches_fairlearn():
    sensitive = np.array(["A"] * 4 + ["B"] * 4)
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 1, 1])  # A: perfect, B: always wrong

    expected_eo = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive
    )
    _, summary = compute_output_fairness(y_true, y_pred, sensitive)

    assert expected_eo == pytest.approx(1.0)
    assert summary["Equalized Odds Difference"] == pytest.approx(expected_eo)


def test_output_fairness_supports_three_or_more_groups():
    sensitive = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
    y_true = np.array([0] * 12)
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])

    metric_frame, summary = compute_output_fairness(y_true, y_pred, sensitive)

    assert set(metric_frame.by_group.index) == {"A", "B", "C"}
    assert summary["Demographic Parity Difference"] == pytest.approx(1.0)


def test_output_fairness_zero_predicted_positives_does_not_crash():
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])

    metric_frame, summary = compute_output_fairness(y_true, y_pred, sensitive)

    assert (metric_frame.by_group["Selection Rate"] == 0).all()
    assert summary["Demographic Parity Difference"] == pytest.approx(0.0)
    assert summary["Equalized Odds Difference"] == pytest.approx(0.0)


def test_output_fairness_zero_positive_labels_in_group_does_not_crash():
    # Group "A" has no positive labels at all, so recall is undefined there.
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array([0, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    metric_frame, summary = compute_output_fairness(y_true, y_pred, sensitive)

    # sklearn's recall with zero_division=0 reports 0.0 for the group with no
    # positives.
    assert metric_frame.by_group.loc["A", "Recall"] == 0.0

    # The Recall ratio is undefined (min group value is 0) and must be surfaced
    # explicitly rather than as a bare inf or a crash.
    recall_ratio = summary["Coverage of Actual Positives"]
    assert isinstance(recall_ratio, dict)
    assert recall_ratio["status"] == "undefined"
    assert recall_ratio["min_value"] == 0.0
    assert recall_ratio["max_value"] == 1.0

    # DP/EO themselves must still compute as real numbers.
    assert isinstance(summary["Demographic Parity Difference"], float)
    assert isinstance(summary["Equalized Odds Difference"], float)


def test_output_fairness_zero_denominator_ratio_is_explicit_not_inf():
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1])  # group A: 0 selected, group B: all selected

    _, summary = compute_output_fairness(y_true, y_pred, sensitive)

    selection_ratio = summary["Group Selection Rate"]
    assert isinstance(selection_ratio, dict)
    assert selection_ratio["status"] == "undefined"
    assert "zero" in selection_ratio["reason"].lower()
    assert selection_ratio["min_value"] == 0.0
    assert selection_ratio["max_value"] == 1.0

    # No entry anywhere in the summary should be a bare, unexplained inf.
    for value in summary.values():
        if isinstance(value, float):
            assert not np.isinf(value)


# ---------------------------------------------------------------------------
# Input / representation fairness (compute_input_fairness)
# ---------------------------------------------------------------------------


def test_input_fairness_benchmark_covers_all_observed_groups():
    df = pd.DataFrame({"Group": ["Black"] * 36 + ["White"] * 64})
    benchmark = {"Black": 0.36, "White": 0.64}

    result = compute_input_fairness(df, "Group", benchmark_distribution=benchmark)
    by_group = result.set_index("Group")

    assert by_group.loc["Black", "Disparity_Ratio"] == pytest.approx(1.0)
    assert by_group.loc["White", "Disparity_Ratio"] == pytest.approx(1.0)
    assert (result["Fair?"] == "Fair").all()
    assert (result["Deviation_Type"] == "Within bounds").all()


def test_input_fairness_unbenchmarked_group_is_explicit_not_fabricated():
    df = pd.DataFrame({"Group": ["Black"] * 30 + ["White"] * 30 + ["Other"] * 10})
    benchmark = {"Black": 0.5, "White": 0.5}  # "Other" intentionally absent

    result = compute_input_fairness(df, "Group", benchmark_distribution=benchmark)
    by_group = result.set_index("Group")

    other_row = by_group.loc["Other"]
    assert pd.isna(other_row["Expected_%"])
    assert pd.isna(other_row["Disparity_Ratio"])
    assert other_row["Fair?"] == "No benchmark available"
    assert other_row["Deviation_Type"] == "No benchmark available"

    # Benchmarked groups must still compute normally.
    assert by_group.loc["Black", "Disparity_Ratio"] == pytest.approx((30 / 70) / 0.5)
    assert by_group.loc["Black", "Fair?"] == "Fair"
    assert by_group.loc["White", "Fair?"] == "Fair"


def test_input_fairness_missing_sensitive_value_is_deterministic():
    df = pd.DataFrame({"Group": ["Black"] * 10 + ["White"] * 10 + [np.nan] * 5})
    benchmark = {"Black": 0.6, "White": 0.4}

    result = compute_input_fairness(df, "Group", benchmark_distribution=benchmark)

    nan_row = result[result["Group"].isna()].iloc[0]
    assert pd.isna(nan_row["Expected_%"])
    assert pd.isna(nan_row["Disparity_Ratio"])
    assert nan_row["Fair?"] == "No benchmark available"
    assert nan_row["Deviation_Type"] == "No benchmark available"

    black_row = result[result["Group"] == "Black"].iloc[0]
    assert black_row["Disparity_Ratio"] == pytest.approx((10 / 25) / 0.6)
    assert black_row["Fair?"] == "Not Fair"
    assert black_row["Deviation_Type"] == "Under-represented"

    white_row = result[result["Group"] == "White"].iloc[0]
    assert white_row["Fair?"] == "Fair"
