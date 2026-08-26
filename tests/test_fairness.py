import numpy as np
import pandas as pd
import pytest
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference
from matplotlib import pyplot as plt

from bias_audit_tool.modeling import fairness as fairness_mod
from bias_audit_tool.modeling.fairness import ALL_GROUPS_WITHIN_THRESHOLD_MESSAGE
from bias_audit_tool.modeling.fairness import BOOTSTRAP_CI_CAVEAT
from bias_audit_tool.modeling.fairness import CLAIM_SAFETY_CAVEAT
from bias_audit_tool.modeling.fairness import compute_input_fairness
from bias_audit_tool.modeling.fairness import compute_output_fairness
from bias_audit_tool.modeling.fairness import FAIRNESS_METRIC_CAVEAT
from bias_audit_tool.modeling.fairness import GROUP_COL
from bias_audit_tool.modeling.fairness import GROUP_SUPPORT_CAPTION
from bias_audit_tool.modeling.fairness import interpret_fairness_metrics
from bias_audit_tool.modeling.fairness import NO_BENCHMARK_AVAILABLE
from bias_audit_tool.modeling.fairness import NO_BENCHMARK_SELECTED_MESSAGE
from bias_audit_tool.modeling.fairness import OUTSIDE_THRESHOLD
from bias_audit_tool.modeling.fairness import parse_user_benchmark
from bias_audit_tool.modeling.fairness import plot_input_fairness
from bias_audit_tool.modeling.fairness import THRESHOLD_STATUS_COL
from bias_audit_tool.modeling.fairness import WITHIN_THRESHOLD


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
    recall_ratio = summary["Recall Ratio"]
    assert isinstance(recall_ratio, dict)
    assert recall_ratio["status"] == "undefined"
    assert recall_ratio["min_value"] == 0.0
    assert recall_ratio["max_value"] == 1.0
    assert "Recall Difference" in summary
    assert summary["Recall Difference"] == pytest.approx(1.0)

    # DP/EO themselves must still compute as real numbers.
    assert isinstance(summary["Demographic Parity Difference"], float)
    assert isinstance(summary["Equalized Odds Difference"], float)


def test_output_fairness_zero_denominator_ratio_is_explicit_not_inf():
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1])  # group A: 0 selected, group B: all selected

    _, summary = compute_output_fairness(y_true, y_pred, sensitive)

    selection_ratio = summary["Selection Rate Ratio"]
    assert isinstance(selection_ratio, dict)
    assert selection_ratio["status"] == "undefined"
    assert "zero" in selection_ratio["reason"].lower()
    assert selection_ratio["min_value"] == 0.0
    assert selection_ratio["max_value"] == 1.0
    assert "Selection Rate Difference" in summary
    assert summary["Selection Rate Difference"] == pytest.approx(1.0)

    # No entry anywhere in the summary should be a bare, unexplained inf.
    for value in summary.values():
        if isinstance(value, float):
            assert not np.isinf(value)


def test_output_fairness_difference_and_ratio_keys_do_not_collide():
    sensitive = np.array(["A"] * 4 + ["B"] * 4)
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])

    _, summary = compute_output_fairness(y_true, y_pred, sensitive)

    metric_bases = ["Accuracy", "Precision", "Recall", "F1", "Selection Rate"]
    for base in metric_bases:
        diff_key = f"{base} Difference"
        ratio_key = f"{base} Ratio"
        assert diff_key in summary
        assert ratio_key in summary
        assert diff_key != ratio_key

    assert len(summary) == len(set(summary))
    assert "Demographic Parity Difference" in summary
    assert "Equalized Odds Difference" in summary
    assert "Demographic Parity Difference" not in {
        f"{base} Difference" for base in metric_bases
    }


def test_output_fairness_undefined_ratio_survives_display_normalization():
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    _, summary = compute_output_fairness(y_true, y_pred, sensitive)

    assert "Selection Rate Difference" in summary
    assert "Selection Rate Ratio" in summary
    assert summary["Selection Rate Difference"] == pytest.approx(1.0)
    assert summary["Selection Rate Ratio"]["status"] == "undefined"
    assert not any(isinstance(v, float) and np.isinf(v) for v in summary.values())


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
    assert (result[THRESHOLD_STATUS_COL] == WITHIN_THRESHOLD).all()
    assert (result["Deviation_Type"] == "Within bounds").all()


def test_input_fairness_unbenchmarked_group_is_explicit_not_fabricated():
    df = pd.DataFrame({"Group": ["Black"] * 30 + ["White"] * 30 + ["Other"] * 10})
    benchmark = {"Black": 0.5, "White": 0.5}  # "Other" intentionally absent

    result = compute_input_fairness(df, "Group", benchmark_distribution=benchmark)
    by_group = result.set_index("Group")

    other_row = by_group.loc["Other"]
    assert pd.isna(other_row["Expected_%"])
    assert pd.isna(other_row["Disparity_Ratio"])
    assert other_row[THRESHOLD_STATUS_COL] == NO_BENCHMARK_AVAILABLE
    assert other_row["Deviation_Type"] == NO_BENCHMARK_AVAILABLE

    # Benchmarked groups must still compute normally.
    assert by_group.loc["Black", "Disparity_Ratio"] == pytest.approx((30 / 70) / 0.5)
    assert by_group.loc["Black", THRESHOLD_STATUS_COL] == WITHIN_THRESHOLD
    assert by_group.loc["White", THRESHOLD_STATUS_COL] == WITHIN_THRESHOLD


def test_input_fairness_missing_sensitive_value_is_deterministic():
    df = pd.DataFrame({"Group": ["Black"] * 10 + ["White"] * 10 + [np.nan] * 5})
    benchmark = {"Black": 0.6, "White": 0.4}

    result = compute_input_fairness(df, "Group", benchmark_distribution=benchmark)

    nan_row = result[result["Group"].isna()].iloc[0]
    assert pd.isna(nan_row["Expected_%"])
    assert pd.isna(nan_row["Disparity_Ratio"])
    assert nan_row[THRESHOLD_STATUS_COL] == NO_BENCHMARK_AVAILABLE
    assert nan_row["Deviation_Type"] == NO_BENCHMARK_AVAILABLE

    black_row = result[result["Group"] == "Black"].iloc[0]
    assert black_row["Disparity_Ratio"] == pytest.approx((10 / 25) / 0.6)
    assert black_row[THRESHOLD_STATUS_COL] == OUTSIDE_THRESHOLD
    assert black_row["Deviation_Type"] == "Under-represented"

    white_row = result[result["Group"] == "White"].iloc[0]
    assert white_row[THRESHOLD_STATUS_COL] == WITHIN_THRESHOLD


def test_input_fairness_no_benchmark_does_not_use_tnbc_or_fabricate_ratios():
    # Shares that would look aligned with the former TNBC race table.
    df = pd.DataFrame(
        {
            "Group": (
                ["Black"] * 36
                + ["White"] * 19
                + ["Hispanic"] * 16
                + ["AIAN"] * 16
                + ["Asian"] * 13
            )
        }
    )

    result = compute_input_fairness(df, "Group")

    assert not hasattr(fairness_mod, "TNBC_RACE_BENCHMARK")
    assert result.attrs["benchmark_status"] == "missing"
    assert result.attrs["benchmark_message"] == NO_BENCHMARK_SELECTED_MESSAGE
    assert result["Expected_%"].isna().all()
    assert result["Disparity_Ratio"].isna().all()
    assert result["Absolute_Difference"].isna().all()
    assert (result[THRESHOLD_STATUS_COL] == NO_BENCHMARK_AVAILABLE).all()
    assert "Fair?" not in result.columns
    assert "Fair" not in set(result[THRESHOLD_STATUS_COL])
    assert "Not Fair" not in set(result[THRESHOLD_STATUS_COL])


def test_input_fairness_empty_benchmark_is_treated_as_missing():
    df = pd.DataFrame({"Group": ["A"] * 5 + ["B"] * 5})
    result = compute_input_fairness(df, "Group", benchmark_distribution={})
    assert result.attrs["benchmark_status"] == "missing"
    assert result["Disparity_Ratio"].isna().all()


def test_input_fairness_explicit_benchmark_computes_ratios():
    df = pd.DataFrame({"Group": ["A"] * 80 + ["B"] * 20})
    benchmark = {"A": 0.5, "B": 0.5}

    result = compute_input_fairness(df, "Group", benchmark_distribution=benchmark)
    by_group = result.set_index("Group")

    assert by_group.loc["A", "Disparity_Ratio"] == pytest.approx(0.8 / 0.5)
    assert by_group.loc["B", "Disparity_Ratio"] == pytest.approx(0.2 / 0.5)
    assert by_group.loc["A", THRESHOLD_STATUS_COL] == OUTSIDE_THRESHOLD
    assert by_group.loc["B", THRESHOLD_STATUS_COL] == OUTSIDE_THRESHOLD
    assert "Not Fair" not in set(result[THRESHOLD_STATUS_COL])
    assert "Fair" not in set(result[THRESHOLD_STATUS_COL])


def test_parse_user_benchmark_does_not_invent_a_distribution():
    assert parse_user_benchmark("") == (None, "missing")
    assert parse_user_benchmark("   ") == (None, "missing")
    assert parse_user_benchmark("{") == (None, "invalid")
    assert parse_user_benchmark("{}") == (None, "missing")
    parsed, status = parse_user_benchmark('{"A": 0.5, "B": 0.5}')
    assert status == "ok"
    assert parsed == {"A": 0.5, "B": 0.5}


def test_input_fairness_uses_stable_group_column_for_race():
    df = pd.DataFrame({"race": ["Black"] * 30 + ["White"] * 70})
    benchmark = {"Black": 0.3, "White": 0.7}

    result = compute_input_fairness(
        df, demographic_col="race", benchmark_distribution=benchmark
    )

    assert GROUP_COL in result.columns
    assert list(result.columns[:1]) == [GROUP_COL]
    assert set(result[GROUP_COL]) == {"Black", "White"}
    by_group = result.set_index(GROUP_COL)
    assert by_group.loc["Black", "Disparity_Ratio"] == pytest.approx(1.0)


def test_plot_input_fairness_consumes_race_named_demographic_column():
    df = pd.DataFrame({"race": ["Black"] * 30 + ["White"] * 70})
    benchmark = {"Black": 0.3, "White": 0.7}
    result = compute_input_fairness(
        df, demographic_col="race", benchmark_distribution=benchmark
    )

    fig = plot_input_fairness(result)
    assert fig is not None
    ax = fig.axes[0]
    ytick_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert set(ytick_labels) == {"Black", "White"}
    plt.close(fig)


def test_interpret_fairness_metrics_accepts_na_when_no_group_matched_benchmark():
    labels = interpret_fairness_metrics("N/A", "N/A", "N/A")
    assert labels["KL Divergence"] == "N/A"
    assert labels["Wasserstein Distance"] == "N/A"
    assert labels["Total Variation"] == "N/A"

    df = pd.DataFrame({"race": ["Black"] * 30 + ["White"] * 70})
    result = compute_input_fairness(
        df,
        demographic_col="race",
        benchmark_distribution={"True": 0.4, "False": 0.6},
    )
    labels = interpret_fairness_metrics(
        result.attrs["KL_Divergence"],
        result.attrs["Wasserstein_Distance"],
        result.attrs["Total_Variation"],
    )
    assert labels["KL Divergence"] == "N/A"


def test_public_fairness_wording_is_criterion_based_not_a_verdict():
    assert THRESHOLD_STATUS_COL == "Within Threshold?"
    assert WITHIN_THRESHOLD == "Yes"
    assert OUTSIDE_THRESHOLD == "No"
    assert ALL_GROUPS_WITHIN_THRESHOLD_MESSAGE == (
        "All benchmarked groups fall within the selected disparity-ratio "
        "threshold."
    )
    assert "fair" not in ALL_GROUPS_WITHIN_THRESHOLD_MESSAGE.lower()
    assert "Not Fair" not in ALL_GROUPS_WITHIN_THRESHOLD_MESSAGE
    assert FAIRNESS_METRIC_CAVEAT == (
        "Different fairness metrics can disagree, especially when outcome "
        "base rates differ by group. No single metric proves a model is fair."
    )
    assert CLAIM_SAFETY_CAVEAT == (
        "This is an exploratory diagnostic based on the selected metric and "
        "benchmark. It does not establish that a model or dataset is fair, "
        "unbiased, non-discriminatory, or legally compliant."
    )
    assert NO_BENCHMARK_SELECTED_MESSAGE.startswith("No benchmark selected.")
    assert "population" not in GROUP_SUPPORT_CAPTION.lower() or (
        "not population" in GROUP_SUPPORT_CAPTION.lower()
    )
    assert "held-out" in GROUP_SUPPORT_CAPTION.lower()
    assert "do not establish that a model is fair" in BOOTSTRAP_CI_CAVEAT.lower()
    assert "Not Fair" not in BOOTSTRAP_CI_CAVEAT
    assert "statistically fair" not in BOOTSTRAP_CI_CAVEAT.lower()
