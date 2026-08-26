import numpy as np

from bias_audit_tool.modeling.fairness import assess_group_support
from bias_audit_tool.modeling.fairness import compute_group_support
from bias_audit_tool.modeling.fairness import compute_output_fairness
from bias_audit_tool.modeling.fairness import DEFAULT_POSITIVE_LABEL
from bias_audit_tool.modeling.fairness import GROUP_COL
from bias_audit_tool.modeling.fairness import MIN_CLASS_SUPPORT_WARNING
from bias_audit_tool.modeling.fairness import MIN_GROUP_N_WARNING
from bias_audit_tool.modeling.fairness import SUPPORT_COL_N
from bias_audit_tool.modeling.fairness import SUPPORT_COL_NEGATIVE_LABELS
from bias_audit_tool.modeling.fairness import SUPPORT_COL_POSITIVE_LABELS
from bias_audit_tool.modeling.fairness import SUPPORT_COL_PREDICTED_NEGATIVES
from bias_audit_tool.modeling.fairness import SUPPORT_COL_PREDICTED_POSITIVES
from bias_audit_tool.modeling.fairness import SUPPORT_COL_SELECTION_DENOMINATOR
from bias_audit_tool.modeling.fairness import SUPPORT_TABLE_COLUMNS
from bias_audit_tool.modeling.fairness import WARNING_FEW_NEGATIVE_LABELS
from bias_audit_tool.modeling.fairness import WARNING_FEW_POSITIVE_LABELS
from bias_audit_tool.modeling.fairness import WARNING_MISSING_SENSITIVE_EXCLUDED
from bias_audit_tool.modeling.fairness import WARNING_SMALL_GROUP_N
from bias_audit_tool.modeling.fairness import WARNING_ZERO_NEGATIVE_LABELS
from bias_audit_tool.modeling.fairness import WARNING_ZERO_POSITIVE_LABELS


def _by_group(support_df):
    return support_df.set_index(GROUP_COL)


def _codes(warnings, group=None):
    return [
        item["code"] for item in warnings if group is None or item["group"] == group
    ]


def test_two_group_support_counts_are_exact():
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])

    support = compute_group_support(y_true, y_pred, sensitive)
    by_group = _by_group(support)

    assert list(support.columns) == SUPPORT_TABLE_COLUMNS
    assert support.attrs["positive_label"] == DEFAULT_POSITIVE_LABEL
    for group in ("A", "B"):
        row = by_group.loc[group]
        assert row[SUPPORT_COL_N] == 2
        assert row[SUPPORT_COL_POSITIVE_LABELS] == 1
        assert row[SUPPORT_COL_NEGATIVE_LABELS] == 1
        assert row[SUPPORT_COL_PREDICTED_POSITIVES] == 1
        assert row[SUPPORT_COL_PREDICTED_NEGATIVES] == 1
        assert row[SUPPORT_COL_SELECTION_DENOMINATOR] == 2
        assert row[SUPPORT_COL_SELECTION_DENOMINATOR] == row[SUPPORT_COL_N]


def test_three_or_more_groups_all_appear_with_correct_counts():
    sensitive = np.array(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0])

    support = compute_group_support(y_true, y_pred, sensitive)
    by_group = _by_group(support)

    assert set(support[GROUP_COL]) == {"A", "B", "C"}
    assert by_group.loc["A", SUPPORT_COL_N] == 4
    assert by_group.loc["A", SUPPORT_COL_POSITIVE_LABELS] == 2
    assert by_group.loc["A", SUPPORT_COL_NEGATIVE_LABELS] == 2
    assert by_group.loc["A", SUPPORT_COL_PREDICTED_POSITIVES] == 4
    assert by_group.loc["A", SUPPORT_COL_PREDICTED_NEGATIVES] == 0
    assert by_group.loc["B", SUPPORT_COL_PREDICTED_POSITIVES] == 0
    assert by_group.loc["C", SUPPORT_COL_PREDICTED_POSITIVES] == 1
    assert by_group.loc["C", SUPPORT_COL_POSITIVE_LABELS] == 0
    assert (
        support[SUPPORT_COL_SELECTION_DENOMINATOR] == support[SUPPORT_COL_N]
    ).all()


def test_string_group_labels_are_preserved():
    sensitive = np.array(["Group A", "Group A", "Group B", "Group B"])
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1])

    support = compute_group_support(y_true, y_pred, sensitive)
    by_group = _by_group(support)

    assert list(support[GROUP_COL]) == ["Group A", "Group B"]
    assert by_group.loc["Group A", SUPPORT_COL_PREDICTED_POSITIVES] == 0
    assert by_group.loc["Group B", SUPPORT_COL_PREDICTED_POSITIVES] == 2


def test_integer_and_boolean_group_labels_are_not_stringified():
    int_support = compute_group_support(
        np.array([1, 0, 1, 0]),
        np.array([1, 0, 0, 1]),
        np.array([0, 0, 1, 1]),
    )
    assert set(int_support[GROUP_COL]) == {0, 1}
    assert all(isinstance(v, (int, np.integer)) for v in int_support[GROUP_COL])

    bool_support = compute_group_support(
        np.array([1, 0, 1, 0]),
        np.array([1, 1, 0, 0]),
        np.array([True, True, False, False]),
    )
    assert set(bool_support[GROUP_COL]) == {True, False}
    assert all(isinstance(v, (bool, np.bool_)) for v in bool_support[GROUP_COL])


def test_sparse_group_emits_small_group_warning_code():
    sensitive = np.array(["small"] * 2 + ["large"] * 40)
    y_true = np.array([1, 0] + [1, 0] * 20)
    y_pred = np.array([1, 0] + [1, 0] * 20)

    support = compute_group_support(y_true, y_pred, sensitive)
    warning_items = assess_group_support(support)

    assert MIN_GROUP_N_WARNING == 30
    assert WARNING_SMALL_GROUP_N in _codes(warning_items, "small")
    assert WARNING_SMALL_GROUP_N not in _codes(warning_items, "large")
    small = next(
        item
        for item in warning_items
        if item["group"] == "small" and item["code"] == WARNING_SMALL_GROUP_N
    )
    assert small["n"] == 2
    assert small["threshold"] == MIN_GROUP_N_WARNING
    assert "heuristic" in small["message"].lower()
    assert (
        "valid" not in small["message"].lower()
        or "not a validity" in small["message"].lower()
    )


def test_zero_positive_labels_emits_explicit_eo_support_warning():
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array([0, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    support = compute_group_support(y_true, y_pred, sensitive)
    warning_items = assess_group_support(support)

    assert _by_group(support).loc["A", SUPPORT_COL_POSITIVE_LABELS] == 0
    assert WARNING_ZERO_POSITIVE_LABELS in _codes(warning_items, "A")
    assert WARNING_FEW_POSITIVE_LABELS not in _codes(warning_items, "A")
    assert WARNING_ZERO_POSITIVE_LABELS not in _codes(warning_items, "B")


def test_zero_negative_labels_emits_explicit_eo_support_warning():
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0])

    support = compute_group_support(y_true, y_pred, sensitive)
    warning_items = assess_group_support(support)

    assert _by_group(support).loc["A", SUPPORT_COL_NEGATIVE_LABELS] == 0
    assert WARNING_ZERO_NEGATIVE_LABELS in _codes(warning_items, "A")
    assert WARNING_FEW_NEGATIVE_LABELS not in _codes(warning_items, "A")
    assert WARNING_ZERO_NEGATIVE_LABELS not in _codes(warning_items, "B")


def test_predictions_all_zero_have_zero_predicted_positives():
    sensitive = np.array(["A", "A", "B", "B"])
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])

    support = compute_group_support(y_true, y_pred, sensitive)
    assert (support[SUPPORT_COL_PREDICTED_POSITIVES] == 0).all()
    assert (support[SUPPORT_COL_PREDICTED_NEGATIVES] == support[SUPPORT_COL_N]).all()


def test_predictions_all_one_have_predicted_positives_equal_group_n():
    sensitive = np.array(["A", "A", "B", "B", "C", "C"])
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1])

    support = compute_group_support(y_true, y_pred, sensitive)
    assert (support[SUPPORT_COL_PREDICTED_POSITIVES] == support[SUPPORT_COL_N]).all()
    assert (support[SUPPORT_COL_PREDICTED_NEGATIVES] == 0).all()


def test_few_class_support_uses_heuristic_threshold_not_zero_code():
    n_pos = MIN_CLASS_SUPPORT_WARNING - 1
    n_neg = MIN_CLASS_SUPPORT_WARNING + 5
    n = n_pos + n_neg
    sensitive = np.array(["A"] * n + ["B"] * n)
    y_true = np.array([1] * n_pos + [0] * n_neg + [1] * n_neg + [0] * n_pos)
    y_pred = np.zeros(len(sensitive), dtype=int)

    warning_items = assess_group_support(
        compute_group_support(y_true, y_pred, sensitive)
    )
    assert WARNING_FEW_POSITIVE_LABELS in _codes(warning_items, "A")
    assert WARNING_ZERO_POSITIVE_LABELS not in _codes(warning_items, "A")
    assert WARNING_FEW_NEGATIVE_LABELS in _codes(warning_items, "B")
    assert WARNING_ZERO_NEGATIVE_LABELS not in _codes(warning_items, "B")


def test_missing_sensitive_values_are_excluded_consistently_with_metricframe():
    sensitive = np.array(["A", "A", "B", "B", None, None], dtype=object)
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 1])

    support = compute_group_support(y_true, y_pred, sensitive)
    metric_frame, _ = compute_output_fairness(y_true, y_pred, sensitive)
    warning_items = assess_group_support(support)

    assert support.attrs["n_missing_sensitive"] == 2
    assert set(support[GROUP_COL]) == {"A", "B"}
    assert set(metric_frame.by_group.index) == {"A", "B"}
    assert support[SUPPORT_COL_N].sum() == 4
    assert WARNING_MISSING_SENSITIVE_EXCLUDED in _codes(warning_items)
    missing = next(
        item
        for item in warning_items
        if item["code"] == WARNING_MISSING_SENSITIVE_EXCLUDED
    )
    assert missing["n_missing"] == 2
    assert missing["group"] is None


def test_support_uses_held_out_rows_not_a_full_dataset_length():
    # Mimic a 100-row upload with a 6-row held-out evaluation set.
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0])
    sensitive = np.array(["A", "A", "A", "B", "B", "B"])

    support = compute_group_support(y_true, y_pred, sensitive)
    assert support[SUPPORT_COL_N].sum() == 6
    assert support[SUPPORT_COL_N].sum() != 100


def test_positive_class_is_one_for_binary_zero_one_labels():
    support = compute_group_support(
        np.array([1, 1, 0, 0]),
        np.array([1, 0, 1, 0]),
        np.array(["A", "A", "B", "B"]),
        positive_label=DEFAULT_POSITIVE_LABEL,
    )
    by_group = _by_group(support)
    assert DEFAULT_POSITIVE_LABEL == 1
    assert by_group.loc["A", SUPPORT_COL_POSITIVE_LABELS] == 2
    assert by_group.loc["B", SUPPORT_COL_POSITIVE_LABELS] == 0
    assert by_group.loc["A", SUPPORT_COL_PREDICTED_POSITIVES] == 1
    assert by_group.loc["B", SUPPORT_COL_PREDICTED_POSITIVES] == 1


def test_explicit_positive_label_counts_string_targets_consistently():
    support = compute_group_support(
        np.array(["yes", "no", "yes", "no"]),
        np.array(["yes", "yes", "no", "no"]),
        np.array(["A", "A", "B", "B"]),
        positive_label="yes",
    )
    by_group = _by_group(support)
    assert by_group.loc["A", SUPPORT_COL_POSITIVE_LABELS] == 1
    assert by_group.loc["A", SUPPORT_COL_PREDICTED_POSITIVES] == 2
    assert by_group.loc["B", SUPPORT_COL_POSITIVE_LABELS] == 1
    assert by_group.loc["B", SUPPORT_COL_PREDICTED_POSITIVES] == 0


def test_support_warnings_are_not_fairness_verdicts():
    sensitive = np.array(["A"] * 2 + ["B"] * 40)
    y_true = np.array([0, 0] + [1, 0] * 20)
    y_pred = np.array([1, 0] + [1, 0] * 20)
    warning_items = assess_group_support(
        compute_group_support(y_true, y_pred, sensitive)
    )
    joined = " ".join(item["message"] for item in warning_items).lower()
    for banned in (
        "fair",
        "not fair",
        "unbiased",
        "compliant",
        "passes fairness",
        "statistically fair",
        "statistically valid",
    ):
        assert banned not in joined
