# Target-shape validation for the classification-only modeling path.
#
# This tool only supports classification-style fairness auditing today (the
# model selection, ROC/AUC display, and the Fairlearn-based output-fairness
# metrics are all built around a single binary positive class). Silently
# treating an unsuitable target as "just another classification problem" —
# e.g. a continuous lab value, or a near-unique patient ID — produces a
# model and fairness numbers that look valid but mean nothing.
#
# Heuristic used here (kept intentionally simple):
#   - Fewer than 2 observed classes            -> reject, nothing to classify.
#   - Exactly 2 observed classes                -> binary classification,
#     accepted regardless of dtype (int, float, bool, or string labels).
#   - More than 2 classes and the column is numeric with non-integer values
#     and "many" unique values                  -> reject as a continuous /
#     regression-shaped target.
#   - More than 2 classes and most rows have a distinct value (high
#     unique-value ratio) with "many" unique values overall -> reject as a
#     near-unique / identifier-like column, regardless of dtype (also catches
#     sequential integer IDs, which are integer-valued but not a label set).
#   - Any other >2-class case (a plausible small int-coded or string
#     multiclass label set) -> still rejected today, because the rest of the
#     pipeline (ROC/AUC assuming one positive-class probability column,
#     sklearn's binary-averaged precision/recall/F1, and Fairlearn's
#     demographic-parity/equalized-odds differences) is only correct for
#     binary targets. Rejecting explicitly here is preferable to training
#     successfully and failing later inside sklearn/Fairlearn.
from dataclasses import dataclass
from typing import Iterable
from typing import Optional
from typing import Sequence

import pandas as pd


class UnsupportedTargetError(ValueError):
    """Raised when the selected target column cannot be used for the
    binary-classification fairness audit this tool supports."""

    def __init__(self, message: str, reason: str, **details):
        super().__init__(message)
        self.reason = reason
        self.details = details


@dataclass
class TargetValidationResult:
    kind: str  # always "binary" when validation succeeds
    n_classes: int
    n_samples: int


def validate_classification_target(
    y: pd.Series,
    target_name: str = "target",
    max_reasonable_classes: int = 20,
    near_unique_ratio: float = 0.5,
) -> TargetValidationResult:
    """
    Validate that `y` is a supported binary classification target.

    Returns a TargetValidationResult on success. Raises
    UnsupportedTargetError with an actionable message otherwise.
    """
    y_clean = pd.Series(y).dropna()
    n = len(y_clean)

    if n == 0:
        raise UnsupportedTargetError(
            f"Target '{target_name}' has no non-null values.",
            reason="empty_target",
        )

    nunique = y_clean.nunique()

    if nunique < 2:
        raise UnsupportedTargetError(
            f"Target '{target_name}' has only {nunique} distinct value(s) "
            "after dropping missing values; at least 2 classes are "
            "required to train a classifier.",
            reason="too_few_classes",
            n_classes=nunique,
        )

    if nunique == 2:
        return TargetValidationResult(kind="binary", n_classes=2, n_samples=n)

    unique_ratio = nunique / n
    is_numeric = pd.api.types.is_numeric_dtype(
        y_clean
    ) and not pd.api.types.is_bool_dtype(y_clean)
    is_integer_valued = is_numeric and (y_clean % 1 == 0).all()

    if is_numeric and not is_integer_valued and nunique > max_reasonable_classes:
        raise UnsupportedTargetError(
            f"Target '{target_name}' looks like a continuous numeric "
            f"(regression-shaped) column with {nunique} unique values. "
            "This tool only supports classification targets — choose a "
            "categorical or binary outcome column, or bin this column "
            "into discrete classes before running the audit.",
            reason="continuous_target",
            n_classes=nunique,
        )

    if unique_ratio > near_unique_ratio and nunique > max_reasonable_classes:
        raise UnsupportedTargetError(
            f"Target '{target_name}' has {nunique} unique values across "
            f"{n} rows ({unique_ratio:.0%} unique) — this looks like an "
            "identifier or free-form value rather than a classification "
            "label. Choose a categorical outcome column instead.",
            reason="near_unique_target",
            n_classes=nunique,
            unique_ratio=unique_ratio,
        )

    raise UnsupportedTargetError(
        "Model-based fairness analysis currently supports binary "
        f"classification targets only. Target '{target_name}' has "
        f"{nunique} classes.",
        reason="multiclass_unsupported",
        n_classes=nunique,
    )


def is_supported_binary_target(y, target_name: str = "target") -> bool:
    """True when ``validate_classification_target`` accepts ``y``."""
    try:
        validate_classification_target(y, target_name=target_name)
        return True
    except UnsupportedTargetError:
        return False


def preferred_target_column(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    current_selection: Optional[str] = None,
    deprioritized: Optional[Iterable[str]] = None,
) -> str:
    """
    Default modeling-target column for a selectbox.

    Preference order:
    1. Keep ``current_selection`` when it is still in ``columns``.
    2. The first supported binary-classification column that is not in
       ``deprioritized`` (callers typically pass the selected sensitive
       attribute and its direct encodings so a grouping variable is not
       the default label).
    3. The first supported binary-classification column.
    4. The first column.

    Column names are not special-cased. Any binary-valid column can win.
    """
    cols = list(columns if columns is not None else df.columns)
    if not cols:
        raise ValueError("Cannot choose a target column from an empty column list.")
    if current_selection in cols:
        return current_selection

    skipped = set(deprioritized or [])
    binary_cols = [
        col
        for col in cols
        if col in df.columns and is_supported_binary_target(df[col], target_name=col)
    ]
    for col in binary_cols:
        if col not in skipped:
            return col
    if binary_cols:
        return binary_cols[0]
    return cols[0]
