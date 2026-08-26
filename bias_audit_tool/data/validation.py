"""Pure input guardrails for uploaded CSVs and binary modeling support.

These checks detect problems early and explain them. They do not rename
columns, convert infinities to NaN, drop rows, resample classes, or
otherwise repair the user's data.

Thresholds are feasibility heuristics for *this app's* held-out workflow
(``test_size=0.2``, stratified binary split). They are not a claim that
data above the threshold is statistically valid.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import BinaryIO
from typing import Iterable
from typing import Optional
from typing import TextIO
from typing import Union

import numpy as np
import pandas as pd


SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"

CODE_DUPLICATE_HEADERS = "duplicate_headers"
CODE_BLANK_HEADER = "blank_header"
CODE_NON_FINITE_VALUES = "non_finite_values"
CODE_DATASET_TOO_SMALL = "dataset_too_small"
CODE_SMALL_DATASET_WARNING = "small_dataset_warning"
CODE_INSUFFICIENT_CLASS_SUPPORT = "insufficient_class_support"
CODE_EXTREME_CLASS_IMBALANCE = "extreme_class_imbalance"

# Hard floor: with test_size=0.2, n=19 yields a held-out set of ~3 rows,
# which cannot support useful stratified binary evaluation. n=20 is the
# smallest size this app will attempt. It is not a validity certificate.
MIN_MODELING_ROWS = 20

# Heuristic stability warning only. Below 100 rows the ~20-row test split
# is still small enough that metrics move with the particular split.
SMALL_DATASET_WARNING_ROWS = 100

# Stratified 80/20 split needs each class in both partitions. sklearn will
# already refuse a class of size 1; size 2–4 still leaves a 1-row (or empty)
# test cell for that class. Five observations is the smallest count that can
# place at least one example in test (~20%) and several in train.
MIN_CLASS_COUNT_FOR_MODELING = 5

# Heuristic warning, not an invalidity rule: minority share at or below 10%
# (majority:minority >= 9:1). Imbalance is surfaced, never auto-corrected.
EXTREME_IMBALANCE_MINORITY_FRACTION = 0.10

_FINGERPRINT_CHUNK_SIZE = 65_536


@dataclass
class DataValidationIssue:
    code: str
    severity: str
    message: str
    details: dict = field(default_factory=dict)


class DataValidationError(ValueError):
    """One or more blocking input-guardrail issues."""

    def __init__(self, issues: list[DataValidationIssue]):
        if not issues:
            raise ValueError("DataValidationError requires at least one issue.")
        self.issues = list(issues)
        blocking = blocking_issues(self.issues)
        if not blocking:
            blocking = self.issues
        super().__init__("\n".join(issue.message for issue in blocking))


FileLike = Union[BinaryIO, TextIO, io.BytesIO, io.StringIO, Any]


def blocking_issues(issues: Iterable[DataValidationIssue]) -> list[DataValidationIssue]:
    return [issue for issue in issues if issue.severity == SEVERITY_ERROR]


def warning_issues(issues: Iterable[DataValidationIssue]) -> list[DataValidationIssue]:
    return [issue for issue in issues if issue.severity == SEVERITY_WARNING]


def _tell(file_obj: FileLike) -> Optional[int]:
    tell = getattr(file_obj, "tell", None)
    if tell is None:
        return None
    try:
        return tell()
    except (OSError, io.UnsupportedOperation):
        return None


def _seek(file_obj: FileLike, position: int) -> None:
    seek = getattr(file_obj, "seek", None)
    if seek is None:
        raise TypeError("Uploaded file object does not support seek().")
    seek(position)


def _restore_position(file_obj: FileLike, position: Optional[int]) -> None:
    if position is None:
        _seek(file_obj, 0)
    else:
        _seek(file_obj, position)


def fingerprint_upload(file_obj: FileLike) -> str:
    """Return a SHA-256 hex digest of the uploaded bytes.

    The file cursor is restored so a subsequent CSV parse still sees the
    complete stream. The digest is for in-session identity only; it must
    not be logged, transmitted, or persisted outside Streamlit session
    state.
    """
    position = _tell(file_obj)
    digest = hashlib.sha256()
    try:
        _seek(file_obj, 0)
        while True:
            chunk = file_obj.read(_FINGERPRINT_CHUNK_SIZE)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        _restore_position(file_obj, position)


def inspect_csv_header(file_obj: FileLike) -> list[str]:
    """Parse the first CSV record with the stdlib csv reader.

    Reads only as far as the first record (not the whole file). Handles
    quoted commas, escaped quotes, empty fields, and a UTF-8 BOM.
    Restores the file cursor before returning.
    """
    position = _tell(file_obj)
    text_wrapper = None
    try:
        _seek(file_obj, 0)
        probe = file_obj.read(4)
        _seek(file_obj, 0)
        if isinstance(probe, bytes):
            text_wrapper = io.TextIOWrapper(
                file_obj,
                encoding="utf-8-sig",
                newline="",
                errors="replace",
            )
            reader_source = text_wrapper
        else:
            first = file_obj.read(1)
            if first != "\ufeff":
                _seek(file_obj, 0)
            reader_source = file_obj
        try:
            header = next(csv.reader(reader_source), [])
        except csv.Error as exc:
            raise DataValidationError(
                [
                    DataValidationIssue(
                        code=CODE_BLANK_HEADER,
                        severity=SEVERITY_ERROR,
                        message=(
                            "The CSV header could not be parsed. Check the "
                            "file encoding and quoting, then upload again."
                        ),
                        details={"error": str(exc)},
                    )
                ]
            ) from exc
        if header and header[0].startswith("\ufeff"):
            header = [header[0].lstrip("\ufeff"), *header[1:]]
        return header
    finally:
        if text_wrapper is not None:
            try:
                text_wrapper.detach()
            except Exception:
                pass
        _restore_position(file_obj, position)


def assess_csv_headers(header: list[str]) -> list[DataValidationIssue]:
    """Return structural header issues. Does not rename columns."""
    issues: list[DataValidationIssue] = []
    if not header:
        issues.append(
            DataValidationIssue(
                code=CODE_BLANK_HEADER,
                severity=SEVERITY_ERROR,
                message=(
                    "The CSV has no column header. Add a header row with "
                    "unique column names and upload the file again."
                ),
                details={"header": []},
            )
        )
        return issues

    blank_indexes = [i for i, name in enumerate(header) if str(name).strip() == ""]
    if blank_indexes:
        issues.append(
            DataValidationIssue(
                code=CODE_BLANK_HEADER,
                severity=SEVERITY_ERROR,
                message=(
                    "Blank CSV column names were found at position(s) "
                    f"{', '.join(str(i + 1) for i in blank_indexes)}. "
                    "Name every column in the source CSV before continuing."
                ),
                details={
                    "blank_positions": [i + 1 for i in blank_indexes],
                    "n_blank": len(blank_indexes),
                },
            )
        )

    named = [name for name in header if str(name).strip() != ""]
    duplicates = sorted(name for name, count in Counter(named).items() if count > 1)
    if duplicates:
        quoted = ", ".join(f"`{name}`" for name in duplicates)
        issues.append(
            DataValidationIssue(
                code=CODE_DUPLICATE_HEADERS,
                severity=SEVERITY_ERROR,
                message=(
                    "Duplicate column names were found in the original CSV "
                    f"header: {quoted}. Rename them in the source file "
                    "before continuing."
                ),
                details={"duplicate_names": duplicates},
            )
        )
    return issues


def validate_finite_values(df: pd.DataFrame) -> Optional[DataValidationIssue]:
    """Flag +inf/-inf in numeric columns. NaN is not treated as infinity.

    Object/string columns are ignored unless pandas already parsed them as
    a numeric dtype. Textual ``\"inf\"`` in an object column is not flagged.
    """
    per_column: dict[str, int] = {}
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        if pd.api.types.is_bool_dtype(series):
            continue
        values = pd.to_numeric(series, errors="coerce")
        n_inf = int(np.isinf(values.to_numpy(dtype=float, copy=False)).sum())
        if n_inf:
            per_column[str(col)] = n_inf

    if not per_column:
        return None

    parts = [f"`{name}` ({count})" for name, count in per_column.items()]
    joined = " and ".join(parts) if len(parts) <= 2 else ", ".join(parts)
    return DataValidationIssue(
        code=CODE_NON_FINITE_VALUES,
        severity=SEVERITY_ERROR,
        message=(
            f"Numeric infinity values were found in {joined}. Replace "
            "`+inf` / `-inf` with finite values or missing values before "
            "preprocessing or modeling."
        ),
        details={"columns": per_column, "n_total": int(sum(per_column.values()))},
    )


def assess_dataset_size(n_effective_rows: int) -> list[DataValidationIssue]:
    """Assess effective (target-non-null) row count for held-out modeling."""
    issues: list[DataValidationIssue] = []
    if n_effective_rows < MIN_MODELING_ROWS:
        issues.append(
            DataValidationIssue(
                code=CODE_DATASET_TOO_SMALL,
                severity=SEVERITY_ERROR,
                message=(
                    "This dataset is too small for the app's held-out "
                    f"modeling workflow ({n_effective_rows} usable rows; "
                    f"minimum {MIN_MODELING_ROWS}). Upload a larger sample "
                    "or evaluate these rows without the modeling path."
                ),
                details={
                    "n_effective_rows": int(n_effective_rows),
                    "min_modeling_rows": MIN_MODELING_ROWS,
                },
            )
        )
        return issues

    if n_effective_rows < SMALL_DATASET_WARNING_ROWS:
        issues.append(
            DataValidationIssue(
                code=CODE_SMALL_DATASET_WARNING,
                severity=SEVERITY_WARNING,
                message=(
                    "This is a small evaluation dataset "
                    f"({n_effective_rows} usable rows). Held-out performance "
                    "and fairness estimates may be unstable and sensitive to "
                    "the particular train/test split. This is a heuristic "
                    "warning, not a formal sample-size calculation."
                ),
                details={
                    "n_effective_rows": int(n_effective_rows),
                    "small_dataset_warning_rows": SMALL_DATASET_WARNING_ROWS,
                },
            )
        )
    return issues


def describe_class_distribution(y: pd.Series) -> dict:
    """Counts and percentages on effective (already non-null) target rows.

    Labels are the original values; they are not renamed to 'positive'.
    """
    counts = pd.Series(y).value_counts(dropna=True)
    total = int(counts.sum())
    classes = []
    for label, count in counts.items():
        classes.append(
            {
                "label": label,
                "count": int(count),
                "percentage": (float(count) / total) if total else 0.0,
            }
        )
    return {
        "n_effective_rows": total,
        "n_classes": int(len(classes)),
        "classes": classes,
    }


def assess_binary_class_support(
    y: pd.Series, target_name: str = "target"
) -> list[DataValidationIssue]:
    """Hard class-support and heuristic imbalance checks on effective rows.

    Does not replace ``validate_classification_target``. Callers should run
    that binary-shape check first (or accept that a non-binary series may
    still produce class-count issues here).
    """
    y_clean = pd.Series(y).dropna()
    distribution = describe_class_distribution(y_clean)
    issues: list[DataValidationIssue] = []
    original_counts = {
        item["label"]: item["count"] for item in distribution["classes"]
    }

    too_small = {
        label: count
        for label, count in original_counts.items()
        if count < MIN_CLASS_COUNT_FOR_MODELING
    }
    if too_small:
        count_text = "; ".join(
            f"class `{label}`: n={count}" for label, count in original_counts.items()
        )
        smallest = min(original_counts.items(), key=lambda item: item[1])
        issues.append(
            DataValidationIssue(
                code=CODE_INSUFFICIENT_CLASS_SUPPORT,
                severity=SEVERITY_ERROR,
                message=(
                    f"Class `{smallest[0]}` has only {smallest[1]} usable "
                    "rows. The held-out binary-classification workflow "
                    f"needs at least {MIN_CLASS_COUNT_FOR_MODELING} "
                    "observations in each class. "
                    f"{count_text}."
                ),
                details={
                    "class_counts": {
                        str(label): int(count)
                        for label, count in original_counts.items()
                    },
                    "min_class_count": MIN_CLASS_COUNT_FOR_MODELING,
                    "n_effective_rows": distribution["n_effective_rows"],
                    "target_name": target_name,
                },
            )
        )

    if original_counts:
        minority_count = min(original_counts.values())
        n = distribution["n_effective_rows"]
        minority_fraction = (minority_count / n) if n else 0.0
        if minority_fraction <= EXTREME_IMBALANCE_MINORITY_FRACTION:
            issues.append(
                DataValidationIssue(
                    code=CODE_EXTREME_CLASS_IMBALANCE,
                    severity=SEVERITY_WARNING,
                    message=(
                        "The selected target is highly imbalanced "
                        f"(minority share {minority_fraction:.1%} of "
                        f"{n} usable rows). Accuracy and fairness metrics "
                        "may be dominated by the majority class; review "
                        "precision, recall, confusion matrix, and group "
                        "support carefully. This threshold "
                        f"({EXTREME_IMBALANCE_MINORITY_FRACTION:.0%} minority "
                        "or a 9:1 ratio) is a heuristic warning, not a "
                        "finding that the data are invalid."
                    ),
                    details={
                        "class_counts": {
                            str(label): int(count)
                            for label, count in original_counts.items()
                        },
                        "minority_fraction": minority_fraction,
                        "threshold": EXTREME_IMBALANCE_MINORITY_FRACTION,
                        "n_effective_rows": n,
                        "target_name": target_name,
                    },
                )
            )
    return issues


def collect_modeling_guardrails(
    y: pd.Series, target_name: str = "target"
) -> list[DataValidationIssue]:
    """Size, class-support, and imbalance issues on effective target rows."""
    y_clean = pd.Series(y).dropna()
    return assess_dataset_size(len(y_clean)) + assess_binary_class_support(
        y_clean, target_name=target_name
    )
