import traceback

import pandas as pd
import streamlit as st

from bias_audit_tool.data.validation import assess_csv_headers
from bias_audit_tool.data.validation import blocking_issues
from bias_audit_tool.data.validation import inspect_csv_header
from bias_audit_tool.data.validation import validate_finite_values


def load_uploaded_dataset(uploaded_file):
    """
    Inspect the raw CSV header, then parse with pandas.

    Header inspection runs on the original stream before ``pd.read_csv``
    so duplicate names are not lost to pandas' ``.1`` renaming. The file
    cursor is restored after inspection.

    Returns:
        tuple[pd.DataFrame | None, list]: DataFrame when the header is
        structurally usable; plus any validation issues (header errors,
        non-finite numeric values).
    """
    header = inspect_csv_header(uploaded_file)
    header_issues = assess_csv_headers(header)
    if blocking_issues(header_issues):
        return None, header_issues

    uploaded_file.seek(0)
    df = pd.read_csv(
        uploaded_file, low_memory=False, na_values=["--", "NA", "N/A", "null"]
    )
    issues = list(header_issues)
    inf_issue = validate_finite_values(df)
    if inf_issue is not None:
        issues.append(inf_issue)
    return df, issues


def load_and_preview_data(uploaded_file):
    """
    Load a CSV file for the Streamlit app and handle errors in the UI.

    Args:
        uploaded_file (UploadedFile): The uploaded CSV file via
        Streamlit file uploader.

    Returns:
        tuple[pd.DataFrame | None, list]: Loaded frame and validation issues.
        Header errors yield ``(None, issues)``.
    """
    try:
        return load_uploaded_dataset(uploaded_file)
    except Exception:
        st.error("❌ Error loading or processing the file:")
        with st.expander("🔍 Show full error details"):
            st.text(traceback.format_exc())
        return None, []
