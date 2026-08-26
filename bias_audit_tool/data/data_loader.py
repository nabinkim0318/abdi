import traceback

import pandas as pd
import streamlit as st


def load_and_preview_data(uploaded_file):
    """
    Load a CSV file for the Streamlit app and handle errors in the UI.

    Args:
        uploaded_file (UploadedFile): The uploaded CSV file via
        Streamlit file uploader.

    Returns:
        pd.DataFrame or None: Returns the loaded DataFrame if successful,
        otherwise None.

    Side Effects:
        - Displays an error message and traceback if loading fails.
    """
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file, low_memory=False, na_values=["--", "NA", "N/A", "null"]
        )
        return df

    except Exception:
        st.error("❌ Error loading or processing the file:")
        with st.expander("🔍 Show full error details"):
            st.text(traceback.format_exc())
        return None
