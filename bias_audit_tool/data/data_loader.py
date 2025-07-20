import traceback

import pandas as pd
import streamlit as st


def load_and_preview_data(uploaded_file):
    """
    Loads a CSV file, displays a basic summary and data preview in Streamlit,
    and handles any errors gracefully.

    Args:
        uploaded_file (UploadedFile): The uploaded CSV file via
        Streamlit file uploader.

    Returns:
        pd.DataFrame or None: Returns the loaded DataFrame if successful,
        otherwise None.

    Side Effects:
        - Displays success message and data preview in the Streamlit app.
        - Displays error message and traceback if an exception occurs.
    """
    try:
        df = pd.read_csv(
            uploaded_file, low_memory=False, na_values=["--", "NA", "N/A", "null"]
        )
        # basic_df_summary(df)
        # st.success("✅ File loaded successfully")
        # st.markdown("#### 📄 Original Data Preview")
        # st.dataframe(df.head())
        return df

    except Exception:
        st.error("❌ Error loading or processing the file:")
        with st.expander("🔍 Show full error details"):
            st.text(traceback.format_exc())
        return None
