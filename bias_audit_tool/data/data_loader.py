import traceback

import pandas as pd
import streamlit as st
from preprocessing.summary import basic_df_summary


def load_and_preview_data(uploaded_file):
    try:
        df = pd.read_csv(
            uploaded_file, low_memory=False, na_values=["--", "NA", "N/A", "null"]
        )
        basic_df_summary(df)
        st.success("✅ File loaded successfully")
        st.markdown("#### 📄 Original Data Preview")
        st.dataframe(df.head())
        return df

    except Exception:
        st.error("❌ Error loading or processing the file:")
        with st.expander("🔍 Show full error details"):
            st.text(traceback.format_exc())
        return None
