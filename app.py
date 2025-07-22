# bias_audit_tool/app.py
import streamlit as st

from bias_audit_tool.data.data_loader import load_and_preview_data
from bias_audit_tool.preprocessing.recommend_columns import (
    recommend_demographic_columns,
)
from bias_audit_tool.utils.ui_helpers import apply_preprocessing_and_display
from bias_audit_tool.utils.ui_helpers import display_preprocessing_recommendations
from bias_audit_tool.utils.ui_helpers import get_user_preprocessing_options
from bias_audit_tool.utils.ui_helpers import run_modeling_and_fairness
from bias_audit_tool.visualization.ui_blocks import audit_and_visualize_fairness
from bias_audit_tool.visualization.ui_blocks import download_processed_csv
from bias_audit_tool.visualization.visualization import auto_group_selector
from bias_audit_tool.visualization.visualization import plot_radar_chart
from bias_audit_tool.visualization.visualization import show_visualizations


st.set_page_config(page_title="Bias Audit Tool", layout="wide")


# ===== Sidebar =====
st.sidebar.title("📊 Bias Audit Assistant")
enable_modeling = st.sidebar.radio("🤖 Run ML Model?", ["No", "Yes"])


# ===== Main Panel =====
st.title("🧪 Bias Audit Dashboard")


def main():
    group_col = None
    uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])

    if "target_col" not in st.session_state:
        st.session_state.target_col = None
    if "preprocessing_applied" not in st.session_state:
        st.session_state.preprocessing_applied = False

    if uploaded_file:
        df = load_and_preview_data(uploaded_file)
        if df is None:
            st.stop()

        # 👉 Step 1: Recommendations
        recommendations = display_preprocessing_recommendations(df)
        st.session_state.recommendations = recommendations
        show_logs = st.checkbox("🪵 Show detailed preprocessing logs", value=False)
        options = get_user_preprocessing_options()

        # 👉 Step 2: Apply Preprocessing
        if button_clicked("preprocessing_button"):
            df_proc = apply_preprocessing_and_display(
                df, recommendations, show_logs, options
            )
            st.session_state.df_proc = df_proc  # persist state
            st.session_state.recommendations = recommendations
            st.session_state.preprocessing_applied = True
            st.success("✅ Preprocessing applied!")

        # If preprocessing is done, allow further analysis
        if st.session_state.preprocessing_applied and "df_proc" in st.session_state:
            df_proc = st.session_state.df_proc
            recommendations = st.session_state.recommendations

            # 🔄 Auto One-Hot Column Restoration (moved earlier)
            # Use a copy to avoid modifying the original
            df_proc_copy = df_proc.copy()
            df_proc_restored, group_col = auto_group_selector(df_proc_copy)

            # Update session state with restored data if restoration was successful
            if group_col:
                st.session_state.group_col = group_col
                df_proc = df_proc_restored
                st.session_state.df_proc = df_proc
                st.success(f"✅ Group column '{group_col}' restored.")

            else:
                group_col = st.session_state.get("group_col", None)

            demo_cols_result = recommend_demographic_columns(df_proc)
            if demo_cols_result:
                # Filter to only include columns that exist in the DataFrame
                demo_cols = [
                    col for col in demo_cols_result if col in df_proc.columns
                ]
                if demo_cols:
                    group_col = st.selectbox(
                        "Select demographic column", demo_cols, index=0
                    )
                    show_visualizations(df_proc, demo_cols)
                else:
                    st.warning(
                        "No suitable demographic columns found in the dataset."
                    )
                    demo_cols = []
            else:
                st.warning("No suitable demographic columns found.")
                demo_cols = []

            # Radar chart (only if group_col is available)
            if group_col:
                metrics = [
                    "demographic.age_at_index",
                    "diagnoses.ajcc_pathologic_stage",
                    "treatments.therapeutic_agents",
                ]
                plot_radar_chart(
                    df_proc,
                    group_col,
                    metrics,
                )

            # Download
            download_processed_csv(df_proc)

            # Audit + Visualizations + Report
            audit_and_visualize_fairness(df_proc, recommendations)

            # 🔹 Modeling
            if enable_modeling == "Yes" and group_col:
                # Get the index for the target column, defaulting to 0 if not found
                target_index = 0
                if st.session_state.target_col in df_proc.columns:
                    try:
                        loc_result = df_proc.columns.get_loc(
                            st.session_state.target_col
                        )
                        target_index = (
                            loc_result if isinstance(loc_result, int) else 0
                        )
                    except (KeyError, ValueError):
                        target_index = 0

                target_col = st.selectbox(  # type: ignore
                    "🎯 Select target column",
                    df_proc.columns,
                    index=int(target_index),
                )
                st.session_state.target_col = target_col

                if target_col:  # if selected, run modeling
                    run_modeling_and_fairness(df_proc, target_col, demo_cols)

    else:
        st.info("⬅️ Please upload a dataset to begin.")


def button_clicked(key):
    if st.button("🚀 Apply Recommended Preprocessing", key=key):
        st.session_state[key + "_clicked"] = True
    return st.session_state.get(key + "_clicked", False)


def extract_valid_demo_cols(candidate_info, df_columns):
    if (
        isinstance(candidate_info, list)
        and candidate_info
        and isinstance(candidate_info[0], tuple)
        and len(candidate_info[0]) == 3
    ):
        demo_cols = [col for col, _, _ in candidate_info]
    else:
        demo_cols = candidate_info  # fallback if already flat list
    return [col for col in demo_cols if col in df_columns]


if __name__ == "__main__":
    main()
