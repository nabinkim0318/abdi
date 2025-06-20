# bias_audit_tool/app.py
import streamlit as st
from utils.ui_helpers import apply_preprocessing_and_display
from utils.ui_helpers import display_preprocessing_recommendations
from utils.ui_helpers import get_user_preprocessing_options
from utils.ui_helpers import run_modeling_and_fairness
from visualization.ui_blocks import audit_and_visualize
from visualization.ui_blocks import download_processed_csv
from visualization.ui_blocks import show_demographic_analysis

from bias_audit_tool.data.data_loader import load_and_preview_data


st.set_page_config(page_title="Bias Audit Tool", layout="wide")


# ===== Sidebar =====
st.sidebar.title("📊 Bias Audit Assistant")
uploaded_file = st.sidebar.file_uploader("### 1️⃣ Upload Dataset", type="csv")

enable_modeling = st.sidebar.radio("🤖 Run ML Model?", ["No", "Yes"])


# ===== Main Panel =====
st.title("🧪 Bias Audit Dashboard")


def main():
    target_col = None
    if uploaded_file is not None:
        df = load_and_preview_data(uploaded_file)
        if df is None:
            st.stop()

        # recommendation and summary
        recommendations = display_preprocessing_recommendations(df)

        # user options
        show_logs = st.checkbox("🪵 Show detailed preprocessing logs", value=False)
        options = get_user_preprocessing_options()

        if st.button("🚀 Apply Recommended Preprocessing"):
            df_proc = apply_preprocessing_and_display(
                df, recommendations, show_logs, options
            )
            st.session_state.df_proc = df_proc  # persist state
            st.session_state.recommendations = recommendations
            st.session_state.preprocessed = True
        else:
            st.session_state.preprocessed = False

        # If preprocessing is done, allow further analysis
        if st.session_state.get("preprocessed") and "df_proc" in st.session_state:
            df_proc = st.session_state.df_proc
            recommendations = st.session_state.recommendations

            # Demographics
            selected_demo_cols, target_col = show_demographic_analysis(df_proc)

            # Download
            download_processed_csv(df_proc)

            # Audit + Visualizations + Report
            audit_and_visualize(df_proc, recommendations)

        # 🔹 Modeling
        if enable_modeling == "Yes" and target_col:
            run_modeling_and_fairness(df_proc, target_col, selected_demo_cols)
        else:
            st.info("🚀 Please apply preprocessing to enable audit.")
    else:
        st.info("⬅️ Please upload a dataset to begin.")


# metric_frame, summary = compute_fairness_metrics(
#     y_true=results["y_test"],
#     y_pred=results["y_pred"],
#     sensitive_features=X[attr].loc[results["y_test"].index]
# )

if __name__ == "__main__":
    main()
