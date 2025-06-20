# bias_audit_tool/app.py
import traceback

import streamlit as st
from sklearn.metrics import roc_auc_score
from utils.data_loader import load_and_preview_data
from utils.model_selector import run_basic_modeling
from utils.ui_helpers import apply_preprocessing_and_display
from utils.ui_helpers import display_preprocessing_recommendations
from utils.ui_helpers import get_user_preprocessing_options
from visualization.ui_blocks import audit_and_visualize
from visualization.ui_blocks import download_processed_csv
from visualization.ui_blocks import show_demographic_analysis


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
            st.markdown("## 🧠 Machine Learning Modeling")

            X = df_proc.drop(columns=[target_col])
            y = df_proc[target_col]
            try:
                results = run_basic_modeling(X, y)

                st.markdown("### 🔍 Classification Report")
                st.dataframe(results["report"])

                if results["y_prob"] is not None:
                    st.markdown(
                        "📈 ROC AUC: {:.2f}".format(
                            roc_auc_score(results["y_test"], results["y_prob"])
                        )
                    )
            except Exception:
                st.error("❌ Modeling failed.")
                st.text(traceback.format_exc())
                return
        else:
            st.info("🚀 Please apply preprocessing to enable audit.")
    else:
        st.info("⬅️ Please upload a dataset to begin.")


if __name__ == "__main__":
    main()
