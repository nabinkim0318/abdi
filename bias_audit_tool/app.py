import os
import sys
import traceback
from collections import defaultdict

import pandas as pd
import streamlit as st
from report.report_generator import generate_pdf_report
from sklearn.metrics import roc_auc_score
from utils.model_selector import run_basic_modeling
from utils.preprocess import recommend_preprocessing
from utils.recommend_columns import identify_demographic_columns
from utils.summary import basic_df_summary
from utils.summary import summarize_categories
from utils.transform import apply_preprocessing
from visualization.visualization import show_groupwise_visualizations
from visualization.visualization import show_visualizations

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


st.set_page_config(page_title="Bias Audit Tool", layout="wide")


# ===== Sidebar =====
st.sidebar.title("📊 Bias Audit Assistant")
uploaded_file = st.sidebar.file_uploader("### 1️⃣ Upload Dataset", type="csv")

enable_modeling = st.sidebar.radio("🤖 Run ML Model?", ["No", "Yes"])


# ===== Main Panel =====
st.title("🧪 Bias Audit Dashboard")


def main():
    if uploaded_file is not None:
        try:
            df = pd.read_csv(
                uploaded_file,
                low_memory=False,
                na_values=["--", "NA", "N/A", "null"],
            )

            basic_df_summary(df)

            st.success("✅ File loaded successfully")
            st.markdown("#### 📄 Original Data Preview")
            st.dataframe(df.head())

            # 🔹 Preprocessing recommendations
            st.markdown("### 🧠 Recommended Preprocessing")
            recommendations = recommend_preprocessing(df)

            with st.expander("📋 Show Detailed Column Recommendations"):
                grouped_recs = defaultdict(list)
                for col, rec in recommendations.items():
                    category = col.split(".")[0] if "." in col else "project"
                    grouped_recs[category].append((col, rec))

                for category, items in grouped_recs.items():
                    with st.expander(f"📁 {category}"):
                        for col, rec in items:
                            st.markdown(f"🔧 **{col}** → _{rec}_")

            # 🔹 Category summary table
            summary_df = summarize_categories(df, recommendations)
            st.markdown("### 📊 Preprocessing Recommendation Summary")
            st.dataframe(summary_df, use_container_width=True)

            st.markdown("### ⚙️ Preprocessing Options")

            enable_scaling = st.checkbox(
                "🔧 Apply Scaling to numeric columns (MinMaxScaler)",
                value=True,
                help="Rescales numeric features between 0 and 1. "
                "Recommended for ML modeling.",
            )

            enable_encoding = st.checkbox(
                "🔧 Encode categorical columns",
                value=True,
                help="Converts text columns into numeric format "
                "(e.g., OneHot or Label encoding).",
            )

            handle_missing = st.checkbox(
                "🧩 Handle missing values automatically",
                value=True,
                help="Impute missing numeric values with mean, "
                "categorical with mode. Drop columns with >95% missing.",
            )

            # 🔹 Apply preprocessing
            show_logs = st.checkbox(
                "🪵 Show detailed preprocessing logs",
                value=False,
            )

            if st.button("🚀 Apply Recommended Preprocessing"):
                st.caption(
                    f"🔧 Applied options: Scaling = {enable_scaling}, "
                    f"Encoding = {enable_encoding}, "
                    f"Missing Handling = {handle_missing}"
                )
                orig_shape = df.shape
                df_proc = apply_preprocessing(df, recommendations, show_logs)
                st.write(
                    f"🔄 Data shape changed from `{orig_shape}` → `{df_proc.shape}`"
                )
                st.success("✅ Preprocessing Applied!")
                st.dataframe(df_proc.head())

                st.markdown("### 🧬 Demographic Column Audit")

                demographic_candidates = identify_demographic_columns(df_proc)
                st.sidebar.markdown("#### 🧬 Auto-Detected Demographics")
                st.sidebar.write(
                    ", ".join(demographic_candidates) or "❌ None found"
                )

                selected_demo_cols = st.multiselect(
                    "👥 Select Demographic Columns for Group-wise Audit",
                    demographic_candidates,
                    default=demographic_candidates,
                )

                target_col = None
                if selected_demo_cols:

                    st.markdown("### 👥 Demographic Group-wise Analysis")
                    target_col = st.selectbox(
                        "🎯 Select Target Column (Optional)", df_proc.columns
                    )
                    show_groupwise_visualizations(
                        df_proc, selected_demo_cols, target_col
                    )

                # 📥 Download processed CSV
                csv_buffer = df_proc.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Processed Data",
                    csv_buffer,
                    "processed_data.csv",
                    "text/csv",
                )

                # 🔹 Select columns for audit
                audit_cols = st.sidebar.multiselect(
                    "### 3️⃣ Select Columns for Audit", df_proc.columns
                )
                if audit_cols:
                    st.markdown("### 📌 Preprocessing Options (Manual Override)")
                    for col in audit_cols:
                        st.selectbox(
                            f"Preprocessing Strategy for `{col}`",
                            options=[
                                "None",
                                "LabelEncoder",
                                "OneHotEncoder",
                                "MinMaxScaler",
                                "Log1pTransform",
                            ],
                            key=f"prep_{col}",  # unique key
                        )

                    # 🔹 Visualization
                    st.markdown("### 📊 Visualizations")
                    show_visualizations(df_proc, audit_cols)

                    st.markdown("### 📥 Export Report")
                    st.caption(
                        "Download a detailed PDF report of the selected audit "
                        "columns."
                    )
                    if st.button("📤 Export PDF Report"):
                        if df_proc is not None and audit_cols:
                            pdf_buffer = generate_pdf_report(
                                df_proc, audit_cols, recommendations
                            )
                            st.download_button(
                                "📥 Download PDF Report",
                                pdf_buffer,
                                "bias_audit_report.pdf",
                                mime="application/pdf",
                            )
                        else:
                            st.warning(
                                "⚠️ Please select columns and apply "
                                "preprocessing first."
                            )

                else:
                    st.info(
                        "⬅️ Select columns from the "
                        "sidebar to audit and visualize."
                    )
            else:
                st.info("🚀 Please apply preprocessing to enable audit.")

            # 🔹 Modeling
            if enable_modeling == "Yes" and target_col:
                X = df_proc.drop(columns=[target_col])
                y = df_proc[target_col]
                results = run_basic_modeling(X, y)

                st.markdown("### 🔍 Classification Report")
                st.dataframe(results["report"])

                if results["y_prob"] is not None:
                    st.markdown(
                        "📈 ROC AUC: {:.2f}".format(
                            roc_auc_score(results["y_test"], results["y_prob"])
                        )
                    )

        except Exception as e:
            st.error(f"❌ Error loading or processing the file:\n\n{e}")
            with st.expander("🔍 Show full error details"):
                st.text(traceback.format_exc())
    else:
        st.info("⬅️ Please upload a dataset to begin.")


main()
