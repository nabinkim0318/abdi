import os
import sys

import pandas as pd
import streamlit as st
from report.report_generator import generate_pdf_report
from utils.preprocess import recommend_preprocessing
from utils.recommend_columns import identify_demographic_columns
from utils.summary import summarize_categories
from utils.transform import apply_preprocessing
from visualization.visualization import show_groupwise_visualizations
from visualization.visualization import show_visualizations

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


st.set_page_config(page_title="Bias Audit Tool", layout="wide")


# ===== Sidebar =====
st.sidebar.title("📊 Bias Audit Assistant")
uploaded_file = st.sidebar.file_uploader("### 1️⃣ Upload Dataset", type="csv")

enable_scaling = st.sidebar.checkbox("🔧 Apply Scaling")
enable_encoding = st.sidebar.checkbox("🔧 Encode Categorical Columns")
enable_modeling = st.sidebar.radio("🤖 Run ML Model?", ["No", "Yes"])
export_btn = st.sidebar.button("📤 Export PDF Report")

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

            st.success("✅ File loaded successfully")
            st.markdown("#### 📄 Original Data Preview")
            st.dataframe(df.head())

            # 🔹 Preprocessing recommendations
            st.markdown("### 🧠 Recommended Preprocessing")
            recommendations = recommend_preprocessing(df)

            with st.expander("📋 Show Detailed Column Recommendations"):
                for col, rec in recommendations.items():
                    st.markdown(f"🔧 **{col}** → _{rec}_")

            # 🔹 Category summary table
            st.markdown("### 📊 Category Summary Table")
            summary_df = summarize_categories(df, recommendations)
            st.dataframe(summary_df)

            # 🔹 Apply preprocessing
            show_logs = st.checkbox(
                "🪵 Show detailed preprocessing logs",
                value=False,
            )

            if st.button("🚀 Apply Recommended Preprocessing"):
                df_proc = apply_preprocessing(df, recommendations, show_logs)
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
                    "### 3️⃣ Select Columns for Audit", df.columns
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

                    if export_btn and df_proc is not None and audit_cols:
                        buffer = generate_pdf_report(
                            df_proc, audit_cols, recommendations
                        )
                        st.download_button(
                            "📥 Download PDF Report",
                            buffer,
                            "bias_audit_report.pdf",
                            mime="application/pdf",
                        )

                else:
                    st.info(
                        "⬅️ Select columns from the "
                        "sidebar to audit and visualize."
                    )
            else:
                st.info("🚀 Please apply preprocessing to enable audit.")

            # 🔹 Modeling
            if enable_modeling == "Yes":
                st.markdown("### 🤖 ML Metrics & Fairness Charts")
                st.write("📉 Placeholder for model performance & fairness metrics.")
                st.write(
                    "⚙️ Placeholder for SHAP or permutation importance by " "group."
                )

        except Exception as e:
            st.error(f"❌ Error loading or processing the file:\n\n{e}")
    else:
        st.info("⬅️ Please upload a dataset to begin.")


main()
