import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from utils.preprocess import recommend_preprocessing
from utils.summary import summarize_categories
from utils.transform import apply_preprocessing

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


def show_visualizations(df, audit_cols):
    for col in audit_cols:
        st.markdown(f"#### 🔍 Visualizations for `{col}`")

        if df[col].dropna().empty:
            st.warning(f"⚠️ Column `{col}` has only NaNs.")
            continue

        # Histogram
        fig1, ax1 = plt.subplots()
        sns.histplot(x=df[col].dropna(), kde=True, ax=ax1)
        st.pyplot(fig1)

        # Boxplot
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=df[col].dropna(), ax=ax2)
        st.pyplot(fig2)

    # NaN Heatmap (entire df)
    st.markdown("#### 🔥 Missing Value Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 0.25 * len(df.columns)))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax3)
    st.pyplot(fig3)


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
                df_processed = apply_preprocessing(df, recommendations, show_logs)
                st.success("✅ Preprocessing Applied!")
                st.markdown("#### ✅ Processed Data Preview")
                st.dataframe(df_processed.head())

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
                    show_visualizations(df_processed, audit_cols)
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
