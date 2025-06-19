import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


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

    # NaN Heatmap
    st.markdown("#### 🔥 Missing Value Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 0.25 * len(df.columns)))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax3)
    st.pyplot(fig3)


def show_groupwise_visualizations(df, demo_cols, target_col=None):
    for col in demo_cols:
        st.markdown(f"#### 👥 Group-wise Distribution by `{col}`")
        if target_col and target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], y=df[target_col], ax=ax)
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                sns.countplot(x=df[col], hue=df[target_col], ax=ax)
                st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            sns.countplot(x=df[col], ax=ax)
            st.pyplot(fig)
