import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def show_visualizations(df, audit_cols):
    sns.set_theme(style="whitegrid", palette="pastel")

    for col in audit_cols:
        st.markdown(f"#### 🔍 Visualizations for `{col}`")

        if df[col].dropna().empty:
            st.warning(f"⚠️ Column `{col}` has only NaNs.")
            continue

        dtype = df[col].dropna().dtype
        unique_vals = df[col].nunique(dropna=True)

        if dtype == "object" or unique_vals < 10:
            # 🔢 Countplot for categorical
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(
                x=col,
                data=df,
                order=df[col].value_counts().index,
                ax=ax,
                palette="pastel",
            )
            ax.set_title(f"Value Counts: {col}", fontsize=14)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)
            plt.close(fig)

        else:
            # 📈 Histogram
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True, ax=ax1, color="#4C72B0")
            ax1.set_title(f"Distribution: {col}", fontsize=14)
            ax1.set_xlabel(col, fontsize=12)
            st.pyplot(fig1)
            plt.close(fig1)

            # 📦 Boxplot
            fig2, ax2 = plt.subplots(figsize=(8, 2))
            sns.boxplot(x=df[col].dropna(), ax=ax2, color="#55A868")
            ax2.set_title(f"Boxplot: {col}", fontsize=13)
            st.pyplot(fig2)
            plt.close(fig2)

    # 🔥 NaN heatmap
    st.markdown("#### 🔥 Missing Value Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 0.3 * len(df.columns)))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax3, cmap="coolwarm")
    ax3.set_title("Missing Values by Column", fontsize=13)
    st.pyplot(fig3)
    plt.close(fig3)


def show_groupwise_visualizations(df, demo_cols, target_col=None):
    sns.set_theme(style="whitegrid", palette="pastel")

    for col in demo_cols:
        st.markdown(f"#### 👥 Group-wise Distribution by `{col}`")

        # 🔁 if the value is too many, take the top 20
        if df[col].nunique() > 20:
            top_k = df[col].value_counts().nlargest(20).index
            df_plot = df[df[col].isin(top_k)]
        else:
            df_plot = df.copy()

        # 🔄 binning (e.g., when the column is numeric and
        # has many unique values)
        if (
            pd.api.types.is_numeric_dtype(df_plot[col])
            and df_plot[col].nunique() > 20
        ):
            df_plot["__binned__"] = pd.cut(
                df_plot[col],
                bins=[0, 20, 40, 60, 80, 100, 120],
                labels=["0-20", "21-40", "41-60", "61-80", "81-100", "100+"],
            )
            x_col = "__binned__"
        else:
            x_col = col

        # 📉 if the target column is numeric, use boxplot
        if target_col and target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(x=x_col, y=target_col, data=df_plot, ax=ax)
                ax.set_title(f"{target_col} by {col}", fontsize=13)
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.countplot(x=x_col, hue=target_col, data=df_plot, ax=ax)
                ax.set_title(f"{target_col} count per {col}", fontsize=13)
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.countplot(x=x_col, data=df_plot, ax=ax)
            ax.set_title(f"Distribution of {col}", fontsize=13)
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
