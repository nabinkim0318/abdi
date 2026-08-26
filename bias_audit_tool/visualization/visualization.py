import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


def clean_label(col):
    return col.split(".")[-1].replace("_", " ").title()


def is_id_column(series):
    return series.nunique() >= 0.9 * len(series) and series.dtype == "object"


def show_visualizations(df, audit_cols):
    sns.set_theme(style="whitegrid", palette="pastel")
    audit_cols = [
        col for col in audit_cols if col in df.columns and not is_id_column(df[col])
    ]

    if not audit_cols:
        st.warning(
            "⚠️ No meaningful columns to visualize. ID-like columns were removed."
        )
        return

    valid_cols = [col for col in audit_cols if not df[col].dropna().empty]
    if not valid_cols:
        st.warning("⚠️ No valid columns found for visualization.")
        return

    st.markdown("#### 🔍 Demographic Distributions")

    n_cols = min(3, len(valid_cols))
    n_rows = (len(valid_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(valid_cols):
        ax = axes[i]

        if "age_at_index" in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
            df_temp = df.copy()
            df_temp[f"{col}_binned"] = pd.cut(
                df_temp[col],
                bins=[0, 25, 50, 75, float("inf")],
                labels=["<25", "25-49", "50-75", ">75"],
                right=False,
            )
            df_temp[f"{col}_binned"] = df_temp[f"{col}_binned"].astype(str)
            df_temp[f"{col}_binned"] = df_temp[f"{col}_binned"].replace(
                "nan", "Unknown"
            )

            sns.countplot(
                data=df_temp,
                x=f"{col}_binned",
                hue=f"{col}_binned",
                ax=ax,
                palette="Set2",
                legend=False,
            )
            ax.set_title(f"{clean_label(col)} (Binned)", fontsize=12)
            ax.set_xlabel("Age Groups", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
        else:
            df_temp = df.copy()
            df_temp[col] = df_temp[col].astype(str)

            sns.countplot(
                data=df_temp,
                x=col,
                hue=col,
                ax=ax,
                palette="Set2",
                legend=False,
            )
            ax.set_title(f"{clean_label(col)}", fontsize=12)
            ax.set_xlabel(clean_label(col), fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=9)

    for j in range(len(valid_cols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_distribution_comparison(result_df, top_n=20):
    if result_df is None or result_df.empty:
        return None
    if "Group" not in result_df.columns:
        return None

    plot_df = result_df.head(top_n).copy()
    labels = plot_df["Group"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, plot_df["Observed_%"], width=0.4, label="Observed", align="edge")
    ax.bar(
        labels,
        plot_df["Expected_%"],
        width=-0.4,
        label="Expected",
        align="edge",
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Observed vs Expected Group Distribution")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig
