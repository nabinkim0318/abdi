import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


def clean_label(col):
    return col.split(".")[-1].replace("_", " ").title()


def is_id_column(series):
    return series.nunique() >= 0.9 * len(series) and series.dtype == "object"


def is_categorical(series, threshold=10):
    return series.nunique() <= threshold or series.dtype == "object"


def bin_numeric_column(series, bins=5):
    return pd.qcut(series, q=bins, duplicates="drop")


def show_visualizations(df, audit_cols):
    """
    Display various visualizations for auditing selected columns.

    For each column in `audit_cols`, renders:
        - Count plot (for categorical features)
        - Histogram + KDE (for numerical features)
        - Boxplot (for numerical features)

    Additionally shows:
        - Heatmap of missing values across all columns

    Args:
        df (pd.DataFrame): The input DataFrame to visualize.
        audit_cols (list[str]): List of column names to audit visually.

    Displays:
        Streamlit-rendered visual plots for each column.
    """
    sns.set_theme(style="whitegrid", palette="pastel")

    audit_cols = [col for col in audit_cols if not is_id_column(df[col])]

    if len(audit_cols) == 0:
        st.warning(
            "⚠️ No meaningful columns to visualize. "
            "ID-like columns were automatically removed."
        )
        return

    # bin numeric columns
    for col in audit_cols:
        df_plot = df.copy()
        if (
            pd.api.types.is_numeric_dtype(df_plot[col])
            and df_plot[col].nunique() > 20
        ):
            df_plot[col] = bin_numeric_column(df_plot[col])

    for col in audit_cols:
        st.markdown(f"#### 🔍 Visualizations for `{col}`")

        if df[col].dropna().empty:
            st.warning(f"⚠️ Column `{col}` has only NaNs.")
            continue

        if is_categorical(df[col]):
            # 🔢 Countplot for categorical
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(
                x=col,
                data=df,
                order=df[col].value_counts().index,
                ax=ax,
                palette="pastel",
            )
            label = clean_label(col)
            ax.set_title(f"Value Counts: {label}", fontsize=14)
            ax.set_xlabel(label, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.tick_params(axis="x", rotation=30)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            st.pyplot(fig)
            plt.close(fig)

        else:
            # 📈 Histogram
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True, ax=ax1, color="#4C72B0")
            label = clean_label(col)
            ax1.set_title(f"Distribution: {label}", fontsize=14)
            ax1.set_xlabel(label, fontsize=12)
            st.pyplot(fig1)
            plt.close(fig1)

            # 📦 Boxplot
            fig2, ax2 = plt.subplots(figsize=(8, 2))
            sns.boxplot(x=df[col].dropna(), ax=ax2, color="#55A868")
            label = clean_label(col)
            ax2.set_title(f"Boxplot: {label}", fontsize=13)
            st.pyplot(fig2)
            plt.close(fig2)

    # 🔥 NaN heatmap
    st.markdown("#### 🔥 Missing Value Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 0.3 * len(df.columns)))
    sns.heatmap(
        df.isnull(),
        cbar=False,
        yticklabels=False,
        ax=ax3,
        cmap="coolwarm",
    )
    ax3.set_title("Missing Values by Column", fontsize=13)
    st.pyplot(fig3)
    plt.close(fig3)


def show_groupwise_visualizations(df, demo_cols, target_col=None):
    """
    Display meaningful group-wise visualizations based on target type.

    For each demographic column:
        - If no target: Show demographic distribution histogram
        - If numeric target: Show boxplot by demographic group
        - If categorical target: Show stacked bar plot with proportions

    Args:
        df (pd.DataFrame): The input DataFrame.
        demo_cols (list[str]): List of demographic columns for group-wise analysis.
        target_col (str, optional): Target variable to plot against groups.

    Displays:
        Streamlit-rendered visualizations per demographic column.
    """
    sns.set_theme(style="whitegrid", palette="pastel")

    # remove id columns from demo_cols
    demo_cols = [col for col in demo_cols if not is_id_column(df[col])]

    if len(demo_cols) == 0:
        st.warning(
            "⚠️ No meaningful columns to visualize. "
            "ID-like columns were automatically removed."
        )
        return

    # Check target_col
    if target_col:
        if target_col not in df.columns:
            st.warning(f"Target column `{target_col}` not found in dataset.")
            target_col = None
        elif is_id_column(df[target_col]):
            st.warning(
                f"Target column `{target_col}` looks like an ID column "
                "and will be ignored."
            )
            target_col = None
        # Additional aggressive check for ID-like columns
        elif any(
            id_word in target_col.lower()
            for id_word in ["id", "case", "patient", "subject"]
        ):
            st.warning(
                f"Target column `{target_col}` appears to be an ID column "
                "and will be ignored."
            )
            target_col = None

    for col in demo_cols:
        st.markdown(f"#### 👥 Group-wise Distribution by `{clean_label(col)}`")

        # 🔁 if the value is too many, take the top 20
        if df[col].nunique() > 20:
            top_k = df[col].value_counts().nlargest(20).index
            df_plot = df[df[col].isin(top_k)]
        else:
            df_plot = df.copy()

        # 🔄 binning (e.g., when the column is numeric and has many unique values)
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

        # 📊 Visualization based on target type
        if target_col and target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                # Numeric target: Boxplot
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(x=x_col, y=target_col, data=df_plot, ax=ax)
                ax.set_title(
                    f"{clean_label(target_col)} by {clean_label(col)}",
                    fontsize=13,
                )
                ax.set_xlabel(clean_label(col), fontsize=12)
                ax.set_ylabel(clean_label(target_col), fontsize=12)
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                # Categorical target: Stacked bar plot with proportions
                prop_df = (
                    df_plot.groupby([x_col, target_col])
                    .size()
                    .groupby(level=0)
                    .apply(lambda x: x / x.sum())
                    .reset_index(name="proportion")
                )
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    data=prop_df,
                    x=x_col,
                    y="proportion",
                    hue=target_col,
                    ax=ax,
                )
                ax.set_title(
                    f"{clean_label(target_col)} Proportion per {clean_label(col)}",
                    fontsize=13,
                )
                ax.set_xlabel(clean_label(col), fontsize=12)
                ax.set_ylabel("Proportion", fontsize=12)
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            # No target: Simple demographic distribution
            fig, ax = plt.subplots(figsize=(10, 4))
            if df_plot[col].dtype == "object" or df_plot[col].nunique() < 10:
                sns.countplot(x=x_col, data=df_plot, ax=ax)
                ax.set_title(
                    f"Distribution of {clean_label(col)}",
                    fontsize=13,
                )
                ax.set_xlabel(clean_label(col), fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
            else:
                sns.histplot(x=df_plot[col], ax=ax, bins=20)
                ax.set_title(
                    f"Distribution of {clean_label(col)}",
                    fontsize=13,
                )
                ax.set_xlabel(clean_label(col), fontsize=12)
                ax.set_ylabel("Count", fontsize=12)
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def show_demographic_overview(df, demographic_cols):
    sns.set_theme(style="whitegrid", palette="pastel")

    for col in demographic_cols:
        st.markdown(f"#### 📊 Distribution of `{col}`")

        if pd.api.types.is_numeric_dtype(df[col]):
            # Histogram + Boxplot for numeric
            fig, axs = plt.subplots(
                2,
                1,
                figsize=(10, 6),
                gridspec_kw={"height_ratios": [3, 1]},
            )
            sns.histplot(df[col].dropna(), ax=axs[0], kde=True)
            axs[0].set_title(f"Histogram of {col}")
            sns.boxplot(x=df[col].dropna(), ax=axs[1], color="#55A868")
            axs[1].set_title(f"Boxplot of {col}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:
            # Countplot for categorical
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.countplot(
                x=col,
                data=df,
                ax=ax,
                order=df[col].value_counts().index,
            )
            ax.set_title(f"Value Counts of {col}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)
            plt.close(fig)


def restore_group_column(df, prefix, new_col_name):
    """
    Restore one-hot encoded columns to group columns.

    Args:
        df (pd.DataFrame): dataframe to restore.
        prefix (str): One-hot prefix (e.g., 'demographic.race_').
        new_col_name (str): new column name (e.g., 'race_group').

    Returns:
        pd.DataFrame: restored dataframe (original df is not modified).
    """
    onehot_cols = [col for col in df.columns if col.startswith(prefix)]
    if not onehot_cols:
        raise ValueError(f"No columns found with prefix: {prefix}")

    def get_group(row):
        for col in onehot_cols:
            if row[col] == 1:
                return col.replace(prefix, "")
        return "Unknown"

    df_copy = df.copy()
    df_copy[new_col_name] = df_copy[onehot_cols].apply(get_group, axis=1)
    return df_copy


def auto_group_selector(df):
    """
    Automatically select one-hot encoded columns and restore them to group columns.

    Side Effects:
        - Displays success message and data preview in the Streamlit app.
        - Displays error message and traceback if an exception occurs.

    Args:
        df (pd.DataFrame): dataframe to restore.

    Returns:
        DataFrame, str: restored dataframe, restored column name
    """
    st.sidebar.markdown("### 🔄 Auto One-Hot Column Restoration")

    # Find one-hot encoded columns
    prefixes = sorted(
        list(
            {
                col.split("_")[0] + "." + col.split("_")[1] + "_"
                for col in df.columns
                if "_" in col
            }
        )
    )

    if not prefixes:
        st.sidebar.info("No One-hot Encoded Columns Found.")
        return df, None

    # Show available prefixes
    st.sidebar.write(f"Found {len(prefixes)} one-hot encoded column groups:")
    for prefix in prefixes[:3]:  # Show first 3
        st.sidebar.write(f"• {prefix}...")

    # Check if we already have restored data in session state
    if "restored_df" in st.session_state and "restored_col" in st.session_state:
        st.sidebar.success(
            f"✅ Using previously restored column: "
            f"{st.session_state['restored_col']}"
        )
        return (
            st.session_state["restored_df"],
            st.session_state["restored_col"],
        )

    # ✅ Session state key - use unique keys to avoid conflicts
    selected_prefix = st.sidebar.selectbox(
        "Select One-hot Prefix to Restore",
        prefixes,
        key="prefix_selector_unique",
    )

    # Preview: Show which columns will be converted when prefix is selected
    if selected_prefix:
        onehot_cols = [col for col in df.columns if col.startswith(selected_prefix)]
        st.sidebar.markdown(f"**Preview:** {len(onehot_cols)} columns found:")
        for col in onehot_cols[:3]:  # Show first 3
            st.sidebar.markdown(f"• `{col}`")
        if len(onehot_cols) > 3:
            st.sidebar.markdown(f"• ... and {len(onehot_cols) - 3} more")

    new_col_name = st.sidebar.text_input(
        "New Column Name", value="group", key="col_name_input_unique"
    )

    if st.sidebar.button("Restore Group Column", key="restore_button_unique"):
        try:
            df_new = restore_group_column(df, selected_prefix, new_col_name)
            # ✅ save to session_state BEFORE returning
            st.session_state["restored_df"] = df_new
            st.session_state["restored_col"] = new_col_name
            st.session_state["restored_prefix"] = selected_prefix
            st.sidebar.success(f"✅ Restored Column: {new_col_name}")
            # Don't use st.rerun() here as it might cause issues
            return df_new, new_col_name
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            st.sidebar.error(f"Traceback: {str(e)}")
            return df, None

    return df, None


def plot_radar_chart(df, group_col, metrics):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[metrics])
    df_scaled = pd.DataFrame(scaled_data, columns=metrics)

    # ✅ 복원 끝났으니 바로 group_col만 써서 그룹화
    df_scaled[group_col] = df[group_col].values

    avg_by_group = df_scaled.groupby(group_col).mean()

    categories = metrics
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Closing the plot

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for group, row in avg_by_group.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, label=group)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Demographic Group Comparison", fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    st.pyplot(fig)
    plt.close(fig)


def plot_distribution_comparison(result_df, top_n=20):
    fig, ax = plt.subplots()
    idx = result_df.index.astype(str)

    ax.bar(
        idx,
        result_df["Observed_%"],
        width=0.4,
        label="Observed",
        align="edge",
    )
    ax.bar(
        idx,
        result_df["Expected_%"],
        width=-0.4,
        label="Expected",
        align="edge",
    )
    ax.set_xticklabels(idx, rotation=45)
    ax.set_ylabel("Proportion")
    ax.set_title("Observed vs Expected Group Distribution")
    ax.legend()
    fig.tight_layout()
    return fig
