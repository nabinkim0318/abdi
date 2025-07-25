import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def clean_label(col):
    return col.split(".")[-1].replace("_", " ").title()


def is_id_column(series):
    return series.nunique() >= 0.9 * len(series) and series.dtype == "object"


def is_categorical(series, threshold=10):
    return series.nunique() <= threshold or series.dtype == "object"


def plot_grouped_bar(df, group_cols, title, palette):
    counts = df[group_cols].sum().reset_index()
    counts.columns = ["Group", "Count"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=counts, x="Count", palette=palette, ax=ax)
    ax.set_title(title, fontsize=14)
    st.pyplot(fig)
    plt.close(fig)


def show_visualizations(df, audit_cols):
    sns.set_theme(style="whitegrid", palette="pastel")
    audit_cols = [col for col in audit_cols if col in df.columns and not is_id_column(df[col])]

    if not audit_cols:
        st.warning("⚠️ No meaningful columns to visualize. ID-like columns were removed.")
        return

    # Dummy-based one‑hot groups (kept)
    gender_cols = ["demographic.gender_female", "demographic.gender_male"]
    race_cols = [
        "demographic.race_white",
        "demographic.race_black or african american",
        "demographic.race_asian",
        "demographic.race_american indian or alaska native",
        "demographic.race_not reported",
    ]
    ethnicity_cols = [
        "demographic.ethnicity_hispanic or latino",
        "demographic.ethnicity_not reported",
    ]

    demographic_columns, other_columns = [], []
    for col in audit_cols:
        if col in gender_cols + race_cols + ethnicity_cols or "demographic" in col.lower():
            demographic_columns.append(col)
        else:
            other_columns.append(col)

    # Include age explicitly if present (belt and suspenders)
    if "demographic.age_at_index" in df.columns and "demographic.age_at_index" not in demographic_columns:
        demographic_columns.append("demographic.age_at_index")

    if demographic_columns:
        st.markdown("#### 🔍 Demographic Distributions")

        # filter out all‑NaN
        valid_demo_cols = [c for c in demographic_columns if c in df.columns and not df[c].dropna().empty]
        if not valid_demo_cols:
            st.warning("⚠️ All demographic columns are empty.")
            return

        n_cols = min(3, len(valid_demo_cols))
        n_rows = (len(valid_demo_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for i, col in enumerate(valid_demo_cols):
            ax = axes[i]
            series = df[col]

            # Numeric demographic (e.g., age)
            if pd.api.types.is_numeric_dtype(series):
                # Histogram
                sns.histplot(series.dropna(), bins=20, kde=True, ax=ax)
                ax.set_title(f"{clean_label(col)} (Histogram)", fontsize=12)
                ax.set_xlabel(clean_label(col), fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
            else:
                # Categorical / one‑hot -> countplot
                temp = series.astype(str)
                sns.countplot(x=temp, ax=ax, order=temp.value_counts().index)
                ax.set_title(f"{clean_label(col)}", fontsize=12)
                ax.set_xlabel(clean_label(col), fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                ax.tick_params(axis="x", rotation=45, labelsize=9)

        for j in range(len(valid_demo_cols), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)




def show_groupwise_visualizations(df, demo_cols, target_col=None):
    """
    Display meaningful group-wise visualizations based on target type as subplots.

    For each demographic column:
        - If no target: Show demographic distribution histogram
        - If numeric target: Show boxplot by demographic group
        - If categorical target: Show stacked bar plot with proportions

    Args:
        df (pd.DataFrame): The input DataFrame.
        demo_cols (list[str]): List of demographic columns for group-wise analysis.
        target_col (str, optional): Target variable to plot against groups.

    Displays:
        Streamlit-rendered visualizations per demographic column as subplots.
    """
    sns.set_theme(style="whitegrid", palette="pastel")
    demo_cols = [
        col
        for col in demo_cols
        if "demographic" in col
        or any(key in col for key in ["gender", "race", "ethnicity"])
    ]
    demo_cols = [col for col in demo_cols if not is_id_column(df[col])]

    if not demo_cols:
        st.warning(
            "⚠️ No meaningful columns to visualize. " "ID-like columns were removed."
        )
        return

    def valid_target(col):
        if col not in df.columns:
            return False
        if is_id_column(df[col]):
            return False
        if any(x in col.lower() for x in ["id", "case", "patient", "subject"]):
            return False
        return True

    if target_col and not valid_target(target_col):
        st.warning(
            f"⚠️ Target column `{target_col}` is invalid or looks like an "
            "ID column."
        )
        target_col = None

    # Filter valid columns
    valid_demo_cols = []
    for col in demo_cols:
        if col not in df.columns:
            logging.warning(f"⚠️ Column '{col}' not in dataframe. Skipping plot.")
            continue
        valid_demo_cols.append(col)
    
    if not valid_demo_cols:
        st.warning("⚠️ No valid demographic columns found.")
        return

    st.markdown("#### 👥 Group-wise Distributions")
    
    # Calculate subplot dimensions
    n_cols = min(3, len(valid_demo_cols))  # Max 3 columns
    n_rows = (len(valid_demo_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for i, col in enumerate(valid_demo_cols):
        ax = axes[i]
        
        df_plot = df.copy()
        if df[col].nunique() > 20:
            top_k = df[col].value_counts().nlargest(20).index
            df_plot = df[df[col].isin(top_k)]

        # if (
        #     pd.api.types.is_numeric_dtype(df_plot[col])
        #     and df_plot[col].nunique() > 20
        # ):
        #     df_plot["__binned__"] = pd.cut(
        #         df_plot[col],
        #         bins=[0, 20, 40, 60, 80, 100, 120],
        #         labels=["0-20", "21-40", "41-60", "61-80", "81-100", "100+"],
        #     )
        #     x_col = "__binned__"
        # else:
        #     x_col = col

        # inside for loop, before determining x_col
        if pd.api.types.is_numeric_dtype(df_plot[col]) and df_plot[col].nunique() > 20:
            if "age" in col.lower():
                # age-specific bins
                df_plot["__binned__"] = pd.cut(
                    df_plot[col],
                    bins=[0, 20, 30, 40, 50, 60, 70, 120],
                    labels=["0-20", "21-30", "31-40", "41-50", "51-60", "61-70", "70+"],
                    include_lowest=True,
                )
            else:
                df_plot["__binned__"] = pd.cut(
                    df_plot[col],
                    bins=[0, 20, 40, 60, 80, 100, 120],
                    labels=["0-20", "21-40", "41-60", "61-80", "81-100", "100+"],
                    include_lowest=True,
                )
            x_col = "__binned__"
        else:
            x_col = col


        if target_col:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                sns.boxplot(x=x_col, y=target_col, data=df_plot, ax=ax)
                ax.set_title(
                    f"{clean_label(target_col)} by {clean_label(col)}",
                    fontsize=11,
                )
                ax.set_xlabel(clean_label(col), fontsize=10)
                ax.set_ylabel(clean_label(target_col), fontsize=10)
            else:
                prop_df = (
                    df_plot.groupby([x_col, target_col])
                    .size()
                    .groupby(level=0)
                    .apply(lambda x: x / x.sum())
                    .reset_index(name="proportion")
                )
                sns.barplot(
                    data=prop_df, x=x_col, y="proportion", hue=target_col, ax=ax
                )
                ax.set_title(
                    f"{clean_label(target_col)} Proportion per {clean_label(col)}",
                    fontsize=11,
                )
                ax.set_xlabel(clean_label(col), fontsize=10)
                ax.set_ylabel("Proportion", fontsize=10)
        else:
            if df_plot[col].dtype == "object" or df_plot[col].nunique() < 10:
                sns.countplot(x=x_col, data=df_plot, ax=ax)
                ax.set_ylabel("Count", fontsize=10)
            else:
                sns.histplot(x=df_plot[col], ax=ax, bins=20)
                ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"{clean_label(col)}", fontsize=11)
            ax.set_xlabel(clean_label(col), fontsize=10)

        ax.tick_params(axis="x", rotation=30, labelsize=9)

    # Hide empty subplots
    for j in range(len(valid_demo_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_demographic_overview(df, demographic_cols):
    sns.set_theme(style="whitegrid", palette="pastel")
    
    # Filter valid demographic columns
    valid_demo_cols = []
    for col in demographic_cols:
        if col in df.columns and not df[col].dropna().empty:
            valid_demo_cols.append(col)
    
    if not valid_demo_cols:
        st.warning("⚠️ No valid demographic columns found for overview.")
        return
    
    num_cols = len(valid_demo_cols)
    fig, axs = plt.subplots(nrows=num_cols, ncols=2, figsize=(12, 4 * num_cols))
    
    # Handle single row case
    if num_cols == 1:
        axs = axs.reshape(1, -1)

    for i, col in enumerate(valid_demo_cols):
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), ax=axs[i, 0], kde=True)
            axs[i, 0].set_title(f"Histogram of {clean_label(col)}")
            sns.boxplot(x=df[col].dropna(), ax=axs[i, 1], color="#55A868")
            axs[i, 1].set_title(f"Boxplot of {clean_label(col)}")
        else:
            sns.countplot(
                x=col, data=df, ax=axs[i, 0], order=df[col].value_counts().index
            )
            axs[i, 0].set_title(f"Value Counts of {clean_label(col)}")
            axs[i, 0].tick_params(axis="x", rotation=30)
            axs[i, 1].axis("off")

    plt.tight_layout()
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


def auto_group_selector(df, merge_all=True):
    """
    Either merge all one-hot columns automatically (if merge_all=True),
    or allow manual selection via Streamlit sidebar UI.

    Returns:
        - df_merged (pd.DataFrame): Processed DataFrame
        - mapping (dict) if merge_all else restored column name (str)
    """

    def extract_prefixes(df):
        return sorted(
            {
                ".".join(col.split("_")[0:2])  # e.g., demographic.gender
                for col in df.columns
                if "_" in col and len(col.split("_")) >= 2
            }
        )

    # === Manual Mode via Sidebar ===
    st.sidebar.markdown("### 🔄 Auto One-Hot Column Restoration")

    prefixes = extract_prefixes(df)

    if not prefixes:
        st.sidebar.info("No One-hot Encoded Columns Found.")
        return df, None

    st.sidebar.write(f"Found {len(prefixes)} one-hot encoded column groups:")
    for prefix in prefixes[:3]:
        st.sidebar.write(f"• {prefix}_...")

    if "restored_df" in st.session_state and "restored_col" in st.session_state:
        st.sidebar.success(
            f"✅ Using previously restored column: "
            f"{st.session_state['restored_col']}"
        )
        return st.session_state["restored_df"], st.session_state["restored_col"]

    selected_prefix = st.sidebar.selectbox(
        "Select One-hot Prefix to Restore",
        prefixes,
        key="prefix_selector_unique",
    )

    if selected_prefix:
        onehot_cols = [
            col for col in df.columns if col.startswith(selected_prefix + "_")
        ]
        st.sidebar.markdown(f"**Preview:** {len(onehot_cols)} columns found:")
        for col in onehot_cols[:3]:
            st.sidebar.markdown(f"• `{col}`")
        if len(onehot_cols) > 3:
            st.sidebar.markdown(f"• ... and {len(onehot_cols) - 3} more")

    new_col_name = st.sidebar.text_input(
        "New Column Name", value="group", key="col_name_input_unique"
    )

    if st.sidebar.button("Restore Group Column", key="restore_button_unique"):
        try:
            df_new = restore_group_column(df, selected_prefix + "_", new_col_name)
            st.session_state["restored_df"] = df_new
            st.session_state["restored_col"] = new_col_name
            st.session_state["restored_prefix"] = selected_prefix
            st.sidebar.success(f"✅ Restored Column: {new_col_name}")
            return df_new, new_col_name
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            return df, None

    return df, None


def plot_radar_chart(df, group_col, metrics):
    try:
        valid_metrics = [metric for metric in metrics if metric in df.columns]

        if not valid_metrics:
            st.warning(
                "No valid metrics found for radar chart. "
                "Please check column names."
            )
            return

        if len(valid_metrics) < 2:
            st.warning("Need at least 2 metrics for radar chart.")
            return

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[valid_metrics])
        df_scaled = pd.DataFrame(scaled_data, columns=valid_metrics)
        df_scaled[group_col] = df[group_col].values

        avg_by_group = df_scaled.groupby(group_col).mean()

        categories = valid_metrics
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for group, row in avg_by_group.iterrows():
            values = row.tolist()
            if len(values) == N:
                values.append(values[0])
            elif len(values) != N + 1:
                continue
            if len(angles) != len(values):
                continue
            ax.plot(angles, values, label=group)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title("Demographic Group Comparison", fontsize=16)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Error creating radar chart: {str(e)}")
        st.write("DataFrame columns:", list(df.columns))
        st.write("Group column:", group_col)
        st.write("Requested metrics:", metrics)


def plot_distribution_comparison(result_df, top_n=20):
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = result_df.index.astype(str)

    ax.bar(idx, result_df["Observed_%"], width=0.4, label="Observed", align="edge")
    ax.bar(idx, result_df["Expected_%"], width=-0.4, label="Expected", align="edge")

    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels(idx, rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Observed vs Expected Group Distribution")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig
