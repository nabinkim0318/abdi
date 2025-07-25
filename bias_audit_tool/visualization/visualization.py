import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# === Utility Functions ===

def clean_label(col):
    return col.split(".")[-1].replace("_", " ").title()

def restore_group_from_onehot(df, prefix):
    """
    Restore one-hot encoded column group into a single categorical column.
    E.g., from ['demographic.gender_male', 'demographic.gender_female']
    to 'Male' / 'Female'.
    """
    onehot_cols = [col for col in df.columns if col.startswith(prefix)]
    def get_group(row):
        for col in onehot_cols:
            if row.get(col) == 1:
                return col.replace(prefix, "").replace("_", " ").title()
        return "Unknown"
    return df[onehot_cols].apply(get_group, axis=1)


# === Main Visualization Function ===

def plot_distribution_comparison(df, demographic_cols):
    sns.set_theme(style="whitegrid", palette="pastel")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    plot_index = 0

    # Gender
    if "demographic.gender" in df.columns:
        sns.countplot(x="demographic.gender", data=df, ax=axes[plot_index])
        axes[plot_index].set_title("Gender Distribution")
        plot_index += 1
    else:
        gender_cols = [col for col in df.columns if col.startswith("demographic.gender_")]
        if gender_cols:
            gender_group = restore_group_from_onehot(df, "demographic.gender_")
            sns.countplot(x=gender_group, ax=axes[plot_index])
            axes[plot_index].set_title("Gender Distribution (Restored)")
            plot_index += 1

    # Race
    if "demographic.race" in df.columns:
        sns.countplot(x="demographic.race", data=df, ax=axes[plot_index])
        axes[plot_index].set_title("Race Distribution")
        axes[plot_index].tick_params(axis='x', rotation=45)
        plot_index += 1
    else:
        race_cols = [col for col in df.columns if col.startswith("demographic.race_")]
        if race_cols:
            race_group = restore_group_from_onehot(df, "demographic.race_")
            sns.countplot(x=race_group, ax=axes[plot_index])
            axes[plot_index].set_title("Race Distribution (Restored)")
            axes[plot_index].tick_params(axis='x', rotation=45)
            plot_index += 1

    # Ethnicity
    if "demographic.ethnicity" in df.columns:
        sns.countplot(x="demographic.ethnicity", data=df, ax=axes[plot_index])
        axes[plot_index].set_title("Ethnicity Distribution")
        axes[plot_index].tick_params(axis='x', rotation=30)
        plot_index += 1
    else:
        eth_cols = [col for col in df.columns if col.startswith("demographic.ethnicity_")]
        if eth_cols:
            eth_group = restore_group_from_onehot(df, "demographic.ethnicity_")
            sns.countplot(x=eth_group, ax=axes[plot_index])
            axes[plot_index].set_title("Ethnicity Distribution (Restored)")
            axes[plot_index].tick_params(axis='x', rotation=30)
            plot_index += 1

    # Age (from days_to_birth)
    if "demographic.days_to_birth" in df.columns:
        ages = pd.to_numeric(df["demographic.days_to_birth"], errors="coerce") / -365.25
        sns.histplot(ages.dropna(), bins=30, kde=True, ax=axes[plot_index], color="skyblue")
        axes[plot_index].set_title("Age Distribution (Years)")
        axes[plot_index].set_xlabel("Age (Years)")
        plot_index += 1

    # Hide unused subplots
    for j in range(plot_index, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# === Example Usage in Streamlit App ===

def main():
    st.title("📊 Breast Cancer Clinical Demographic Overview")

    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Auto-detect demo columns or use defaults
        demo_cols = [
            "demographic.gender", "demographic.race",
            "demographic.ethnicity", "demographic.days_to_birth"
        ]
        demo_cols = [col for col in demo_cols if col in df.columns or any(df.columns.str.startswith(col + "_"))]

        if demo_cols:
            plot_distribution_comparison(df, demo_cols)
        else:
            st.warning("No demographic columns found to visualize.")

if __name__ == "__main__":
    main()
