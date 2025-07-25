import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, wasserstein_distance

# === Config ===
st.set_page_config(page_title="Fairness Audit", layout="wide")
sns.set_theme(style="whitegrid")

# === Benchmark: TNBC Example ===
TNBC_RACE_BENCHMARK = {
    'Black': 0.36,
    'White': 0.19,
    'Hispanic': 0.16,
    'AIAN': 0.16,
    'Asian': 0.13
}


# === Justifiability Logic (customize this) ===
def default_justifiability_fn(row):
    if row["Group"] == "Black":
        return "Yes"  # Clinical rationale
    return "Unknown"


# === Utilities ===
def compare_distributions(p, q, method="kl"):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    if method == "kl":
        epsilon = 1e-8
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
    p_norm = p / p.sum()
    q_norm = q / q.sum()
    if method == "kl":
        return entropy(p_norm, q_norm)
    elif method == "wasserstein":
        return wasserstein_distance(p_norm, q_norm)
    else:
        raise ValueError("method must be 'kl' or 'wasserstein'")


def compute_input_fairness(
    df: pd.DataFrame,
    demographic_col: str,
    benchmark_distribution: dict,
    threshold_low: float = 0.8,
    threshold_high: float = 1.25,
    justifiability_fn=None,
) -> pd.DataFrame:
    observed_counts = df[demographic_col].value_counts(dropna=False)
    total = len(df)
    observed_percent = observed_counts / total

    result_df = pd.DataFrame({
        "Observed_Count": observed_counts,
        "Observed_%": observed_percent,
    })

    result_df["Expected_%"] = result_df.index.map(benchmark_distribution)
    result_df["Benchmark_Valid"] = ~result_df["Expected_%"].isnull()
    result_df["Expected_%"] = result_df["Expected_%"].fillna(0.0001)

    result_df["Disparity_Ratio"] = result_df["Observed_%"] / result_df["Expected_%"]
    result_df["Magnitude_Flag"] = result_df["Disparity_Ratio"].apply(
        lambda x: "Not Fair" if x < threshold_low or x > threshold_high else "Fair"
    )
    result_df["Absolute_Difference"] = abs(result_df["Observed_%"] - result_df["Expected_%"])

    if justifiability_fn:
        result_df["Justifiable?"] = result_df.apply(justifiability_fn, axis=1)
    else:
        result_df["Justifiable?"] = "Unknown"

    result_df["Framework_Flag"] = result_df.apply(
        lambda row: "⚠️ Check" if (
            not row["Benchmark_Valid"] or
            row["Magnitude_Flag"] == "Not Fair" or
            row["Justifiable?"] != "Yes"
        ) else "✅ OK", axis=1
    )

    obs_dist = result_df["Observed_%"].values
    exp_dist = result_df["Expected_%"].values
    result_df.attrs["KL_Divergence"] = compare_distributions(obs_dist, exp_dist, "kl")
    result_df.attrs["Wasserstein_Distance"] = compare_distributions(obs_dist, exp_dist, "wasserstein")
    result_df.attrs["Total_Variation"] = 0.5 * np.sum(np.abs(obs_dist - exp_dist))

    result_df = result_df.reset_index().rename(columns={"index": "Group"})
    return result_df


def plot_disparity_chart(result_df: pd.DataFrame):
    plot_df = result_df.copy()
    colors = plot_df["Framework_Flag"].map({
        "✅ OK": "#2E8B57",
        "⚠️ Check": "#DC143C"
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        y="Group",
        x="Disparity_Ratio",
        palette=colors,
        ax=ax,
    )
    ax.axvline(1.0, color="black", linestyle="--", label="Perfect Parity")
    ax.axvline(0.8, color="orange", linestyle=":", label="Lower Bound (0.8)")
    ax.axvline(1.25, color="orange", linestyle=":", label="Upper Bound (1.25)")

    ax.set_title("📊 Disparity Ratios by Demographic Group", fontsize=14)
    ax.set_xlabel("Disparity Ratio (Observed / Expected)")
    ax.set_ylabel("Group")
    ax.legend()
    st.pyplot(fig)


# === Streamlit UI ===
st.title("🧮 Fairness Audit (3-Question Framework)")

with st.expander("📘 How it works", expanded=False):
    st.markdown("""
This tool implements a **3-question fairness framework**:

1. **Benchmark Validity**: Is your reference population appropriate?
2. **Magnitude of Disparity**: Is the disparity ratio extreme? (outside 0.8–1.25)
3. **Justifiability**: Can you explain any disparities (clinically, ethically, etc.)?

It also computes **KL Divergence**, **Wasserstein Distance**, and **Total Variation** between distributions.
""")

uploaded_file = st.file_uploader("📁 Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

    with st.expander("🔍 Preview Data"):
        st.dataframe(df.head(), use_container_width=True)

    # === Column Selection
    candidate_cols = df.columns[df.dtypes == "object"].tolist()
    demo_col = st.selectbox("👤 Select a demographic column to audit", candidate_cols)

    # === Audit
    if st.button("🚨 Run Fairness Audit"):
        result_df = compute_input_fairness(
            df=df,
            demographic_col=demo_col,
            benchmark_distribution=TNBC_RACE_BENCHMARK,
            justifiability_fn=default_justifiability_fn
        )

        st.markdown("## ✅ Audit Results")

        # Show summary metrics
        kl = result_df.attrs.get("KL_Divergence", np.nan)
        wd = result_df.attrs.get("Wasserstein_Distance", np.nan)
        tv = result_df.attrs.get("Total_Variation", np.nan)

        st.write(f"📐 **KL Divergence:** `{kl:.4f}`")
        st.write(f"🌊 **Wasserstein Distance:** `{wd:.4f}`")
        st.write(f"📊 **Total Variation:** `{tv:.4f}`")

        st.markdown("### 🧾 Disparity Table")
        st.dataframe(result_df.style.format({
            "Observed_%": "{:.1%}",
            "Expected_%": "{:.1%}",
            "Disparity_Ratio": "{:.2f}",
            "Absolute_Difference": "{:.2%}"
        }).background_gradient(subset=["Disparity_Ratio"], cmap="coolwarm", vmin=0.5, vmax=1.5),
        use_container_width=True)

        st.markdown("### 📉 Visualize Disparity")
        plot_disparity_chart(result_df)
