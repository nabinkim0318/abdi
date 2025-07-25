# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import entropy, wasserstein_distance

# === TNBC Race Benchmark ===
TNBC_RACE_BENCHMARK = {
    'Black': 0.36,
    'White': 0.19,
    'Hispanic': 0.16,
    'AIAN': 0.16,
    'Asian': 0.13
}

# === Distribution Comparison Utilities ===
def compare_distributions(p, q, method="kl"):
    """Compare two distributions using KL divergence or Wasserstein distance."""
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
        raise ValueError("Method must be 'kl' or 'wasserstein'.")

# === Fairness Summary Core Function ===
def compute_fairness_summary(df, demographic_col, benchmark_dist, threshold_low=0.8, threshold_high=1.25):
    if demographic_col not in df.columns:
        raise ValueError(f"Column '{demographic_col}' not found in dataset.")

    observed_counts = df[demographic_col].value_counts(dropna=False)
    total = len(df)
    observed_percent = observed_counts / total

    result_df = pd.DataFrame({
        'Observed_Count': observed_counts,
        'Observed_%': observed_percent
    })

    result_df['Expected_%'] = result_df.index.map(benchmark_dist)
    result_df["Expected_%"].fillna(0.0001, inplace=True)

    result_df['Disparity_Ratio'] = result_df['Observed_%'] / result_df['Expected_%']
    result_df['Fair?'] = result_df['Disparity_Ratio'].apply(
        lambda x: 'Fair' if threshold_low <= x <= threshold_high else 'Not Fair'
    )

    # Attach distance metrics to the dataframe as attributes
    obs_dist = result_df["Observed_%"].values
    exp_dist = result_df["Expected_%"].values
    result_df.attrs["KL_Divergence"] = compare_distributions(obs_dist, exp_dist, "kl")
    result_df.attrs["Wasserstein_Distance"] = compare_distributions(obs_dist, exp_dist, "wasserstein")
    result_df.attrs["Total_Variation"] = 0.5 * np.sum(np.abs(obs_dist - exp_dist))

    return result_df

# === Streamlit UI ===
st.set_page_config(page_title="Fairness Summary", layout="centered")

st.title("🔍 TNBC Fairness Summary (Demographics Only)")

uploaded_file = st.file_uploader("Upload Clinical CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    demographic_col = st.selectbox("Select Demographic Column", df.columns, index=0)

    try:
        fairness_result = compute_fairness_summary(df, demographic_col, TNBC_RACE_BENCHMARK)

        # Summarize results
        total_groups = len(fairness_result)
        fair_groups = (fairness_result["Fair?"] == "Fair").sum()
        unfair_groups = total_groups - fair_groups

        kl_div = fairness_result.attrs["KL_Divergence"]
        wass_dist = fairness_result.attrs["Wasserstein_Distance"]
        total_var = fairness_result.attrs["Total_Variation"]

        # === Output Metrics ===
        st.markdown("### ✅ Fairness Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Groups", total_groups)
        col2.metric("Fair Groups", fair_groups, delta=f"{fair_groups / total_groups:.1%}")
        col3.metric("Unfair Groups", unfair_groups, delta=f"-{unfair_groups / total_groups:.1%}")

        st.markdown("### 📐 Distribution Distance Metrics")
        col4, col5, col6 = st.columns(3)
        col4.metric("KL Divergence", f"{kl_div:.4f}")
        col5.metric("Wasserstein Distance", f"{wass_dist:.4f}")
        col6.metric("Total Variation", f"{total_var:.4f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
else:
    st.info("📂 Please upload a clinical dataset CSV file to begin.")
