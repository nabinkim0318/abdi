# utils/fairness.py
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import entropy
from scipy.stats import wasserstein_distance

# Updated TNBC benchmark based on your requirements
TNBC_RACE_BENCHMARK = {
    'Black': 0.36,     # 36% - highest incidence
    'White': 0.19,     # 19% 
    'Hispanic': 0.16,  # 16%
    'AIAN': 0.16,      # 16% (American Indian/Alaska Native)
    'Asian': 0.13,     # 13% (updated from your table)
}


def validate_inputs(df, demographic_col, benchmark_distribution):
    """Validate inputs for fairness analysis."""
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be None or empty")
    
    if demographic_col not in df.columns:
        raise ValueError(f"Column '{demographic_col}' not found in DataFrame")
    
    if not isinstance(benchmark_distribution, dict):
        raise ValueError("benchmark_distribution must be a dictionary")
    
    # Check if benchmark sums to ~1.0
    total_benchmark = sum(benchmark_distribution.values())
    if not 0.95 <= total_benchmark <= 1.05:
        warnings.warn(f"Benchmark distribution sums to {total_benchmark:.3f}, not 1.0")


def compare_distributions(p, q, method="kl"):
    """
    Compare two distributions using KL divergence or Wasserstein distance.
    
    Returns:
    - divergence score (lower = more similar)
    """
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    
    # Handle zeros for KL divergence
    if method == "kl":
        epsilon = 1e-8
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)

    # Normalize to valid probability distributions
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
    benchmark_distribution: dict = None,
    threshold_low: float = 0.8,
    threshold_high: float = 1.25,
    sort_by: str = "Observed_%",
) -> pd.DataFrame:
    """
    Detects unfair representation by comparing observed vs expected proportions.
    
    Returns:
    - DataFrame with fairness assessment and distribution metrics
    """
    # Use TNBC benchmark as default
    if benchmark_distribution is None:
        benchmark_distribution = TNBC_RACE_BENCHMARK
        print("Using default TNBC race benchmark distribution")
    
    validate_inputs(df, demographic_col, benchmark_distribution)
    
    # Count and compute proportions
    observed_counts = df[demographic_col].value_counts(dropna=False)
    total = len(df)
    observed_percent = observed_counts / total

    result_df = pd.DataFrame({
        "Observed_Count": observed_counts,
        "Observed_%": observed_percent
    })

    # Map benchmark (handle missing groups)
    result_df["Expected_%"] = result_df.index.map(benchmark_distribution)
    missing_groups = result_df["Expected_%"].isnull().sum()
    if missing_groups > 0:
        print(f"Warning: {missing_groups} groups missing from benchmark. Using 0.01% default.")
        result_df["Expected_%"] = result_df["Expected_%"].fillna(0.0001)

    # Compute disparity metrics
    result_df["Disparity_Ratio"] = result_df["Observed_%"] / result_df["Expected_%"]
    result_df["Absolute_Difference"] = abs(result_df["Observed_%"] - result_df["Expected_%"])
    
    # Fairness assessment with detailed reasons
    result_df["Fair?"] = result_df["Disparity_Ratio"].apply(
        lambda x: "Fair" if threshold_low <= x <= threshold_high else "Not Fair"
    )
    
    result_df["Deviation_Type"] = result_df["Disparity_Ratio"].apply(
        lambda x: "Under-represented" if x < threshold_low 
                 else "Over-represented" if x > threshold_high
                 else "Within bounds"
    )

    # Store distribution comparison metrics
    obs_dist = result_df["Observed_%"].values
    exp_dist = result_df["Expected_%"].values
    
    result_df.attrs["KL_Divergence"] = compare_distributions(obs_dist, exp_dist, "kl")
    result_df.attrs["Wasserstein_Distance"] = compare_distributions(obs_dist, exp_dist, "wasserstein")
    result_df.attrs["Total_Variation"] = 0.5 * np.sum(np.abs(obs_dist - exp_dist))

    # Sort and clean up
    result_df = result_df.sort_values(sort_by, ascending=False).reset_index()
    result_df = result_df.rename(columns={"index": "Group"})
    
    return result_df


def display_fairness_summary(result_df: pd.DataFrame):
    """Simplified summary showing only key metrics."""
    if result_df is None or result_df.empty:
        st.error("No fairness data to display")
        return
        
    total_groups = len(result_df)
    fair_groups = (result_df["Fair?"] == "Fair").sum()
    unfair_groups = total_groups - fair_groups
    
    # Get distribution metrics
    kl_div = result_df.attrs.get("KL_Divergence", "N/A")
    wass_dist = result_df.attrs.get("Wasserstein_Distance", "N/A") 
    total_var = result_df.attrs.get("Total_Variation", "N/A")

    # Main summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Groups", total_groups)
    with col2:
        st.metric("Fair Groups", fair_groups, delta=f"{fair_groups/total_groups:.1%}")
    with col3:
        st.metric("Unfair Groups", unfair_groups, delta=f"-{unfair_groups/total_groups:.1%}")

    # Distribution distance metrics
    st.markdown("### 📐 Distribution Distance Metrics")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("KL Divergence", f"{kl_div:.4f}" if isinstance(kl_div, float) else kl_div)
    with metrics_col2:
        st.metric("Wasserstein Distance", f"{wass_dist:.4f}" if isinstance(wass_dist, float) else wass_dist)
    with metrics_col3:
        st.metric("Total Variation", f"{total_var:.4f}" if isinstance(total_var, float) else total_var)
