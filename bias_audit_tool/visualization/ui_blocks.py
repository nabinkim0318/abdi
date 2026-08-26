import streamlit as st

from bias_audit_tool.modeling.fairness import compute_input_fairness
from bias_audit_tool.modeling.fairness import display_fairness_summary
from bias_audit_tool.modeling.fairness import parse_user_benchmark
from bias_audit_tool.visualization.visualization import plot_distribution_comparison


def download_processed_csv(df_proc):
    """
    Allow user to download the processed DataFrame as a CSV file.

    Args:
        df_proc (pd.DataFrame): The preprocessed DataFrame to export.

    Displays:
        - Streamlit download button for exporting CSV.
    """
    csv_buffer = df_proc.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Processed Data",
        csv_buffer,
        "processed_data.csv",
        "text/csv",
    )


def audit_and_visualize_fairness(df, group_col):
    st.subheader("📊 Representation disparity diagnostics")
    st.markdown(
        "📌 Provide an **explicit expected distribution** to compute "
        "benchmark-relative representation disparities. JSON format: "
        "`{'GroupA': 0.5, 'GroupB': 0.5}`. Leave empty if no benchmark "
        "has been selected."
    )

    if "benchmark_json" not in st.session_state:
        st.session_state["benchmark_json"] = ""

    benchmark_json = st.text_area(
        "Expected distribution (JSON). Leave empty for no benchmark.",
        key="benchmark_json",
    )

    benchmark, benchmark_status = parse_user_benchmark(benchmark_json)
    if benchmark_status == "invalid":
        st.warning(
            "Invalid JSON. Benchmark-relative representation analysis "
            "was not computed."
        )

    fairness_result = compute_input_fairness(
        df, demographic_col=group_col, benchmark_distribution=benchmark
    )
    st.session_state["fairness_result"] = fairness_result
    st.session_state["step3_done"] = True

    display_fairness_summary(fairness_result)

    with st.expander("📈 Observed vs Expected Distribution"):
        if fairness_result is not None and not fairness_result.empty:
            top_n = st.slider("Top N Groups to Show", 5, 50, 20, key="top_n_slider")
            fig = plot_distribution_comparison(fairness_result, top_n=top_n)
            if fig:
                st.pyplot(fig)
            else:
                st.warning("⚠️ Could not generate distribution plot.")
        else:
            st.info("ℹ️ Fairness result is empty or unavailable.")
