import streamlit as st

from bias_audit_tool.modeling.fairness import compute_input_fairness
from bias_audit_tool.modeling.fairness import display_fairness_summary
from bias_audit_tool.modeling.fairness import parse_user_benchmark
from bias_audit_tool.report.report_generator import generate_pdf_report
from bias_audit_tool.visualization.visualization import plot_distribution_comparison
from bias_audit_tool.visualization.visualization import show_visualizations


def clean_label(col):
    return col.split(".")[-1].replace("_", " ").title()


def show_demographic_analysis(df_proc):
    """
    Comprehensive demographic analysis with summary
    table, heatmap, and friendly explanations.

    Displays:
        - Sidebar list of auto-detected demographic columns
        - Multiselect to choose columns for group-wise audit
        - Optional target selection (excluding ID columns)
        - Group-wise visualizations by demographic feature
        - Summary table of key demographic metrics
        - Heatmap showing structural bias patterns
        - Top-k most important demographic columns
        - Friendly explanations of why each metric matters

    Args:
        df_proc (pd.DataFrame): Preprocessed DataFrame to analyze.

    Returns:
        tuple: (selected_demo_cols, target_col)
    """
    # st.markdown("### 🧬 Demographic Bias Analysis")

    # if not demographic_candidates:
    #     st.warning("No demographic columns detected. Please check your data.")
    #     return [], None

    # # 2. Summary Table - Key demographic metrics at a glance
    # st.markdown("#### 📊 Demographic Summary Table")
    # summary_data = []
    # for col in demographic_candidates[:10]:  # Top 10 only
    #     if col in df_proc.columns:
    #         unique_count = df_proc[col].nunique()
    #         missing_pct = (df_proc[col].isnull().sum() / len(df_proc)) * 100
    #         summary_data.append(
    #             {
    #                 "Column": clean_label(col),
    #                 "Unique Values": unique_count,
    #                 "Missing %": f"{missing_pct:.1f}%",
    #                 "Type": str(df_proc[col].dtype),
    #             }
    #         )

    # summary_df = pd.DataFrame(summary_data)
    # st.dataframe(summary_df, use_container_width=True)

    # 3. Heatmap - Structural bias visualization
    # st.markdown("#### 🔥 Demographic Distribution Heatmap")
    # if len(demographic_candidates) > 1:
    #     # Create correlation/co-occurrence matrix for demographic columns
    #     demo_subset = df_proc[demographic_candidates[:5]]  # Top 5 for heatmap
    #     demo_encoded = pd.get_dummies(demo_subset, drop_first=True)

    #     if not demo_encoded.empty:
    #         fig, ax = plt.subplots(figsize=(10, 6))
    #         sns.heatmap(
    #             demo_encoded.corr(),
    #             annot=True,
    #             cmap="coolwarm",
    #             center=0,
    #             ax=ax,
    #         )
    #         ax.set_title("Demographic Feature Correlations")
    #         st.pyplot(fig)
    #         plt.close(fig)

    # # 4. Top-k Auto Show - Most important columns first
    # st.markdown("#### 🎯 Key Demographic Insights")
    # top_k = min(5, len(demographic_candidates))  # Show top 5
    # selected_demo_cols = demographic_candidates[:top_k]

    # with st.expander("🔍 Why are these columns important?"):
    #     st.markdown(
    #         """
    #     **Demographic columns help identify potential bias in your data:**
    #     - **Age, Gender, Race**: Common sources of algorithmic bias
    #     - **Location, Education**: Socioeconomic factors that may
    #     affect outcomes
    #     - **Income, Occupation**: Economic disparities in
    #     data representation

    #     **Why this matters:** Uneven representation across these groups
    #     can lead to biased models that perform poorly for
    #     underrepresented populations.
    #     """
    #     )

    # # 5. Simple distribution plots for top columns
    # for col in selected_demo_cols[:3]:  # Show top 3 distributions
    #     if col in df_proc.columns:
    #         st.markdown(f"**Distribution of {clean_label(col)}**")
    #         fig, ax = plt.subplots(figsize=(8, 4))
    #         if df_proc[col].dtype == "object" or df_proc[col].nunique() < 10:
    #             sns.countplot(data=df_proc, x=col, ax=ax, hue=df_proc[col])
    #             ax.set_title(f"Distribution of {clean_label(col)}")
    #             ax.set_xlabel(clean_label(col))
    #             ax.tick_params(axis="x", rotation=45)
    #         else:
    #             sns.histplot(data=df_proc, x=col, ax=ax, bins=20)
    #             ax.set_title(f"Distribution of {clean_label(col)}")
    #             ax.set_xlabel(clean_label(col))
    #         st.pyplot(fig)
    #         plt.close(fig)

    # # 6. Optional target selection (simplified)
    # target_col = None
    # if st.checkbox("🎯 Add target variable analysis (optional)"):
    #     # Completely exclude ID columns from target candidates
    #     target_candidates = [
    #         col
    #         for col in df_proc.columns
    #         if not is_id_column(df_proc[col]) and col not in demographic_candidates
    #     ]
    #     if target_candidates:
    #         target_col = st.selectbox(
    #             "Select target variable for bias analysis",
    #             options=["None"] + target_candidates[:10],  # Top 10 targets
    #             index=0,
    #         )
    #         if target_col == "None":
    #             target_col = None
    #     else:
    #         st.warning("No suitable target columns found " "(ID columns excluded)")

    # return selected_demo_cols, target_col


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


def audit_and_visualize(df_proc, recommendations):
    """
    Display audit options, visualizations, and allow PDF report export.

    UI Elements:
        - Sidebar column selector for audit
        - Optional manual override for preprocessing strategy per column
        - Visualizations for selected audit columns
        - Button to generate and download PDF report

    Args:
        df_proc (pd.DataFrame): Preprocessed DataFrame to audit.
        recommendations (dict): Dictionary of preprocessing recommendations
                                for columns.

    Displays:
        - Grouped histograms or KDE plots
        - Downloadable bias audit PDF report
    """
    audit_cols = st.sidebar.multiselect(
        "### 3️⃣ Select Columns for Audit", df_proc.columns
    )
    if audit_cols:
        st.markdown("#### 🗂️ Preprocessing Legend")
        with st.expander("Show explanation for each preprocessing option"):
            st.markdown(
                """
- **LabelEncoder**: Converts each category to a unique integer (0, 1, 2, ...).
- **OneHotEncoder**: Creates a new binary column for each category (0 or 1).
- **MinMaxScaler**: Scales numeric values to the range 0 to 1.
- **Log1pTransform**: Applies log(1 + x) transformation to numeric data.
- **ImputeMissing**: Fills missing values with
    the mean (numeric) or mode (categorical).
- **DropHighNaNs**: Drops columns with a high proportion of missing values.
            """
            )
        st.markdown("### 📌 Preprocessing Options (Manual Override)")
        for col in audit_cols:
            st.selectbox(
                f"Preprocessing Strategy for `{col}`",
                options=[
                    "None",
                    "LabelEncoder",
                    "OneHotEncoder",
                    "MinMaxScaler",
                    "Log1pTransform",
                ],
                key=f"prep_{col}",
            )

        st.markdown("### 📊 Visualizations")
        show_visualizations(df_proc, audit_cols)

        st.markdown("### 📥 Export Report")
        st.caption(
            "Download a detailed PDF report of " "the selected audit columns."
        )
        if st.button("📤 Export PDF Report"):
            pdf_buffer = generate_pdf_report(
                df_proc,
                audit_cols,
                recommendations,
            )
            st.download_button(
                "📥 Download PDF Report",
                pdf_buffer,
                "bias_audit_report.pdf",
                mime="application/pdf",
            )
    else:
        st.info("⬅️ Select columns from the sidebar to " "audit and visualize.")


def audit_and_visualize_fairness(df, group_col):
    try:
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
        print(f"[DEBUG] fairness_result: {fairness_result}")

        display_fairness_summary(fairness_result)

        with st.expander("📈 Observed vs Expected Distribution"):
            if fairness_result is not None and not fairness_result.empty:
                top_n = st.slider(
                    "Top N Groups to Show", 5, 50, 20, key="top_n_slider"
                )
                fig = plot_distribution_comparison(fairness_result, top_n=top_n)
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Could not generate distribution plot.")
            else:
                st.info("ℹ️ Fairness result is empty or unavailable.")

        print(f"[DEBUG] fairness_result: {fairness_result}")
        # with st.expander("📊 Disparity Ratio Histogram"):
        #     if fairness_result is not None and not fairness_result.empty:
        #         fig = plot_input_fairness(fairness_result)
        #         if fig:
        #             st.pyplot(fig)
        #         else:
        #             st.warning("⚠️ Fairness plot could not be generated.")
        #     else:
        #         st.info("ℹ️ Fairness result is empty or unavailable.")
    except Exception as e:
        print(f"[ERROR] audit_and_visualize_fairness failed: {e}")
        return None
