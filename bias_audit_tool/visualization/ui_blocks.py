import streamlit as st
from preprocessing.recommend_columns import identify_demographic_columns
from report.report_generator import generate_pdf_report
from visualization.visualization import show_groupwise_visualizations
from visualization.visualization import show_visualizations


def show_demographic_analysis(df_proc):
    """
    Identify and analyze demographic columns in the processed DataFrame.

    Displays:
        - Sidebar list of auto-detected demographic columns
        - Multiselect to choose columns for group-wise audit
        - Optional target selection
        - Group-wise visualizations by demographic feature

    Args:
        df_proc (pd.DataFrame): Preprocessed DataFrame to analyze.

    Returns:
        tuple: (selected_demo_cols, target_col)
            - selected_demo_cols (list[str]): Columns selected for
                                              demographic grouping
            - target_col (str or None): Optional column for prediction-based
                                        group analysis
    """
    st.markdown("### 🧬 Demographic Column Audit")

    demographic_candidates = identify_demographic_columns(df_proc)
    st.sidebar.markdown("#### 🧬 Auto-Detected Demographics")
    st.sidebar.write(", ".join(demographic_candidates) or "❌ None found")

    selected_demo_cols = st.multiselect(
        "👥 Select Demographic Columns for Group-wise Audit",
        demographic_candidates,
        default=demographic_candidates,
    )

    target_col = None
    if selected_demo_cols:
        st.markdown("### 👥 Demographic Group-wise Analysis")
        target_col = st.selectbox(
            "🎯 Select Target Column (Optional)",
            df_proc.columns,
        )
        show_groupwise_visualizations(
            df_proc,
            selected_demo_cols,
            target_col,
        )

    return selected_demo_cols, target_col


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
            "Download a detailed PDF report of the selected " "audit columns."
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
