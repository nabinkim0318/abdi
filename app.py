# bias_audit_tool/app.py
import traceback
import uuid

import streamlit as st

from bias_audit_tool.data.data_loader import load_and_preview_data
from bias_audit_tool.data.upload_state import apply_upload_identity
from bias_audit_tool.data.upload_state import begin_new_upload
from bias_audit_tool.data.upload_state import clear_dataset_if_upload_removed
from bias_audit_tool.data.upload_state import reset_dataset_state
from bias_audit_tool.data.upload_state import uploader_widget_key
from bias_audit_tool.data.validation import blocking_issues
from bias_audit_tool.data.validation import CODE_NON_FINITE_VALUES
from bias_audit_tool.data.validation import fingerprint_upload
from bias_audit_tool.data.validation import SEVERITY_ERROR
from bias_audit_tool.data.validation import SEVERITY_WARNING
from bias_audit_tool.modeling.target_validation import preferred_target_column
from bias_audit_tool.preprocessing.recommend_columns import (
    direct_columns_for_sensitive_attribute,
)
from bias_audit_tool.preprocessing.recommend_columns import (
    recommend_demographic_columns,
)
from bias_audit_tool.preprocessing.recommend_columns import (
    SENSITIVE_ATTRIBUTE_CANDIDATE_CAPTION,
)
from bias_audit_tool.utils.ui_helpers import apply_preprocessing_and_display
from bias_audit_tool.utils.ui_helpers import display_preprocessing_recommendations
from bias_audit_tool.utils.ui_helpers import run_modeling_and_fairness
from bias_audit_tool.visualization.ui_blocks import audit_and_visualize_fairness
from bias_audit_tool.visualization.ui_blocks import download_processed_csv
from bias_audit_tool.visualization.visualization import show_visualizations


st.set_page_config(page_title="Bias Audit Tool", layout="wide")


# ===== Sidebar =====
st.sidebar.title("📊 Bias Audit Assistant")


# ===== Main Panel =====
st.title("🧪 Bias Audit Dashboard")
st.caption(
    "Exploratory bias and fairness diagnostics for tabular datasets "
    "using Fairlearn and scikit-learn."
)


def _render_validation_issues(issues):
    for issue in issues:
        if issue.severity == SEVERITY_ERROR:
            st.error(issue.message)
        elif issue.severity == SEVERITY_WARNING:
            st.warning(issue.message)


def main():

    enable_modeling = st.sidebar.radio("🤖 Run ML Model?", ["No", "Yes"])
    uploaded_file = st.file_uploader(
        "📤 Upload CSV",
        type=["csv"],
        key=uploader_widget_key(st.session_state),
    )

    # 📌 Initialize session state
    if "target_col" not in st.session_state:
        st.session_state.target_col = None
    if "preprocessing_applied" not in st.session_state:
        st.session_state.preprocessing_applied = False
    if "step3_ready" not in st.session_state:
        st.session_state.step3_ready = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "df_proc" not in st.session_state:
        st.session_state.df_proc = None
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None
    if "trigger_audit" not in st.session_state:
        st.session_state.trigger_audit = False
    if "audit_run_id" not in st.session_state:
        st.session_state.audit_run_id = uuid.uuid4()

    infinity_blocked = False
    if uploaded_file is None:
        clear_dataset_if_upload_removed(st.session_state, uploaded_file)
    else:
        fingerprint = fingerprint_upload(uploaded_file)
        dataset_changed = apply_upload_identity(
            st.session_state,
            filename=uploaded_file.name,
            fingerprint=fingerprint,
        )
        if st.session_state.df is None or dataset_changed:
            df, issues = load_and_preview_data(uploaded_file)
            header_blockers = [
                issue
                for issue in blocking_issues(issues)
                if issue.code != CODE_NON_FINITE_VALUES
            ]
            if df is None or header_blockers:
                _render_validation_issues(header_blockers or issues)
                st.stop()

            st.session_state.df = df
            st.session_state.structural_issues = issues
            st.session_state.recommendations = display_preprocessing_recommendations(
                df
            )
            st.success("✅ File successfully loaded!")

        issues = st.session_state.get("structural_issues") or []
        _render_validation_issues(issues)
        infinity_blocked = any(
            issue.code == CODE_NON_FINITE_VALUES for issue in issues
        )

    # 👉 Step 1: Recommendations (only once)
    if st.session_state.df is not None:
        df = st.session_state.df
        recommendations = st.session_state.recommendations
        issues = st.session_state.get("structural_issues") or []
        infinity_blocked = any(
            issue.code == CODE_NON_FINITE_VALUES for issue in issues
        )
        show_logs = st.checkbox("🪵 Show detailed preprocessing logs", value=False)

        # Infinities are allowed to load for explanation, but scalers/log
        # transforms cannot run on non-finite values.
        if infinity_blocked:
            st.info(
                "Preprocessing and modeling are blocked until infinity "
                "values are replaced in the source file."
            )
        elif st.button(
            "🚀 Apply Recommended Preprocessing", key="preprocessing_button"
        ):
            df_proc = apply_preprocessing_and_display(df, recommendations, show_logs)
            st.session_state.df_proc = df_proc
            st.session_state.preprocessing_applied = True
            st.session_state.step3_ready = True
            st.session_state.trigger_audit = True
            st.session_state.pop("demo_cols", None)
            st.success("✅ Preprocessing applied!")

        if (
            st.session_state.get("preprocessing_applied")
            and st.session_state.get("df_proc") is not None
        ):
            download_processed_csv(st.session_state.df_proc)

        # 👉 Step 3: Post-Preprocessing Analysis
        if st.session_state.get("step3_ready") and "df_proc" in st.session_state:
            df_proc = st.session_state.df_proc

            # 🧠 Step 3a: Demographic column recommendation (only once)
            if "demo_cols" not in st.session_state and df_proc is not None:
                df_proc, demo_cols_result = recommend_demographic_columns(df_proc)

                demo_cols = [
                    str(col)
                    for col in (demo_cols_result or [])
                    if isinstance(col, str) and col in df_proc.columns
                ]
                st.session_state.demo_cols = demo_cols
                st.session_state.df_proc = df_proc
            else:
                demo_cols = st.session_state.demo_cols

            if demo_cols:
                st.caption(SENSITIVE_ATTRIBUTE_CANDIDATE_CAPTION)
                previous_selection = st.session_state.get("group_col", demo_cols[0])
                default_index = (
                    demo_cols.index(previous_selection)
                    if previous_selection in demo_cols
                    else 0
                )
            else:
                st.warning(
                    "No candidate sensitive attributes matched the "
                    "column-name heuristic. This does not mean the dataset "
                    "has no sensitive attributes — review columns before use."
                )
                return

            # Step 3b: Visualizations + Audit
            st.header("📊 Data Preprocessing and Visualization")
            show_visualizations(df_proc, demo_cols)

            # Persist this checkbox state
            if "show_visualization" not in st.session_state:
                st.session_state.show_visualization = True

            show_vis = st.checkbox(
                "Show visualization", value=st.session_state.show_visualization
            )
            if show_vis != st.session_state.show_visualization:
                st.session_state.show_visualization = show_vis

            if st.session_state.show_visualization:
                if df_proc is None:
                    st.warning("No processed data.")
                    return

                demo_cols = st.session_state.get("demo_cols", [])
                if not demo_cols:
                    st.warning("No demographic columns.")
                    return

                # Allow user to experiment with multiple demographic columns
                previous_selection = st.session_state.get("group_col", demo_cols[0])
                group_col = st.selectbox(
                    "Select candidate sensitive attribute",
                    options=demo_cols,
                    index=(
                        demo_cols.index(previous_selection)
                        if previous_selection in demo_cols
                        else 0
                    ),
                    key="group_col_selectbox",
                )
                st.session_state.group_col = group_col

                if group_col not in df_proc.columns:
                    st.error(
                        f"❌ Column '{group_col}' not found in "
                        "DataFrame after preprocessing."
                    )
                    st.session_state.demo_cols = [
                        col for col in demo_cols if col != group_col
                    ]
                    return

                current_group_col = group_col
                last_group_col = st.session_state.get("last_group_col", None)

                if current_group_col != last_group_col or st.session_state.get(
                    "trigger_audit", False
                ):
                    if "audit_run_id" not in st.session_state:
                        st.session_state.audit_run_id = uuid.uuid4()

                    error_occurred = False
                    try:
                        audit_and_visualize_fairness(df_proc, group_col)
                    except Exception as e:
                        st.session_state.audit_error_msg = f"{e}"
                        st.session_state.audit_error_trace = traceback.format_exc()
                        error_occurred = True

                    with st.expander("🔎 Fairness Audit", expanded=True):
                        if error_occurred:
                            st.error(
                                f"❌ Error occurred during Fairness audit: "
                                f"{st.session_state.audit_error_msg}"
                            )
                            st.text(st.session_state.audit_error_trace)
                        else:
                            st.session_state["last_group_col"] = current_group_col
                            st.session_state["trigger_audit"] = False
                            st.write(
                                "🔁 Audit Run ID:", st.session_state.audit_run_id
                            )

            # Step 3c: Modeling
            if enable_modeling == "Yes" and "group_col" in st.session_state:
                if infinity_blocked:
                    st.error(
                        "Modeling is blocked because numeric infinity values "
                        "are present. Replace them before running the model."
                    )
                else:
                    # Modeling uses the raw uploaded columns (not df_proc) so
                    # that preprocessing can be fit on the train split only.
                    raw_cols = df.columns.tolist()
                    default_col = preferred_target_column(
                        df,
                        columns=raw_cols,
                        current_selection=st.session_state.get("target_col"),
                        deprioritized=direct_columns_for_sensitive_attribute(
                            st.session_state.group_col, raw_cols
                        ),
                    )
                    default_index = raw_cols.index(default_col)
                    target_col = st.selectbox(
                        "🎯 Select target column",
                        options=raw_cols,
                        index=default_index,
                    )
                    st.session_state.target_col = target_col

                    include_sensitive_as_feature = st.checkbox(
                        "Include selected sensitive attribute as a model feature?",
                        value=False,
                        help=(
                            "By default, the selected sensitive attribute and "
                            "its direct encodings (for example one-hot "
                            "race_* columns) are excluded from the model's "
                            "predictive features. Unrelated correlated "
                            "variables (proxies) are not removed "
                            "automatically. Including the attribute is "
                            "not a fairness guarantee."
                        ),
                    )

                    if target_col:
                        run_modeling_and_fairness(
                            raw_df=df,
                            df_proc=df_proc,
                            target_col=target_col,
                            group_col=st.session_state.group_col,
                            include_sensitive_as_feature=include_sensitive_as_feature,
                            recommendations=recommendations,
                        )

            st.markdown("---")

            if st.button("🔁 Try with another dataset?"):
                begin_new_upload(st.session_state)
                st.rerun()

    else:
        st.info("⬅️ Please upload a dataset to begin.")


def initialize_session():
    """Ensure dataset-derived keys exist without wiping a valid rerun."""
    if "df" not in st.session_state:
        reset_dataset_state(st.session_state, clear_identity=False)


if __name__ == "__main__":
    main()
