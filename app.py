# bias_audit_tool/app.py
import traceback
import uuid

import streamlit as st

from bias_audit_tool.data.data_loader import load_and_preview_data
from bias_audit_tool.preprocessing.recommend_columns import (
    recommend_demographic_columns,
)
from bias_audit_tool.utils.ui_helpers import apply_preprocessing_and_display
from bias_audit_tool.utils.ui_helpers import display_preprocessing_recommendations
from bias_audit_tool.utils.ui_helpers import get_user_preprocessing_options
from bias_audit_tool.utils.ui_helpers import run_modeling_and_fairness
from bias_audit_tool.visualization.ui_blocks import audit_and_visualize_fairness
from bias_audit_tool.visualization.ui_blocks import download_processed_csv
from bias_audit_tool.visualization.visualization import show_visualizations


st.set_page_config(page_title="Bias Audit Tool", layout="wide")


# ===== Sidebar =====
st.sidebar.title("📊 Bias Audit Assistant")


# ===== Main Panel =====
st.title("🧪 Bias Audit Dashboard")


def main():

    enable_modeling = st.sidebar.radio("🤖 Run ML Model?", ["No", "Yes"])
    uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])

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

    if uploaded_file is not None:
        if st.session_state.df is None or uploaded_file.name != getattr(
            st.session_state, "uploaded_file_name", None
        ):
            df = load_and_preview_data(uploaded_file)
            if df is None:
                st.stop()

            st.session_state.df = df
            st.session_state.df_proc = None
            st.session_state.recommendations = display_preprocessing_recommendations(
                df
            )
            st.session_state.preprocessing_applied = False
            st.session_state.step3_ready = False
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success("✅ File successfully loaded!")

    # 👉 Step 1: Recommendations (only once)
    if st.session_state.df is not None:
        df = st.session_state.df
        recommendations = st.session_state.recommendations
        show_logs = st.checkbox("🪵 Show detailed preprocessing logs", value=False)
        options = get_user_preprocessing_options()

        # 👉 Step 2: Apply Preprocessing
        if button_clicked("preprocessing_button"):
            df_proc = apply_preprocessing_and_display(
                df, recommendations, show_logs, options
            )
            st.session_state.df_proc = df_proc
            st.session_state.preprocessing_applied = True
            st.session_state.step3_ready = True
            st.session_state.trigger_audit = True

            st.success("✅ Preprocessing applied!")
            # Download CSV
            download_processed_csv(df_proc)

        # 👉 Step 3: Post-Preprocessing Analysis
        if st.session_state.get("step3_ready") and "df_proc" in st.session_state:
            print("[DEBUG] step3_ready is True")
            df_proc = st.session_state.df_proc
            print(
                "[DEBUG] if df_proc has gender before "
                "recommend_demographic_columns:",
                "gender_mapped" in df_proc.columns if df_proc is not None else False,
            )

            # 🧠 Step 3a: Demographic column recommendation (only once)
            if "demo_cols" not in st.session_state and df_proc is not None:
                df_proc, demo_cols_result = recommend_demographic_columns(df_proc)
                print(
                    "[DEBUG] Columns after recommend_demographic_columns:",
                    df_proc.columns.tolist(),
                )
                if "demographic.race_mapped" not in df_proc.columns:
                    print("[⚠️] demographic.race_mapped was dropped!")

                demo_cols = [
                    str(col)
                    for col in (demo_cols_result or [])
                    if isinstance(col, str) and col in df_proc.columns
                ]
                st.session_state.demo_cols = demo_cols
                st.session_state.df_proc = df_proc
                if "demographic.race_mapped" not in df_proc.columns:
                    print(
                        f"[⚠️] demographic.race_mapped was dropped! "
                        f"after session_status.demo_cols: "
                        f"{st.session_state.demo_cols}"
                    )
            else:
                demo_cols = st.session_state.demo_cols

            if demo_cols:
                previous_selection = st.session_state.get("group_col", demo_cols[0])
                default_index = (
                    demo_cols.index(previous_selection)
                    if previous_selection in demo_cols
                    else 0
                )
            else:
                st.warning("No suitable demographic columns found.")
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
                    "Select demographic column",
                    options=demo_cols,
                    index=(
                        demo_cols.index(previous_selection)
                        if previous_selection in demo_cols
                        else 0
                    ),
                    key="group_col_selectbox",
                )
                st.session_state.group_col = group_col
                print(f"[INFO] group_col selected: {group_col}")
                print(
                    "[DEBUG] if df_proc has gender after group_col selected:",
                    "gender_mapped" in df_proc.columns,
                )

                if group_col not in df_proc.columns:
                    st.error(
                        f"❌ Column '{group_col}' not found in "
                        "DataFrame after preprocessing."
                    )
                    st.session_state.demo_cols = [
                        col for col in demo_cols if col != group_col
                    ]
                    return

                if df_proc is not None:
                    print("[DEBUG] df_proc columns:", df_proc.columns)
                    print(
                        "[DEBUG] if df_proc has gender:",
                        "gender_mapped" in df_proc.columns,
                    )
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
                cols = df_proc.columns.tolist() if df_proc is not None else []
                default_index = (
                    cols.index(st.session_state.group_col)
                    if st.session_state.group_col in cols
                    else 0
                )
                target_col = st.selectbox(
                    "🎯 Select target column", options=cols, index=default_index
                )
                st.session_state.target_col = target_col

                if target_col:
                    run_modeling_and_fairness(df_proc, target_col, demo_cols)

            st.markdown("---")

            if st.button("🔁 Try with another dataset?"):
                # Clear only relevant session keys
                initialize_session()

    else:
        st.info("⬅️ Please upload a dataset to begin.")


def initialize_session():
    defaults = {
        "target_col": None,
        "preprocessing_applied": False,
        "step3_ready": False,
        "df": None,
        "df_proc": None,
        "recommendations": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def button_clicked(key):
    if st.button("🚀 Apply Recommended Preprocessing", key=key):
        st.session_state[key + "_clicked"] = True
    return st.session_state.get(key + "_clicked", False)


def extract_valid_demo_cols(candidate_info, df_columns):
    if (
        isinstance(candidate_info, list)
        and candidate_info
        and isinstance(candidate_info[0], tuple)
        and len(candidate_info[0]) == 3
    ):
        demo_cols = [col for col, _, _ in candidate_info]
    else:
        demo_cols = candidate_info  # fallback if already flat list
    return [col for col in demo_cols if col in df_columns]


if __name__ == "__main__":
    main()
