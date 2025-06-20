import traceback
from collections import defaultdict

import streamlit as st
from sklearn.metrics import roc_auc_score
from utils.fairness import compute_fairness_metrics
from utils.model_selector import run_basic_modeling
from utils.preprocess import recommend_preprocessing
from utils.summary import summarize_categories
from utils.transform import apply_preprocessing


def display_preprocessing_recommendations(df):
    st.markdown("### 🧠 Recommended Preprocessing")
    recommendations = recommend_preprocessing(df)

    with st.expander("📋 Show Detailed Column Recommendations"):
        grouped_recs = defaultdict(list)
        for col, rec in recommendations.items():
            category = col.split(".")[0] if "." in col else "project"
            grouped_recs[category].append((col, rec))

        for category, items in grouped_recs.items():
            with st.expander(f"📁 {category}"):
                for col, rec in items:
                    st.markdown(f"🔧 **{col}** → _{rec}_")

    summary_df = summarize_categories(df, recommendations)
    st.markdown("### 📊 Preprocessing Recommendation Summary")
    st.dataframe(summary_df, use_container_width=True)

    return recommendations


def get_user_preprocessing_options():
    st.markdown("### ⚙️ Preprocessing Options")

    enable_scaling = st.checkbox(
        "🔧 Apply Scaling to numeric columns (MinMaxScaler)",
        value=True,
        help=(
            "Rescales numeric features between 0 and 1. "
            "Recommended for ML modeling."
        ),
    )

    enable_encoding = st.checkbox(
        "🔧 Encode categorical columns",
        value=True,
        help=(
            "Converts text columns into numeric format "
            "(e.g., OneHot or Label encoding)."
        ),
    )

    handle_missing = st.checkbox(
        "🧩 Handle missing values automatically",
        value=True,
        help=(
            "Impute missing numeric values with mean, categorical with mode. "
            "Drop columns with >95% missing."
        ),
    )

    return enable_scaling, enable_encoding, handle_missing


def show_selected_options(enable_scaling, enable_encoding, handle_missing):
    st.caption(
        f"🔧 Applied options: Scaling = {enable_scaling}, "
        f"Encoding = {enable_encoding}, "
        f"Missing Handling = {handle_missing}"
    )


def execute_preprocessing(df, recommendations, show_logs=False):
    df_proc = apply_preprocessing(df, recommendations, show_logs)
    st.write(f"🔄 Data shape changed from `{df.shape}` → `{df_proc.shape}`")
    st.success("✅ Preprocessing Applied!")
    st.dataframe(df_proc.head())
    return df_proc


def apply_preprocessing_and_display(df, recommendations, show_logs, options):
    enable_scaling, enable_encoding, handle_missing = options
    show_selected_options(enable_scaling, enable_encoding, handle_missing)
    df_proc = execute_preprocessing(df, recommendations, show_logs)
    return df_proc


def run_modeling_and_fairness(df_proc, target_col, selected_demo_cols):
    """
    Run machine learning modeling and fairness audit with Streamlit UI.

    Args:
        df_proc (pd.DataFrame): Preprocessed DataFrame
        target_col (str): Name of the target column
        selected_demo_cols (list[str]): List of sensitive attributes
    """
    st.markdown("## 🧠 Machine Learning Modeling")

    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]

    try:
        results = run_basic_modeling(X, y)

        st.markdown("### 🔍 Classification Report")
        st.dataframe(results["report"])

        if results["y_prob"] is not None:
            auc = roc_auc_score(results["y_test"], results["y_prob"])
            st.markdown(f"📈 ROC AUC: `{auc:.2f}`")

        if selected_demo_cols:
            st.markdown("### ⚖️ Fairness Audit with `fairlearn`")
            for attr in selected_demo_cols:
                st.markdown(f"#### Sensitive Attribute: `{attr}`")

                try:
                    metric_frame, fairness_summary = compute_fairness_metrics(
                        y_true=results["y_test"],
                        y_pred=results["y_pred"],
                        sensitive_features=X[attr].loc[results["y_test"].index],
                    )

                    st.markdown("📊 Group-wise Metrics")
                    st.dataframe(metric_frame.by_group)

                    st.markdown("🧾 Summary of Fairness Disparities")
                    for key, value in fairness_summary.items():
                        st.markdown(f"- **{key}**: `{value:.4f}`")

                except Exception as e:
                    st.warning(f"Could not compute fairness for `{attr}`: {e}")

    except Exception:
        st.error("❌ Modeling failed.")
        st.text(traceback.format_exc())
