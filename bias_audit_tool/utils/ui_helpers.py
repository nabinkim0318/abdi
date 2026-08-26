import traceback

import streamlit as st
from sklearn.metrics import roc_auc_score

from bias_audit_tool.modeling.fairness import compute_output_fairness
from bias_audit_tool.modeling.target_validation import UnsupportedTargetError
from bias_audit_tool.preprocessing.modeling_pipeline import run_modeling_pipeline
from bias_audit_tool.preprocessing.preprocess import recommend_preprocessing
from bias_audit_tool.preprocessing.summary import summarize_categories
from bias_audit_tool.preprocessing.transform import apply_preprocessing


def display_preprocessing_recommendations(df):
    """
    Display preprocessing recommendations in the Streamlit interface.

    Args:
        df (pd.DataFrame): The original input DataFrame.

    Returns:
        dict: Dictionary mapping column names to recommended preprocessing steps.
    """
    st.markdown("#### 🗂️ Preprocessing Legend")
    with st.expander("Show explanation for each preprocessing option"):
        st.markdown(
            """
- **LabelEncoder**: Converts each category to a unique integer (0, 1, 2, ...).
- **OneHotEncoder**: Creates a new binary column for each category (0 or 1).
- **MinMaxScaler**: Scales numeric values to the range 0 to 1.
- **Log1pTransform**: Applies log(1 + x) transformation to numeric data.
- **ImputeMissing**: Fills missing values with the mean
    (numeric) or mode (categorical).
- **DropHighNaNs**: Drops columns with a high proportion of missing values.
        """
        )
    # st.markdown("### 🧠 Recommended Preprocessing")
    recommendations = recommend_preprocessing(df)

    # with st.expander("📋 Show Detailed Column Recommendations"):
    #     grouped_recs = defaultdict(list)
    #     for col, rec in recommendations.items():
    #         category = col.split(".")[0] if "." in col else "project"
    #         grouped_recs[category].append((col, rec))

    #     for category, items in grouped_recs.items():
    #         with st.expander(f"📁 {category}"):
    #             for col, rec in items:
    #                 st.markdown(f"🔧 **{col}** → _{rec}_")

    summary_df = summarize_categories(df, recommendations)
    st.markdown("### 📊 Preprocessing Recommendation Summary")
    st.dataframe(summary_df, use_container_width=True)

    return recommendations


def get_user_preprocessing_options():
    """
    Render checkboxes for user-selected preprocessing options in Streamlit.

    Returns:
        tuple: Boolean flags for (enable_scaling,
               enable_encoding, handle_missing).
    """
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
    """
    Display the currently selected preprocessing options as a summary caption.

    Args:
        enable_scaling (bool): Whether to apply scaling.
        enable_encoding (bool): Whether to apply encoding.
        handle_missing (bool): Whether to handle missing values.
    """
    st.caption(
        f"🔧 Applied options: Scaling = {enable_scaling}, "
        f"Encoding = {enable_encoding}, "
        f"Missing Handling = {handle_missing}"
    )


def execute_preprocessing(df, recommendations, show_logs=False):
    """
    Apply preprocessing pipeline and display results.

    Args:
        df (pd.DataFrame): Original input DataFrame.
        recommendations (dict): Preprocessing actions for each column.
        show_logs (bool): Whether to show detailed logs of each step.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df_proc = apply_preprocessing(df, recommendations, show_logs)
    st.write(f"🔄 Data shape changed from `{df.shape}` → `{df_proc.shape}`")
    st.success("✅ Preprocessing Applied!")
    st.dataframe(df_proc.head())
    return df_proc


def apply_preprocessing_and_display(df, recommendations, show_logs, options):
    """
    Wrapper function that applies preprocessing with user-selected options.

    Args:
        df (pd.DataFrame): Original input DataFrame.
        recommendations (dict): Preprocessing actions per column.
        show_logs (bool): Whether to display log messages.
        options (tuple): Tuple of booleans (enable_scaling,
        enable_encoding, handle_missing).

    Returns:
        pd.DataFrame: Transformed DataFrame after preprocessing.
    """
    enable_scaling, enable_encoding, handle_missing = options
    show_selected_options(enable_scaling, enable_encoding, handle_missing)
    df_proc = execute_preprocessing(df, recommendations, show_logs)
    return df_proc


def run_modeling_and_fairness(
    raw_df,
    df_proc,
    target_col,
    group_col,
    include_sensitive_as_feature,
    recommendations,
):
    """
    Run the leakage-safe modeling pipeline and fairness evaluation with
    Streamlit UI.

    Preprocessing (imputation, scaling, encoding) is fit on the training
    split only, using `raw_df` (the original uploaded data) rather than
    `df_proc` (which was preprocessed on the full dataset for the
    exploratory/visualization workflow). `df_proc` is used only to source
    the already-cleaned row set and the human-readable sensitive-attribute
    values used for fairness grouping.

    Args:
        raw_df (pd.DataFrame): Original uploaded DataFrame.
        df_proc (pd.DataFrame): Exploratory-preprocessed DataFrame (row set
            + sensitive attribute values for fairness grouping).
        target_col (str): Name of the target variable column.
        group_col (str): Name of the single sensitive attribute selected
            for fairness grouping.
        include_sensitive_as_feature (bool): Whether `group_col` should also
            be used as a predictive model feature.
        recommendations (dict): Column preprocessing recommendations
            computed on `raw_df`.

    Displays:
        - Classification report
        - ROC AUC score (if available)
        - Fairness metrics and disparity summary for the selected group
    """
    st.markdown("## 🧠 Machine Learning Modeling")

    try:
        results = run_modeling_pipeline(
            raw_df=raw_df,
            df_proc=df_proc,
            target_col=target_col,
            sensitive_col=group_col,
            include_sensitive_in_features=include_sensitive_as_feature,
            recommendations=recommendations,
        )
    except UnsupportedTargetError as e:
        st.error(f"❌ {e}")
        return
    except Exception:
        st.error("❌ Modeling failed.")
        st.text(traceback.format_exc())
        return

    st.markdown("### 🔍 Classification Report")
    st.dataframe(results.report)

    if results.y_prob is not None:
        auc = roc_auc_score(results.y_test, results.y_prob)
        st.markdown(f"📈 ROC AUC: `{auc:.2f}`")

    if results.feature_importance is not None:
        st.markdown("### 🔍 Feature Importance (Permutation)")
        st.dataframe(results.feature_importance.head(10))

    if group_col:
        st.markdown("### ⚖️ Fairness Audit with `fairlearn`")
        st.markdown(f"#### Sensitive Attribute: `{group_col}`")

        try:
            metric_frame, fairness_summary = compute_output_fairness(
                y_true=results.y_test,
                y_pred=results.y_pred,
                sensitive_features=results.sensitive_test,
            )

            st.markdown("📊 Group-wise Metrics")
            st.dataframe(metric_frame.by_group)

            st.markdown("🧾 Summary of Fairness Disparities")
            for key, value in fairness_summary.items():
                if isinstance(value, (int, float)):
                    st.markdown(f"- **{key}**: `{value:.4f}`")
                else:
                    reason = (
                        value.get("reason", "undefined")
                        if isinstance(value, dict)
                        else value
                    )
                    st.markdown(f"- **{key}**: `Undefined` — {reason}")

        except Exception as e:
            st.warning(f"Could not compute fairness for `{group_col}`: {e}")
