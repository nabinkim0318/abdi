import traceback

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

from bias_audit_tool.data.validation import DataValidationError
from bias_audit_tool.data.validation import describe_class_distribution
from bias_audit_tool.data.validation import SEVERITY_ERROR
from bias_audit_tool.data.validation import SEVERITY_WARNING
from bias_audit_tool.modeling.fairness import assess_group_support
from bias_audit_tool.modeling.fairness import BOOTSTRAP_CI_CAVEAT
from bias_audit_tool.modeling.fairness import bootstrap_output_fairness
from bias_audit_tool.modeling.fairness import compute_group_support
from bias_audit_tool.modeling.fairness import compute_output_fairness
from bias_audit_tool.modeling.fairness import GROUP_SUPPORT_CAPTION
from bias_audit_tool.modeling.fairness import GROUP_SUPPORT_SECTION_TITLE
from bias_audit_tool.modeling.fairness import render_fairness_caveats
from bias_audit_tool.modeling.fairness import WARNING_ZERO_NEGATIVE_LABELS
from bias_audit_tool.modeling.fairness import WARNING_ZERO_POSITIVE_LABELS
from bias_audit_tool.modeling.target_validation import UnsupportedTargetError
from bias_audit_tool.modeling.target_validation import validate_classification_target
from bias_audit_tool.preprocessing.modeling_pipeline import (
    prepare_modeling_target_frame,
)
from bias_audit_tool.preprocessing.modeling_pipeline import run_modeling_pipeline
from bias_audit_tool.preprocessing.preprocess import recommend_preprocessing
from bias_audit_tool.preprocessing.summary import summarize_categories
from bias_audit_tool.preprocessing.transform import apply_preprocessing
from bias_audit_tool.visualization.evaluation_plots import (
    build_confusion_matrix_figure,
)
from bias_audit_tool.visualization.evaluation_plots import build_roc_curve_figure


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


def apply_preprocessing_and_display(df, recommendations, show_logs):
    """
    Apply exploratory preprocessing and show a preview.

    Modeling-path scaling/encoding is fit later on the train split only
    and is not controlled by this exploratory step.
    """
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
        - Confusion matrix and ROC curve on the same held-out predictions
        - Permutation feature importance
        - Fairness metrics and disparity summary for the selected group
    """
    st.markdown("## 🧠 Model Evaluation")

    try:
        _, y_effective = prepare_modeling_target_frame(raw_df, df_proc, target_col)
        present_supported_class_distribution(
            y_effective, target_col, renderer=_render_class_distribution
        )
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
    except DataValidationError as e:
        for issue in e.issues:
            if issue.severity == SEVERITY_ERROR:
                st.error(issue.message)
            elif issue.severity == SEVERITY_WARNING:
                st.warning(issue.message)
        return
    except Exception:
        st.error("❌ Modeling failed.")
        st.text(traceback.format_exc())
        return

    for issue in results.modeling_warnings:
        st.warning(issue.message)

    st.markdown("### 🔍 Classification Report")
    st.dataframe(results.report)

    if results.y_prob is not None:
        try:
            auc = roc_auc_score(results.y_test, results.y_prob)
            st.markdown(f"📈 ROC AUC: `{auc:.2f}`")
        except ValueError:
            st.info(
                "ROC AUC is unavailable because the held-out test split "
                "contains only one class."
            )

    plot_col, roc_col = st.columns(2)
    with plot_col:
        st.markdown("### Confusion Matrix")
        cm_fig, _ = build_confusion_matrix_figure(results.y_test, results.y_pred)
        st.pyplot(cm_fig)
        plt.close(cm_fig)
    with roc_col:
        st.markdown("### ROC Curve")
        roc_fig, roc_message = build_roc_curve_figure(results.y_test, results.y_prob)
        if roc_fig is not None:
            st.pyplot(roc_fig)
            plt.close(roc_fig)
        else:
            st.info(roc_message)

    if results.feature_importance is not None:
        st.markdown("### 🔍 Feature Importance (Permutation)")
        st.dataframe(results.feature_importance.head(10))

    if group_col:
        st.markdown("## ⚖️ Group Fairness Diagnostics")
        st.markdown(f"#### Sensitive Attribute: `{group_col}`")
        render_fairness_caveats()

        try:
            with st.spinner(
                "Computing held-out group support and DP/EO bootstrap "
                "intervals..."
            ):
                metric_frame, fairness_summary = compute_output_fairness(
                    y_true=results.y_test,
                    y_pred=results.y_pred,
                    sensitive_features=results.sensitive_test,
                )
                support_df = compute_group_support(
                    y_true=results.y_test,
                    y_pred=results.y_pred,
                    sensitive_features=results.sensitive_test,
                )
                support_warnings = assess_group_support(support_df)
                bootstrap_results = bootstrap_output_fairness(
                    y_true=results.y_test,
                    y_pred=results.y_pred,
                    sensitive_features=results.sensitive_test,
                )

            render_group_support_section(support_df, support_warnings)

            st.markdown("📊 Group-wise Metrics")
            st.dataframe(metric_frame.by_group)

            st.markdown("🧾 Summary of group disparities")
            for key, value in fairness_summary.items():
                if key in bootstrap_results:
                    render_disparity_with_bootstrap(
                        key, value, bootstrap_results[key]
                    )
                elif isinstance(value, (int, float)):
                    st.markdown(f"- **{key}**: `{value:.4f}`")
                else:
                    reason = (
                        value.get("reason", "undefined")
                        if isinstance(value, dict)
                        else value
                    )
                    st.markdown(f"- **{key}**: `Undefined` — {reason}")

            st.caption(BOOTSTRAP_CI_CAVEAT)

        except Exception as e:
            st.warning(f"Could not compute fairness for `{group_col}`: {e}")


def present_supported_class_distribution(y, target_name, renderer):
    """Render class counts only after the target is a supported binary label.

    Continuous, near-unique, and multiclass columns must fail validation
    before any per-value table is shown, so identifier-like values are not
    listed in the UI and then rejected.
    """
    validate_classification_target(y, target_name=target_name)
    renderer(y, target_name)


def _render_class_distribution(y, target_name):
    """Show effective target counts using original labels."""
    distribution = describe_class_distribution(y)
    n = distribution["n_effective_rows"]
    st.caption(
        f"Effective modeling rows for `{target_name}` (target non-null): **{n}**."
    )
    rows = []
    for item in distribution["classes"]:
        rows.append(
            {
                "class": item["label"],
                "count": item["count"],
                "percentage": f"{item['percentage']:.1%}",
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_group_support_section(support_df, support_warnings):
    """Render held-out group-support counts and sparse-support warnings."""
    st.markdown(f"### {GROUP_SUPPORT_SECTION_TITLE}")
    st.caption(GROUP_SUPPORT_CAPTION)
    st.dataframe(support_df, use_container_width=True)
    render_group_support_warnings(support_warnings)


def render_group_support_warnings(support_warnings):
    """Render structured support warnings without fairness verdicts."""
    unsupported = {
        WARNING_ZERO_POSITIVE_LABELS,
        WARNING_ZERO_NEGATIVE_LABELS,
    }
    for item in support_warnings:
        message = item.get("message", "")
        group = item.get("group")
        prefix = f"**{group}.** " if group is not None else ""
        text = f"{prefix}{message}"
        if item.get("code") in unsupported:
            st.warning(text)
        else:
            st.info(text)


def render_disparity_with_bootstrap(key, point_estimate, bootstrap_result):
    """Render a Fairlearn point estimate plus optional bootstrap interval."""
    if isinstance(point_estimate, (int, float)):
        st.markdown(f"- **{key}**: `{point_estimate:.4f}`")
    elif isinstance(point_estimate, dict):
        reason = point_estimate.get("reason", "undefined")
        st.markdown(f"- **{key}**: `Undefined` — {reason}")
        return
    else:
        st.markdown(f"- **{key}**: `{point_estimate}`")
        return

    if not isinstance(bootstrap_result, dict):
        return

    n_requested = bootstrap_result.get("n_requested")
    n_valid = bootstrap_result.get("n_valid")
    if bootstrap_result.get("status") == "ok":
        ci_lower = bootstrap_result["ci_lower"]
        ci_upper = bootstrap_result["ci_upper"]
        level = bootstrap_result.get("confidence_level", 0.95)
        pct = int(round(level * 100))
        st.markdown(f"  - {pct}% bootstrap CI: `[{ci_lower:.4f}, {ci_upper:.4f}]`")
    else:
        reason = bootstrap_result.get("reason", "undefined")
        st.markdown(f"  - bootstrap CI: `unavailable` — {reason}")

    if n_valid is not None and n_requested is not None:
        st.markdown(f"  - valid replicates: `{n_valid} / {n_requested}`")

    for item in bootstrap_result.get("warnings") or []:
        st.caption(item.get("message", ""))
