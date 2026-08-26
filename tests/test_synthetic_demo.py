import importlib.util
import json
import math
from io import StringIO
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from bias_audit_tool.modeling.fairness import compute_output_fairness
from bias_audit_tool.modeling.target_validation import preferred_target_column
from bias_audit_tool.modeling.target_validation import validate_classification_target
from bias_audit_tool.preprocessing.modeling_pipeline import run_modeling_pipeline
from bias_audit_tool.preprocessing.preprocess import recommend_preprocessing
from bias_audit_tool.preprocessing.recommend_columns import (
    direct_columns_for_sensitive_attribute,
)
from bias_audit_tool.preprocessing.recommend_columns import (
    recommend_demographic_columns,
)
from bias_audit_tool.visualization.evaluation_plots import (
    build_confusion_matrix_figure,
)
from bias_audit_tool.visualization.evaluation_plots import build_roc_curve_figure


ROOT = Path(__file__).resolve().parents[1]
DEMO_CSV = ROOT / "bias_audit_tool" / "sample_data" / "demo.csv"
DEMO_BENCHMARK = ROOT / "bias_audit_tool" / "sample_data" / "demo_benchmark.json"
GENERATOR_PATH = ROOT / "scripts" / "generate_demo_data.py"

EXPECTED_COLUMNS = [
    "feature_a",
    "feature_b",
    "feature_c",
    "age_band",
    "demo_group",
    "outcome",
]
FORBIDDEN_IDENTIFIER_SUBSTRINGS = (
    "case_id",
    "patient",
    "mrn",
    "ssn",
    "address",
    "dob",
    "date_of_birth",
    "name",
    "email",
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_demo_data", GENERATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_demo():
    return pd.read_csv(DEMO_CSV)


def test_committed_demo_exists_and_is_non_empty():
    assert DEMO_CSV.is_file()
    assert DEMO_CSV.stat().st_size > 0
    df = _read_demo()
    assert not df.empty
    assert 200 <= len(df) <= 1000
    assert list(df.columns) == EXPECTED_COLUMNS


def test_demo_schema_has_binary_target_and_two_groups():
    df = _read_demo()
    assert set(df["outcome"].dropna().unique()) == {0, 1}
    assert df["demo_group"].nunique(dropna=True) >= 2
    assert set(df["demo_group"].dropna().unique()) == {"Group A", "Group B"}


def test_demo_required_columns_are_not_all_null():
    df = _read_demo()
    for column in EXPECTED_COLUMNS:
        assert df[column].notna().any(), column


def test_demo_has_intentional_missingness_without_identifiers():
    df = _read_demo()
    assert df["feature_b"].isna().any()
    lowered = [column.lower() for column in df.columns]
    for needle in FORBIDDEN_IDENTIFIER_SUBSTRINGS:
        assert all(needle not in name for name in lowered)


def test_demo_target_passes_binary_validation():
    df = _read_demo()
    result = validate_classification_target(df["outcome"], target_name="outcome")
    assert result.kind == "binary"
    assert result.n_classes == 2


def test_demo_initial_target_preference_is_binary_and_not_the_group_column():
    df = _read_demo()
    default = preferred_target_column(
        df,
        deprioritized=direct_columns_for_sensitive_attribute(
            "demo_group_mapped", df.columns
        ),
    )
    validate_classification_target(df[default], target_name=default)
    assert default != "feature_a"
    assert default != "demo_group"


def test_demo_group_is_a_recommended_sensitive_candidate():
    df = _read_demo()
    _, candidates = recommend_demographic_columns(df)
    assert "demo_group" in candidates


def test_demo_group_is_selectable_after_exploratory_preprocessing():
    from bias_audit_tool.preprocessing.transform import apply_preprocessing

    df = _read_demo()
    recommendations = recommend_preprocessing(df)
    df_proc = apply_preprocessing(df, recommendations, show_logs=False)
    df_proc, candidates = recommend_demographic_columns(df_proc)
    assert "demo_group_mapped" in df_proc.columns
    assert "demo_group_mapped" in candidates
    assert set(df_proc["demo_group_mapped"].dropna().unique()) == {
        "Group A",
        "Group B",
    }


def test_generator_is_deterministic_and_matches_committed_demo():
    generator = _load_generator()
    first = generator.dataframe_to_csv_text(generator.build_demo_dataframe())
    second = generator.dataframe_to_csv_text(generator.build_demo_dataframe())
    assert first == second

    # CSV float text can differ by 1 ULP across platforms; compare numerically.
    committed = pd.read_csv(DEMO_CSV)
    generated = pd.read_csv(StringIO(first))
    pd.testing.assert_frame_equal(
        committed, generated, check_dtype=False, rtol=1e-12, atol=1e-15
    )
    expected_benchmark = json.dumps(generator.SYNTHETIC_BENCHMARK, indent=2) + "\n"
    assert DEMO_BENCHMARK.read_text(encoding="utf-8") == expected_benchmark


def test_demo_runs_through_modeling_pipeline_with_finite_metrics():
    from bias_audit_tool.preprocessing.transform import apply_preprocessing

    df = _read_demo()
    recommendations = recommend_preprocessing(df)
    df_proc = apply_preprocessing(df, recommendations, show_logs=False)
    df_proc, _ = recommend_demographic_columns(df_proc)
    result = run_modeling_pipeline(
        raw_df=df,
        df_proc=df_proc,
        target_col="outcome",
        sensitive_col="demo_group_mapped",
        include_sensitive_in_features=False,
        recommendations=recommendations,
        test_size=0.2,
        random_state=42,
    )
    assert result.report is not None
    assert len(result.y_test) > 0
    assert result.feature_importance is not None
    assert not result.feature_importance.empty
    assert set(result.sensitive_test.dropna().unique()) <= {"Group A", "Group B"}

    numeric_cells = pd.to_numeric(
        result.report.select_dtypes(include="number").stack(),
        errors="coerce",
    )
    assert numeric_cells.notna().any()
    assert numeric_cells.map(lambda value: math.isfinite(float(value))).all()

    auc = roc_auc_score(result.y_test, result.y_prob)
    assert math.isfinite(float(auc))

    metric_frame, disparities = compute_output_fairness(
        result.y_test, result.y_pred, result.sensitive_test
    )
    assert metric_frame is not None
    assert "Demographic Parity Difference" in disparities
    assert "Equalized Odds Difference" in disparities
    dp = disparities["Demographic Parity Difference"]
    eo = disparities["Equalized Odds Difference"]
    assert isinstance(dp, (int, float)) and math.isfinite(float(dp))
    assert isinstance(eo, (int, float)) and math.isfinite(float(eo))

    fig_cm, matrix = build_confusion_matrix_figure(result.y_test, result.y_pred)
    assert fig_cm is not None
    assert matrix.size > 0
    fig_roc, roc_message = build_roc_curve_figure(result.y_test, result.y_prob)
    assert roc_message is None
    assert fig_roc is not None
