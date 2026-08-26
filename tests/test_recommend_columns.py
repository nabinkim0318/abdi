from pathlib import Path

import pandas as pd

from bias_audit_tool.preprocessing import recommend_columns as rec_mod
from bias_audit_tool.preprocessing.recommend_columns import (
    direct_columns_for_sensitive_attribute,
)
from bias_audit_tool.preprocessing.recommend_columns import identify_by_hierarchy
from bias_audit_tool.preprocessing.recommend_columns import (
    recommend_demographic_columns,
)
from bias_audit_tool.preprocessing.recommend_columns import (
    SENSITIVE_ATTRIBUTE_CANDIDATE_CAPTION,
)


def _heuristic_frame():
    # Cardinality is 2 for every grouping column so metadata filters pass.
    n = 10
    return pd.DataFrame(
        {
            "gender": ["F"] * n + ["M"] * n,
            "sex": ["F"] * n + ["M"] * n,
            "race": ["A"] * n + ["B"] * n,
            "ethnicity": ["X"] * n + ["Y"] * n,
            "age": [20] * n + [40] * n,
            "nationality": ["US"] * n + ["CA"] * n,
            "citizenship": ["US"] * n + ["CA"] * n,
            "region_id": ["east"] * n + ["west"] * n,
            "family_history_of_cancer": ["yes"] * n + ["no"] * n,
            "device_orientation": ["portrait"] * n + ["landscape"] * n,
            "nacionalidad": ["a"] * n + ["b"] * n,
            "score": list(range(n * 2)),
        }
    )


def test_identify_demographic_columns():
    df = pd.DataFrame(columns=["gender", "age", "zipcode", "score", "likes_cats"])
    result = identify_by_hierarchy(df)
    # Only gender and age are in DEMOGRAPHIC_CATEGORIES, zipcode is not
    assert set(result) == {"gender", "age"}


def test_recommendation_caption_does_not_claim_automatic_detection():
    assert SENSITIVE_ATTRIBUTE_CANDIDATE_CAPTION == (
        "Candidate sensitive attributes based on column-name and metadata "
        "heuristics — review before use."
    )
    lowered = SENSITIVE_ATTRIBUTE_CANDIDATE_CAPTION.lower()
    assert "automatic" not in lowered
    assert "detection" not in lowered


def test_dead_keyword_and_value_matchers_are_not_present():
    # Option B: unused regex/value matchers were removed rather than wired.
    assert not hasattr(rec_mod, "DEMOGRAPHIC_KEYWORDS")
    assert not hasattr(rec_mod, "VALUE_PATTERNS")


def test_expected_name_candidates_are_recommended():
    _, candidates = recommend_demographic_columns(_heuristic_frame())
    for col in ("gender", "sex", "race", "ethnicity", "age"):
        assert col in candidates


def test_false_positive_risk_names_are_currently_recommended():
    # Documented current heuristic behavior, not a claim that these are
    # solved. Names contain "region", "family", or "orientation".
    _, candidates = recommend_demographic_columns(_heuristic_frame())
    for col in ("region_id", "family_history_of_cancer", "device_orientation"):
        assert col in candidates


def test_unsupported_vocabulary_is_not_guaranteed_to_be_detected():
    names = identify_by_hierarchy(_heuristic_frame())
    for col in ("nationality", "citizenship", "nacionalidad", "score"):
        assert col not in names

    _, candidates = recommend_demographic_columns(_heuristic_frame())
    for col in ("nationality", "citizenship", "nacionalidad", "score"):
        assert col not in candidates


def test_readme_does_not_claim_automatic_detection_or_fairness_verdicts():
    readme = Path("README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    assert "automatic recommendation of sensitive columns" not in lowered
    assert "automatic sensitive attribute detection" not in lowered
    assert "regulatory teams" not in lowered
    assert "dei officers" not in lowered
    assert (
        "Candidate sensitive attributes based on column-name and metadata" in readme
    )
    assert "review before use" in readme
    assert "exploratory bias and fairness diagnostics" in lowered
    assert "shap" not in lowered
    assert "automated reports" not in lowered


def test_direct_columns_for_mapped_race_exclude_proxies():
    columns = [
        "age",
        "race_Black",
        "race_White",
        "race_Asian",
        "zipcode",
        "income",
        "education",
        "race_mapped",
    ]
    related = direct_columns_for_sensitive_attribute("race_mapped", columns)
    assert related == [
        "race_Black",
        "race_White",
        "race_Asian",
        "race_mapped",
    ]


def test_recommend_demographic_columns_merges_bare_race_onehot_dummies():
    n = 10
    df = pd.DataFrame(
        {
            "race_Black": [1, 0] * n,
            "race_White": [0, 1] * n,
            "score": list(range(n * 2)),
        }
    )
    merged, candidates = recommend_demographic_columns(df)
    assert "race_mapped" in merged.columns
    assert "race_Black" not in merged.columns
    assert "race_White" not in merged.columns
    assert "race_mapped" in candidates
