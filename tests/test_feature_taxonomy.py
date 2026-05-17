"""Smoke tests for feature taxonomy and actionability scoring."""
import pandas as pd
import pytest

from src.pipelines.counterfactual.actionability import actionability_score
from src.pipelines.counterfactual.feature_taxonomy import (
    FEATURE_TAXONOMY,
    Mutability,
    get_actionable_features,
    get_immutable_features,
    get_features_by_mutability,
    set_conditional_disabled,
    set_socioeconomic_proxies_immutable,
)


BRFSS_2021_FEATURES = {
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
}


def test_taxonomy_covers_all_brfss_2021_features():
    assert set(FEATURE_TAXONOMY) == BRFSS_2021_FEATURES


def test_taxonomy_counts():
    by_m = {m: get_features_by_mutability(m) for m in Mutability}
    assert len(by_m[Mutability.IMMUTABLE]) == 4
    assert len(by_m[Mutability.MONOTONIC_UP]) == 7
    assert len(by_m[Mutability.MONOTONIC_DOWN]) == 8
    assert len(by_m[Mutability.BIDIRECTIONAL]) == 1
    assert len(by_m[Mutability.CONDITIONAL]) == 1


def test_immutables_include_age_sex_stroke_chd():
    immutables = set(get_immutable_features())
    assert immutables == {"Age", "Sex", "Stroke", "HeartDiseaseorAttack"}


def test_actionable_excludes_immutable_and_conditional():
    actionable = set(get_actionable_features())
    assert "Age" not in actionable
    assert "Sex" not in actionable
    assert "DiffWalk" not in actionable
    assert "BMI" in actionable
    assert "Smoker" in actionable


def test_actionability_score_penalizes_immutable_change():
    query = pd.Series({"Age": 5, "BMI": 28, "Smoker": 1})
    cf = pd.Series({"Age": 6, "BMI": 25, "Smoker": 0})  # Age changed -> immutable violation
    score = actionability_score(query, cf)
    assert score["immutable_violations"] == 1
    assert score["total_changes"] == 3


def test_actionability_score_penalizes_wrong_direction_smoker():
    # Smoker is MONOTONIC_DOWN; 0 -> 1 is wrong direction (started smoking)
    query = pd.Series({"Smoker": 0, "BMI": 25})
    cf = pd.Series({"Smoker": 1, "BMI": 28})
    score = actionability_score(query, cf)
    assert score["wrong_direction_violations"] >= 1


def test_actionability_score_rewards_correct_direction():
    # Smoker 1 -> 0 (quit), PhysActivity 0 -> 1 (started)
    query = pd.Series({"Smoker": 1, "PhysActivity": 0, "Age": 5})
    cf = pd.Series({"Smoker": 0, "PhysActivity": 1, "Age": 5})
    score = actionability_score(query, cf)
    assert score["immutable_violations"] == 0
    assert score["wrong_direction_violations"] == 0
    assert score["actionable_changes"] == 2
    assert score["score"] == 1.0


# --------------------------------------------------------------------
# Tests that the actionability score respects ablation-flag collapses.
# These guard against the bug where actionability_score used raw
# spec.mutability instead of _effective_mutability, causing scoring
# inconsistency with the constraint set under 4-class and conservative
# SE-proxy variants.
# --------------------------------------------------------------------


@pytest.fixture
def reset_flags():
    """Reset both ablation flags after each test that touches them."""
    yield
    set_conditional_disabled(False)
    set_socioeconomic_proxies_immutable(False)


def test_actionability_diffwalk_default_5class_is_conditional_violation(reset_flags):
    # Default 5-class: DiffWalk is CONDITIONAL -> any change counts as wrong_dir
    query = pd.Series({"DiffWalk": 1, "BMI": 28})
    cf = pd.Series({"DiffWalk": 0, "BMI": 25})
    score = actionability_score(query, cf)
    assert score["wrong_direction_violations"] == 1, "DiffWalk change should violate under 5-class"


def test_actionability_diffwalk_4class_collapsed_to_monotonic_down(reset_flags):
    # 4-class: DiffWalk -> MONOTONIC_DOWN; 1->0 is correct direction
    set_conditional_disabled(True)
    query = pd.Series({"DiffWalk": 1, "BMI": 28})
    cf = pd.Series({"DiffWalk": 0, "BMI": 25})
    score = actionability_score(query, cf)
    assert score["wrong_direction_violations"] == 0, "DiffWalk 1->0 must be actionable under 4-class"
    assert score["actionable_changes"] == 2


def test_actionability_diffwalk_4class_wrong_direction_still_caught(reset_flags):
    # 4-class: DiffWalk MONOTONIC_DOWN; 0->1 must remain a wrong-dir violation
    set_conditional_disabled(True)
    query = pd.Series({"DiffWalk": 0, "BMI": 28})
    cf = pd.Series({"DiffWalk": 1, "BMI": 25})
    score = actionability_score(query, cf)
    assert score["wrong_direction_violations"] == 1, "DiffWalk 0->1 must violate under 4-class"


def test_actionability_se_proxy_conservative_treats_income_as_immutable(reset_flags):
    # Conservative SE-proxy: Income MONOTONIC_UP -> IMMUTABLE
    # Any change must register as immutable_violations (not actionable_changes)
    set_socioeconomic_proxies_immutable(True)
    query = pd.Series({"Income": 3, "BMI": 28})
    cf = pd.Series({"Income": 5, "BMI": 25})
    score = actionability_score(query, cf)
    assert score["immutable_violations"] == 1, "Income change must be immutable violation under conservative"
    assert score["actionable_changes"] == 1, "Only BMI change should be actionable"


def test_actionability_education_conservative_not_double_counted(reset_flags):
    # Without flag (default): Education MONOTONIC_UP, 3->5 is actionable
    query = pd.Series({"Education": 3, "BMI": 28})
    cf = pd.Series({"Education": 5, "BMI": 25})
    score_default = actionability_score(query, cf)
    assert score_default["actionable_changes"] == 2
    assert score_default["immutable_violations"] == 0

    # With conservative flag: Education -> IMMUTABLE, same change becomes immutable violation
    set_socioeconomic_proxies_immutable(True)
    score_cons = actionability_score(query, cf)
    assert score_cons["immutable_violations"] == 1
    assert score_cons["actionable_changes"] == 1
