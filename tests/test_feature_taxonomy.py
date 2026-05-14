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
