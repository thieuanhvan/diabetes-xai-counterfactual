"""Tests for src/evaluate/per_feature.py"""
from __future__ import annotations

import pandas as pd
import pytest

from src.pipelines.evaluate.per_feature import (
    aggregate_per_feature,
    per_feature_breakdown_one_query,
)
from src.pipelines.counterfactual.feature_taxonomy import (
    FEATURE_TAXONOMY,
    Mutability,
    set_conditional_disabled,
    set_socioeconomic_proxies_immutable,
)


def _make_query():
    """Minimal valid BRFSS query row."""
    row = {f: spec.value_range[0] for f, spec in FEATURE_TAXONOMY.items()}
    # Set BMI middle
    row["BMI"] = 25.0
    return pd.Series(row)


def test_no_changes_means_zero_counts():
    query = _make_query()
    cfs = pd.DataFrame([query.copy()])  # identical CF -> no changes
    result = per_feature_breakdown_one_query(query, cfs)
    for feature, stats in result.items():
        assert stats["n_cf_changed"] == 0


def test_immutable_violation_counted():
    query = _make_query()
    cf = query.copy()
    cf["Age"] = query["Age"] + 5  # immutable changed
    cfs = pd.DataFrame([cf])
    result = per_feature_breakdown_one_query(query, cfs)
    assert result["Age"]["n_immutable"] == 1
    assert result["Age"]["n_wrong_dir"] == 0
    assert result["Age"]["n_actionable"] == 0


def test_monotonic_up_wrong_direction_counted():
    query = _make_query()
    query["PhysActivity"] = 1  # already at max
    cf = query.copy()
    cf["PhysActivity"] = 0  # decrease — wrong direction for MONOTONIC_UP
    cfs = pd.DataFrame([cf])
    result = per_feature_breakdown_one_query(query, cfs)
    assert result["PhysActivity"]["n_wrong_dir"] == 1
    assert result["PhysActivity"]["n_actionable"] == 0


def test_monotonic_down_correct_direction_counted():
    query = _make_query()
    query["Smoker"] = 1
    cf = query.copy()
    cf["Smoker"] = 0  # decrease — correct for MONOTONIC_DOWN
    cfs = pd.DataFrame([cf])
    result = per_feature_breakdown_one_query(query, cfs)
    assert result["Smoker"]["n_actionable"] == 1
    assert result["Smoker"]["n_wrong_dir"] == 0


def test_bidirectional_always_actionable():
    query = _make_query()
    cf = query.copy()
    cf["BMI"] = 22.0  # decrease
    cfs = pd.DataFrame([cf])
    result = per_feature_breakdown_one_query(query, cfs)
    assert result["BMI"]["n_actionable"] == 1


def test_aggregate_across_queries():
    query = _make_query()
    cf_good = query.copy()
    cf_good["Smoker"] = 0  # decrease
    query["Smoker"] = 1

    cf_bad = query.copy()
    cf_bad["Age"] = query["Age"] + 3  # immutable violation

    per_q = [
        per_feature_breakdown_one_query(query, pd.DataFrame([cf_good])),
        per_feature_breakdown_one_query(query, pd.DataFrame([cf_bad])),
    ]
    agg = aggregate_per_feature(per_q, mode_label="test")
    smoker = agg[agg["feature"] == "Smoker"].iloc[0]
    age = agg[agg["feature"] == "Age"].iloc[0]

    assert smoker["n_actionable"] == 1
    assert smoker["actionability_rate"] == 1.0
    assert age["n_immutable"] == 1
    assert age["violation_rate"] == 1.0


def test_aggregate_handles_none_breakdowns():
    """Aggregate should skip None entries (failed DiCE queries)."""
    query = _make_query()
    cf = query.copy()
    cf["BMI"] = 22.0

    per_q = [
        None,  # failed query
        per_feature_breakdown_one_query(query, pd.DataFrame([cf])),
        None,
    ]
    agg = aggregate_per_feature(per_q, mode_label="test")
    bmi = agg[agg["feature"] == "BMI"].iloc[0]
    assert bmi["n_total_cf_changes"] == 1
    assert bmi["n_queries_with_change"] == 1


# --------------------------------------------------------------------
# Ablation-flag-aware scoring tests for per_feature_breakdown_one_query.
# These guard against the bug where per_feature.py used raw
# spec.mutability instead of _effective_mutability, causing per-feature
# CSV to be inconsistent with the aggregate actionability_score under
# 4-class and conservative SE-proxy variants.
# --------------------------------------------------------------------


@pytest.fixture
def reset_flags():
    yield
    set_conditional_disabled(False)
    set_socioeconomic_proxies_immutable(False)


def test_per_feature_diffwalk_5class_default_is_wrong_dir(reset_flags):
    # 5-class: DiffWalk CONDITIONAL -> any change is wrong_dir
    query = _make_query()
    query["DiffWalk"] = 1
    cf = query.copy()
    cf["DiffWalk"] = 0
    cfs = pd.DataFrame([cf])
    result = per_feature_breakdown_one_query(query, cfs)
    assert result["DiffWalk"]["n_wrong_dir"] == 1
    assert result["DiffWalk"]["n_actionable"] == 0


def test_per_feature_diffwalk_4class_collapsed_is_actionable(reset_flags):
    # 4-class: DiffWalk -> MONOTONIC_DOWN; 1->0 is correct direction
    set_conditional_disabled(True)
    query = _make_query()
    query["DiffWalk"] = 1
    cf = query.copy()
    cf["DiffWalk"] = 0
    cfs = pd.DataFrame([cf])
    result = per_feature_breakdown_one_query(query, cfs)
    assert result["DiffWalk"]["n_actionable"] == 1
    assert result["DiffWalk"]["n_wrong_dir"] == 0


def test_per_feature_income_conservative_treated_immutable(reset_flags):
    # Conservative SE-proxy: Income MONOTONIC_UP -> IMMUTABLE
    set_socioeconomic_proxies_immutable(True)
    query = _make_query()
    query["Income"] = 3
    cf = query.copy()
    cf["Income"] = 5
    cfs = pd.DataFrame([cf])
    result = per_feature_breakdown_one_query(query, cfs)
    assert result["Income"]["n_immutable"] == 1
    assert result["Income"]["n_actionable"] == 0
    assert result["Income"]["n_wrong_dir"] == 0


def test_aggregate_taxonomy_class_reflects_effective_mutability(reset_flags):
    # Aggregate output's taxonomy_class column should reflect the effective
    # class (after ablation collapses), not the raw default 5-class label.
    set_socioeconomic_proxies_immutable(True)
    query = _make_query()
    query["Income"] = 3
    cf = query.copy()
    cf["Income"] = 5
    per_q = [per_feature_breakdown_one_query(query, pd.DataFrame([cf]))]
    agg = aggregate_per_feature(per_q, mode_label="test")
    income_row = agg[agg["feature"] == "Income"].iloc[0]
    assert income_row["taxonomy_class"] == "immutable", (
        f"Income should be labelled 'immutable' under conservative variant, got {income_row['taxonomy_class']}"
    )
