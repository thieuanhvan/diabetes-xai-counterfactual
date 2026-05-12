"""Actionability taxonomy for BRFSS 2021 features.

Defines for each feature:
- mutability: immutable | bidirectional | monotonic_up | monotonic_down | conditional
- value_range: valid encoded values
- semantic_label: human-readable description

Used by:
- DiCE-ML to constrain counterfactual generation (features_to_vary, permitted_range)
- Actionability metric to score CFs (penalize immutable violations, wrong-direction changes)

Taxonomy design (v1, 12/05/2026) — "intervention-direction" lens:
- IMMUTABLE: demographic / biological / irreversible history (cannot be changed)
- MONOTONIC_UP: positive behaviors/states (CF may only INCREASE, e.g. PhysActivity 0->1)
- MONOTONIC_DOWN: negative behaviors/states (CF may only DECREASE, e.g. Smoker 1->0)
- BIDIRECTIONAL: both extremes can be unhealthy (e.g. BMI clinical range)
- CONDITIONAL: depends on other features, cannot be acted on directly (e.g. DiffWalk)

Total: 21 features = 4 immutable + 7 monotonic_up + 8 monotonic_down + 1 bidirectional + 1 conditional.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class Mutability(str, Enum):
    IMMUTABLE = "immutable"
    BIDIRECTIONAL = "bidirectional"
    MONOTONIC_UP = "monotonic_up"
    MONOTONIC_DOWN = "monotonic_down"
    CONDITIONAL = "conditional"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    mutability: Mutability
    value_range: Tuple[float, float]   # (min, max) of valid encoded values
    semantic_label: str


# BRFSS 2021 feature taxonomy
FEATURE_TAXONOMY: Dict[str, FeatureSpec] = {
    # ---- Immutable: demographic, biological, irreversible history (4) ----
    "Age": FeatureSpec(
        "Age", Mutability.IMMUTABLE, (1, 13),
        "Age category (1=18-24, 13=80+)"),
    "Sex": FeatureSpec(
        "Sex", Mutability.IMMUTABLE, (0, 1),
        "Sex (0=female, 1=male)"),
    "Stroke": FeatureSpec(
        "Stroke", Mutability.IMMUTABLE, (0, 1),
        "Ever had a stroke (irreversible history)"),
    "HeartDiseaseorAttack": FeatureSpec(
        "HeartDiseaseorAttack", Mutability.IMMUTABLE, (0, 1),
        "Had CHD or MI (irreversible history)"),

    # ---- Monotonic up: positive behaviors / states (7) ----
    "PhysActivity": FeatureSpec(
        "PhysActivity", Mutability.MONOTONIC_UP, (0, 1),
        "Physical activity in last 30 days (0->1 = recommended)"),
    "Fruits": FeatureSpec(
        "Fruits", Mutability.MONOTONIC_UP, (0, 1),
        "Consume fruit >=1/day (0->1 = recommended)"),
    "Veggies": FeatureSpec(
        "Veggies", Mutability.MONOTONIC_UP, (0, 1),
        "Consume vegetables >=1/day (0->1 = recommended)"),
    "AnyHealthcare": FeatureSpec(
        "AnyHealthcare", Mutability.MONOTONIC_UP, (0, 1),
        "Has healthcare coverage (0->1 = recommended)"),
    "CholCheck": FeatureSpec(
        "CholCheck", Mutability.MONOTONIC_UP, (0, 1),
        "Cholesterol check in last 5 years (0->1 = recommended)"),
    "Education": FeatureSpec(
        "Education", Mutability.MONOTONIC_UP, (1, 6),
        "Education level (1=none, 6=college graduate)"),
    "Income": FeatureSpec(
        "Income", Mutability.MONOTONIC_UP, (1, 11),
        "Income category 2021 (1=<$10K, 11=>=$200K)"),

    # ---- Monotonic down: negative behaviors / states (8) ----
    "Smoker": FeatureSpec(
        "Smoker", Mutability.MONOTONIC_DOWN, (0, 1),
        "Smoked >=100 cigarettes lifetime (1->0 = quit)"),
    "HvyAlcoholConsump": FeatureSpec(
        "HvyAlcoholConsump", Mutability.MONOTONIC_DOWN, (0, 1),
        "Heavy alcohol consumption (1->0 = reduce)"),
    "NoDocbcCost": FeatureSpec(
        "NoDocbcCost", Mutability.MONOTONIC_DOWN, (0, 1),
        "Could not see doctor due to cost (1->0 = improved access)"),
    "HighBP": FeatureSpec(
        "HighBP", Mutability.MONOTONIC_DOWN, (0, 1),
        "High blood pressure (1->0 = controlled)"),
    "HighChol": FeatureSpec(
        "HighChol", Mutability.MONOTONIC_DOWN, (0, 1),
        "High cholesterol (1->0 = controlled)"),
    "GenHlth": FeatureSpec(
        "GenHlth", Mutability.MONOTONIC_DOWN, (1, 5),
        "General health (1=excellent -> 5=poor; lower = better)"),
    "MentHlth": FeatureSpec(
        "MentHlth", Mutability.MONOTONIC_DOWN, (0, 30),
        "Mental health bad days in last 30 (lower = better)"),
    "PhysHlth": FeatureSpec(
        "PhysHlth", Mutability.MONOTONIC_DOWN, (0, 30),
        "Physical health bad days in last 30 (lower = better)"),

    # ---- Bidirectional: both extremes can be unhealthy (1) ----
    "BMI": FeatureSpec(
        "BMI", Mutability.BIDIRECTIONAL, (18.5, 35.0),
        "Body Mass Index (clinical healthy range; underweight and obese both elevate risk)"),

    # ---- Conditional: caused by other conditions, cannot act on directly (1) ----
    "DiffWalk": FeatureSpec(
        "DiffWalk", Mutability.CONDITIONAL, (0, 1),
        "Serious difficulty walking (depends on co-morbidity)"),
}


def get_actionable_features() -> List[str]:
    """Features DiCE may vary: everything except IMMUTABLE and CONDITIONAL."""
    return [
        name for name, spec in FEATURE_TAXONOMY.items()
        if spec.mutability not in (Mutability.IMMUTABLE, Mutability.CONDITIONAL)
    ]


def get_immutable_features() -> List[str]:
    return [
        name for name, spec in FEATURE_TAXONOMY.items()
        if spec.mutability == Mutability.IMMUTABLE
    ]


def get_features_by_mutability(mutability: Mutability) -> List[str]:
    return [
        name for name, spec in FEATURE_TAXONOMY.items()
        if spec.mutability == mutability
    ]


def get_feature_ranges() -> Dict[str, Tuple[float, float]]:
    """For DiCE `permitted_range` parameter."""
    return {name: spec.value_range for name, spec in FEATURE_TAXONOMY.items()}


def get_continuous_features() -> List[str]:
    """Features treated as continuous by DiCE (range > 2)."""
    return [
        name for name, spec in FEATURE_TAXONOMY.items()
        if (spec.value_range[1] - spec.value_range[0]) > 2
    ]
def get_discrete_features() -> List[str]:
    """Features whose CF values should be rounded to nearest integer post-DiCE.

    Reason: in dice_runner we pass ALL features as continuous to bypass
    dice-ml 0.11 categorical stringify bug. DiCE then samples floats for
    integer-valued features (e.g. Smoker=0.73). We round these back to int
    before evaluation and actionability scoring.

    Only BMI is true continuous in BRFSS 2021 (range 18.5-35.0); all other
    features are integer-encoded ordinals or binaries.
    """
    return [name for name in FEATURE_TAXONOMY if name != "BMI"]