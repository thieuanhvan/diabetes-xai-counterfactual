"""Actionability taxonomy for BRFSS 2021 features.

Defines mutability + range + semantic label per feature. Used by:
- DiCE-ML to constrain CF generation (features_to_vary, permitted_range)
- Actionability metric to score CFs post-hoc
- Per-query filters (drop features at semantic extremes; restrict direction)

Mutability "intervention-direction" lens (v1, 12/05/2026):
- IMMUTABLE: demographic / biological / irreversible history
- MONOTONIC_UP: positive behaviors (CF may only INCREASE)
- MONOTONIC_DOWN: negative behaviors/states (CF may only DECREASE)
- BIDIRECTIONAL: both extremes unhealthy (e.g. BMI)
- CONDITIONAL: caused by co-morbidity, not directly actionable

Total: 21 features = 4 + 7 + 8 + 1 + 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import pandas as pd


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
    value_range: Tuple[float, float]
    semantic_label: str


FEATURE_TAXONOMY: Dict[str, FeatureSpec] = {
    # Immutable (4)
    "Age": FeatureSpec("Age", Mutability.IMMUTABLE, (1, 13), "Age category (1=18-24, 13=80+)"),
    "Sex": FeatureSpec("Sex", Mutability.IMMUTABLE, (0, 1), "Sex (0=female, 1=male)"),
    "Stroke": FeatureSpec("Stroke", Mutability.IMMUTABLE, (0, 1), "Ever had a stroke (irreversible history)"),
    "HeartDiseaseorAttack": FeatureSpec("HeartDiseaseorAttack", Mutability.IMMUTABLE, (0, 1), "Had CHD or MI (irreversible history)"),

    # Monotonic up (7)
    "PhysActivity": FeatureSpec("PhysActivity", Mutability.MONOTONIC_UP, (0, 1), "Physical activity in last 30 days (0->1)"),
    "Fruits": FeatureSpec("Fruits", Mutability.MONOTONIC_UP, (0, 1), "Consume fruit >=1/day (0->1)"),
    "Veggies": FeatureSpec("Veggies", Mutability.MONOTONIC_UP, (0, 1), "Consume vegetables >=1/day (0->1)"),
    "AnyHealthcare": FeatureSpec("AnyHealthcare", Mutability.MONOTONIC_UP, (0, 1), "Has healthcare coverage (0->1)"),
    "CholCheck": FeatureSpec("CholCheck", Mutability.MONOTONIC_UP, (0, 1), "Cholesterol check in last 5 years (0->1)"),
    "Education": FeatureSpec("Education", Mutability.MONOTONIC_UP, (1, 6), "Education level (1=none, 6=college graduate)"),
    "Income": FeatureSpec("Income", Mutability.MONOTONIC_UP, (1, 11), "Income category 2021 (1=<$10K, 11=>=$200K)"),

    # Monotonic down (8)
    "Smoker": FeatureSpec("Smoker", Mutability.MONOTONIC_DOWN, (0, 1), "Smoked >=100 cigarettes lifetime (1->0 = quit)"),
    "HvyAlcoholConsump": FeatureSpec("HvyAlcoholConsump", Mutability.MONOTONIC_DOWN, (0, 1), "Heavy alcohol consumption (1->0)"),
    "NoDocbcCost": FeatureSpec("NoDocbcCost", Mutability.MONOTONIC_DOWN, (0, 1), "Could not see doctor due to cost (1->0)"),
    "HighBP": FeatureSpec("HighBP", Mutability.MONOTONIC_DOWN, (0, 1), "High blood pressure (1->0 = controlled)"),
    "HighChol": FeatureSpec("HighChol", Mutability.MONOTONIC_DOWN, (0, 1), "High cholesterol (1->0 = controlled)"),
    "GenHlth": FeatureSpec("GenHlth", Mutability.MONOTONIC_DOWN, (1, 5), "General health (1=excellent, lower=better)"),
    "MentHlth": FeatureSpec("MentHlth", Mutability.MONOTONIC_DOWN, (0, 30), "Mental health bad days in last 30 (lower=better)"),
    "PhysHlth": FeatureSpec("PhysHlth", Mutability.MONOTONIC_DOWN, (0, 30), "Physical health bad days in last 30 (lower=better)"),

    # Bidirectional (1)
    "BMI": FeatureSpec("BMI", Mutability.BIDIRECTIONAL, (18.5, 35.0), "Body Mass Index (clinical healthy range)"),

    # Conditional (1)
    "DiffWalk": FeatureSpec("DiffWalk", Mutability.CONDITIONAL, (0, 1), "Serious difficulty walking (depends on co-morbidity)"),
}


def get_actionable_features() -> List[str]:
    """GLOBAL list: everything except IMMUTABLE + CONDITIONAL. Used in per_query=False mode."""
    return [
        name for name, spec in FEATURE_TAXONOMY.items()
        if spec.mutability not in (Mutability.IMMUTABLE, Mutability.CONDITIONAL)
    ]


def get_immutable_features() -> List[str]:
    return [name for name, spec in FEATURE_TAXONOMY.items() if spec.mutability == Mutability.IMMUTABLE]


def get_features_by_mutability(mutability: Mutability) -> List[str]:
    return [name for name, spec in FEATURE_TAXONOMY.items() if spec.mutability == mutability]


def get_feature_ranges() -> Dict[str, Tuple[float, float]]:
    return {name: spec.value_range for name, spec in FEATURE_TAXONOMY.items()}


def get_continuous_features() -> List[str]:
    return [name for name, spec in FEATURE_TAXONOMY.items() if (spec.value_range[1] - spec.value_range[0]) > 2]


def get_discrete_features() -> List[str]:
    """Features rounded to int post-DiCE. Only BMI is true continuous."""
    return [name for name in FEATURE_TAXONOMY if name != "BMI"]


def get_features_to_vary_for_query(query: pd.Series) -> List[str]:
    """Per-query features_to_vary: exclude features already at monotonic extreme.

    Examples:
    - Smoker (MONOTONIC_DOWN): excluded if query.Smoker == 0
    - PhysActivity (MONOTONIC_UP): excluded if query.PhysActivity == 1
    - Education (MONOTONIC_UP): excluded if query.Education == 6
    - GenHlth (MONOTONIC_DOWN): excluded if query.GenHlth == 1
    - BMI (BIDIRECTIONAL): always included
    - IMMUTABLE / CONDITIONAL: always excluded
    """
    features = []
    for name, spec in FEATURE_TAXONOMY.items():
        if name not in query.index:
            continue
        if spec.mutability in (Mutability.IMMUTABLE, Mutability.CONDITIONAL):
            continue
        current = float(query[name])
        lo, hi = spec.value_range
        if spec.mutability == Mutability.MONOTONIC_DOWN and current <= lo:
            continue
        if spec.mutability == Mutability.MONOTONIC_UP and current >= hi:
            continue
        features.append(name)
    return features


def get_permitted_range_for_query(query: pd.Series) -> Dict[str, List[float]]:
    """Per-query permitted_range that restricts CF values to monotonic-correct direction.

    For features IN get_features_to_vary_for_query(query):
    - MONOTONIC_DOWN: [feature_min, current] — CF may only decrease or stay
    - MONOTONIC_UP: [current, feature_max] — CF may only increase or stay
    - BIDIRECTIONAL: [feature_min, feature_max]

    For other features (immutable, conditional, or at-extreme): full feature range.
    DiCE won't perturb them since they're not in features_to_vary, but the range
    key must exist to satisfy DiCE's API expectations.
    """
    ftv = set(get_features_to_vary_for_query(query))
    permitted: Dict[str, List[float]] = {}
    for name, spec in FEATURE_TAXONOMY.items():
        if name not in query.index:
            continue
        lo, hi = float(spec.value_range[0]), float(spec.value_range[1])
        if name in ftv:
            current = float(query[name])
            if spec.mutability == Mutability.MONOTONIC_DOWN:
                permitted[name] = [lo, current]
            elif spec.mutability == Mutability.MONOTONIC_UP:
                permitted[name] = [current, hi]
            else:
                permitted[name] = [lo, hi]
        else:
            permitted[name] = [lo, hi]
    return permitted