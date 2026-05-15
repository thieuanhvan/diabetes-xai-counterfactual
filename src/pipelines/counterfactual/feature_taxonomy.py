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

v4 Part C extension (15/05/2026): added module-level flag
_CONDITIONAL_CLASS_DISABLED. When True, the CONDITIONAL class is collapsed to
MONOTONIC_DOWN — DiffWalk treated as monotonic_down for both
get_features_to_vary_for_query and get_permitted_range_for_query. Used by
Ablation 5 (5-class vs 4-class taxonomy) per ablation_vi v2 §7.

Setter: set_conditional_disabled(True/False). main.py calls this once at
startup based on cfg["taxonomy"]["conditional_class_disabled"]. Flag persists
for the duration of the run.
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


# ──────────────────────────────────────────────────────────────────────
# v4 Part C: conditional class disable flag for Ablation 5
# ──────────────────────────────────────────────────────────────────────
_CONDITIONAL_CLASS_DISABLED = False


def set_conditional_disabled(flag: bool) -> None:
    """Set whether CONDITIONAL class is collapsed to MONOTONIC_DOWN.

    When True (Ablation 5 baseline = 4-class taxonomy), DiffWalk is treated
    as MONOTONIC_DOWN — included in features_to_vary if patient.DiffWalk > 0,
    permitted_range restricted to [0, current].

    When False (default = 5-class taxonomy), DiffWalk remains CONDITIONAL —
    always excluded from features_to_vary regardless of patient context.

    Effect persists until set_conditional_disabled(False) or process exit.
    """
    global _CONDITIONAL_CLASS_DISABLED
    _CONDITIONAL_CLASS_DISABLED = bool(flag)


def is_conditional_disabled() -> bool:
    """Return current state of the conditional-class disable flag."""
    return _CONDITIONAL_CLASS_DISABLED


def _effective_mutability(spec: FeatureSpec) -> Mutability:
    """Return spec.mutability, with CONDITIONAL collapsed to MONOTONIC_DOWN
    when the global disable flag is set. Used by query helpers below.
    """
    if spec.mutability == Mutability.CONDITIONAL and _CONDITIONAL_CLASS_DISABLED:
        return Mutability.MONOTONIC_DOWN
    return spec.mutability


# ──────────────────────────────────────────────────────────────────────
# Public accessors (unchanged signatures)
# ──────────────────────────────────────────────────────────────────────
def get_actionable_features() -> List[str]:
    """All features except IMMUTABLE (and CONDITIONAL when flag is OFF)."""
    out = []
    for name, spec in FEATURE_TAXONOMY.items():
        eff = _effective_mutability(spec)
        if eff == Mutability.IMMUTABLE:
            continue
        if eff == Mutability.CONDITIONAL:
            continue
        out.append(name)
    return out


def get_immutable_features() -> List[str]:
    return [name for name, spec in FEATURE_TAXONOMY.items()
            if _effective_mutability(spec) == Mutability.IMMUTABLE]


def get_features_by_mutability(mutability: Mutability) -> List[str]:
    return [name for name, spec in FEATURE_TAXONOMY.items()
            if _effective_mutability(spec) == mutability]


def get_feature_ranges() -> Dict[str, Tuple[float, float]]:
    return {name: spec.value_range for name, spec in FEATURE_TAXONOMY.items()}


def get_continuous_features() -> List[str]:
    """Only BMI is treated as continuous; rest are discrete."""
    return ["BMI"]


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
    - IMMUTABLE / CONDITIONAL (5-class): always excluded
    - CONDITIONAL → MONOTONIC_DOWN (4-class via flag): treated like other
      monotonic_down features (DiffWalk eligible if patient.DiffWalk > 0)
    """
    features = []
    for name, spec in FEATURE_TAXONOMY.items():
        if name not in query.index:
            continue
        eff = _effective_mutability(spec)
        if eff in (Mutability.IMMUTABLE, Mutability.CONDITIONAL):
            continue
        current = float(query[name])
        lo, hi = spec.value_range
        if eff == Mutability.MONOTONIC_DOWN and current <= lo:
            continue
        if eff == Mutability.MONOTONIC_UP and current >= hi:
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

    When _CONDITIONAL_CLASS_DISABLED (flag set), DiffWalk is treated as
    MONOTONIC_DOWN — gets [0, current] range if patient.DiffWalk > 0.
    """
    ftv = set(get_features_to_vary_for_query(query))
    permitted: Dict[str, List[float]] = {}
    for name, spec in FEATURE_TAXONOMY.items():
        if name not in query.index:
            continue
        lo, hi = float(spec.value_range[0]), float(spec.value_range[1])
        eff = _effective_mutability(spec)
        if name in ftv:
            current = float(query[name])
            if eff == Mutability.MONOTONIC_DOWN:
                permitted[name] = [lo, current]
            elif eff == Mutability.MONOTONIC_UP:
                permitted[name] = [current, hi]
            else:
                permitted[name] = [lo, hi]
        else:
            permitted[name] = [lo, hi]
    return permitted
