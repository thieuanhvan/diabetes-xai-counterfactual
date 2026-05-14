"""Actionability scoring for counterfactuals.

Quantifies how 'actionable' a CF is given the feature taxonomy:
- Penalize changes in IMMUTABLE features (should be 0 — DiCE prevents this,
  but verify post-hoc).
- Penalize wrong-direction changes in monotonic features.
- Score = actionable_changes / total_changes, in [0, 1].
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from src.pipelines.counterfactual.feature_taxonomy import (
    FEATURE_TAXONOMY,
    Mutability,
)


def actionability_score(
    query: pd.Series,
    cf: pd.Series,
) -> Dict[str, float]:
    """Score a single CF against actionability constraints.

    Returns:
        {
            'immutable_violations': int,
            'wrong_direction_violations': int,
            'actionable_changes': int,
            'total_changes': int,
            'score': float in [0, 1] (1 = perfectly actionable)
        }
    """
    immutable_v = 0
    wrong_dir_v = 0
    actionable_c = 0
    total_changes = 0

    for feature, spec in FEATURE_TAXONOMY.items():
        if feature not in query.index or feature not in cf.index:
            continue
        delta = float(cf[feature]) - float(query[feature])
        if delta == 0:
            continue
        total_changes += 1

        if spec.mutability == Mutability.IMMUTABLE:
            immutable_v += 1
        elif spec.mutability == Mutability.CONDITIONAL:
            # CF shouldn't act on conditional features directly
            wrong_dir_v += 1
        elif spec.mutability == Mutability.MONOTONIC_UP and delta < 0:
            wrong_dir_v += 1
        elif spec.mutability == Mutability.MONOTONIC_DOWN and delta > 0:
            wrong_dir_v += 1
        else:
            actionable_c += 1

    if total_changes == 0:
        score = 0.0  # CF identical to query -> not useful
    else:
        score = actionable_c / total_changes

    return {
        "immutable_violations": int(immutable_v),
        "wrong_direction_violations": int(wrong_dir_v),
        "actionable_changes": int(actionable_c),
        "total_changes": int(total_changes),
        "score": float(score),
    }
