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
    _effective_mutability,
)


def actionability_score(
    query: pd.Series,
    cf: pd.Series,
) -> Dict[str, float]:
    """Score a single CF against actionability constraints.

    Uses the effective mutability of each feature (after ablation-flag
    collapses applied by _effective_mutability) so the score remains
    consistent with the constraint set DiCE was given. Without this,
    the 4-class collapsed variant would score DiffWalk changes as
    CONDITIONAL violations even though the taxonomy treats them as
    MONOTONIC_DOWN; the conservative SE-proxy variant would similarly
    score Income/Education/AnyHealthcare changes against the wrong
    mutability class.

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
        eff = _effective_mutability(spec)

        if eff == Mutability.IMMUTABLE:
            immutable_v += 1
        elif eff == Mutability.CONDITIONAL:
            # CF shouldn't act on conditional features directly
            wrong_dir_v += 1
        elif eff == Mutability.MONOTONIC_UP and delta < 0:
            wrong_dir_v += 1
        elif eff == Mutability.MONOTONIC_DOWN and delta > 0:
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
