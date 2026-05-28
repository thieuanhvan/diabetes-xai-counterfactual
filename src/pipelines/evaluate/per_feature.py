"""Per-feature actionability breakdown for CF analysis.

For each feature in the taxonomy, tally across all (query, CF) pairs:
- How often was this feature changed?
- Of those changes, how many were in the "correct" direction per taxonomy?
- How many were wrong-direction or immutable violations?

This complements aggregate actionability_score by exposing WHICH features
contribute most to (lack of) actionability.

Used by main.py to build the per-feature breakdown table.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.pipelines.counterfactual.feature_taxonomy import (
    FEATURE_TAXONOMY,
    Mutability,
    _effective_mutability,
)


def per_feature_breakdown_one_query(
    query: pd.Series,
    cfs_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Per-feature stats for a single (query, set-of-CFs) pair.

    Uses the effective mutability of each feature (after ablation-flag
    collapses) so the per-feature scoring stays consistent with the
    aggregate actionability_score under 4-class and conservative SE-proxy
    variants.

    Returns: {feature: {n_cf_changed, n_actionable, n_wrong_dir, n_immutable, sum_delta, mean_delta_signed}}
    """
    out: Dict[str, Dict[str, float]] = {}
    for feature, spec in FEATURE_TAXONOMY.items():
        if feature not in query.index or feature not in cfs_df.columns:
            continue
        n_cf_changed = 0
        n_actionable = 0
        n_wrong_dir = 0
        n_immutable = 0
        sum_delta = 0.0
        q_val = float(query[feature])
        eff = _effective_mutability(spec)
        for _, cf_row in cfs_df.iterrows():
            delta = float(cf_row[feature]) - q_val
            if delta == 0:
                continue
            n_cf_changed += 1
            sum_delta += delta
            if eff == Mutability.IMMUTABLE:
                n_immutable += 1
            elif eff == Mutability.CONDITIONAL:
                n_wrong_dir += 1
            elif eff == Mutability.MONOTONIC_UP and delta < 0:
                n_wrong_dir += 1
            elif eff == Mutability.MONOTONIC_DOWN and delta > 0:
                n_wrong_dir += 1
            else:
                n_actionable += 1
        out[feature] = {
            "n_cf_changed": n_cf_changed,
            "n_actionable": n_actionable,
            "n_wrong_dir": n_wrong_dir,
            "n_immutable": n_immutable,
            "sum_delta": sum_delta,
            "mean_delta_signed": (sum_delta / n_cf_changed) if n_cf_changed > 0 else 0.0,
        }
    return out


def aggregate_per_feature(
    per_query_results: List[Dict[str, Dict[str, float]]],
    mode_label: str,
) -> pd.DataFrame:
    """Aggregate per-query per-feature stats into a single DataFrame.

    Args:
        per_query_results: list (one per query) of per-feature dicts
        mode_label: tag this aggregation as e.g. "per_query" or "global"

    Returns DataFrame columns:
        feature, taxonomy_class, mode, n_queries_with_change, n_total_cf_changes,
        n_actionable, n_wrong_dir, n_immutable, mean_delta_signed,
        actionability_rate, violation_rate
    """
    feature_totals: Dict[str, Dict[str, float]] = {}
    for feature in FEATURE_TAXONOMY:
        feature_totals[feature] = {
            "n_queries_with_change": 0,
            "n_total_cf_changes": 0,
            "n_actionable": 0,
            "n_wrong_dir": 0,
            "n_immutable": 0,
            "sum_delta": 0.0,
        }

    for q_result in per_query_results:
        if q_result is None:
            continue
        for feature, stats in q_result.items():
            if stats["n_cf_changed"] > 0:
                feature_totals[feature]["n_queries_with_change"] += 1
            feature_totals[feature]["n_total_cf_changes"] += stats["n_cf_changed"]
            feature_totals[feature]["n_actionable"] += stats["n_actionable"]
            feature_totals[feature]["n_wrong_dir"] += stats["n_wrong_dir"]
            feature_totals[feature]["n_immutable"] += stats["n_immutable"]
            feature_totals[feature]["sum_delta"] += stats["sum_delta"]

    rows = []
    for feature, spec in FEATURE_TAXONOMY.items():
        t = feature_totals[feature]
        n_changes = t["n_total_cf_changes"]
        rows.append({
            "feature": feature,
            "taxonomy_class": _effective_mutability(spec).value,
            "mode": mode_label,
            "n_queries_with_change": int(t["n_queries_with_change"]),
            "n_total_cf_changes": int(n_changes),
            "n_actionable": int(t["n_actionable"]),
            "n_wrong_dir": int(t["n_wrong_dir"]),
            "n_immutable": int(t["n_immutable"]),
            "mean_delta_signed": (t["sum_delta"] / n_changes) if n_changes > 0 else 0.0,
            "actionability_rate": (t["n_actionable"] / n_changes) if n_changes > 0 else 0.0,
            "violation_rate": ((t["n_wrong_dir"] + t["n_immutable"]) / n_changes) if n_changes > 0 else 0.0,
        })
    return pd.DataFrame(rows)
