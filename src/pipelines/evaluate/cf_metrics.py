"""Counterfactual evaluation metrics.

Standard DiCE metrics: validity, proximity, sparsity, diversity.
Plus plausibility (k-NN distance to training distribution).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def validity(cf_predictions: np.ndarray, desired_class: int) -> float:
    """Fraction of CFs achieving the desired predicted class."""
    if len(cf_predictions) == 0:
        return 0.0
    return float((cf_predictions == desired_class).mean())


def proximity_l1(
    query: pd.Series,
    cfs: pd.DataFrame,
    feature_ranges: Dict[str, Tuple[float, float]] | None = None,
) -> float:
    """Mean L1 distance between query and each CF, optionally normalized by range.

    Lower = closer to original query.
    """
    deltas = (cfs - query).abs()
    if feature_ranges:
        for f, (lo, hi) in feature_ranges.items():
            if f in deltas.columns and hi > lo:
                deltas[f] = deltas[f] / (hi - lo)
    return float(deltas.sum(axis=1).mean())


def sparsity(query: pd.Series, cfs: pd.DataFrame) -> float:
    """Mean fraction of features changed per CF.

    Lower = fewer features need to change.
    """
    changed = (cfs != query).sum(axis=1)
    return float(changed.mean() / cfs.shape[1])


def diversity(cfs: pd.DataFrame) -> float:
    """Mean pairwise L1 distance between CFs (DiCE-style diversity).

    Higher = CFs are more different from each other.
    """
    n = len(cfs)
    if n < 2:
        return 0.0
    values = cfs.values
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(np.abs(values[i] - values[j]).sum())
    return float(np.mean(diffs))


def plausibility(
    cfs: pd.DataFrame,
    X_train: pd.DataFrame,
    k: int = 50,
) -> float:
    """Plausibility = mean distance to k-nearest training neighbors.

    Lower = more plausible (closer to training distribution).
    """
    if len(cfs) == 0:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_train.values)
    distances, _ = nn.kneighbors(cfs.values)
    return float(distances.mean())
