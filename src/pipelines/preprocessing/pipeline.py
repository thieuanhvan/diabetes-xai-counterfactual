"""Preprocessing for BRFSS 2021.

Reuse from P2: stratified 80/20 split, random_state=42.

Design note for P4: counterfactual generation works in the ORIGINAL encoded
space (BRFSS uses integer codes for categorical features) so that CFs are
interpretable as "change BMI from 28 to 25" rather than "change scaled BMI
from 0.3 to -0.2". XGBoost is scale-invariant, so we skip StandardScaler.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split, raw-encoded features (no scaling)."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if stratify else None,
        random_state=seed,
    )
