"""Train XGBoost on BRFSS 2021.

Hyperparameters baseline from P2 run 12 (AUC=0.8228 on BRFSS 2021).
P4 uses single XGBoost (P2 used 3 models LR/RF/XGB).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


@dataclass
class XGBConfig:
    """Hyperparameters reused from P2 run 12."""
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: int = 1
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    eval_metric: str = "auc"
    random_state: int = 42
    n_jobs: int = -1


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: XGBConfig | None = None,
) -> Dict[str, Any]:
    """Train XGBoost and return model + test metrics.

    Returns:
        dict with keys: 'model' (XGBClassifier), 'auc' (float),
        'predictions' (np.ndarray), 'proba' (np.ndarray).
    """
    cfg = config or XGBConfig()
    model = XGBClassifier(**cfg.__dict__)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    preds = (proba >= 0.5).astype(int)

    return {
        "model": model,
        "auc": float(auc),
        "predictions": preds,
        "proba": proba,
    }
