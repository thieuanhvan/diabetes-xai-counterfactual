"""Train XGBoost on BRFSS 2021.

Hyperparameters set from community defaults for the BRFSS 2021 cohort;
no tuning is performed so that downstream actionability differences cannot
be attributed to classifier-tuning bias. Single XGBoost classifier is
sufficient for the per-query CF actionability analysis.

The `extended_metrics()` helper computes precision, recall, specificity,
F1, balanced accuracy, MCC alongside AUC. Backward compatible — train_xgb()
return dict gets an 'extended_metrics' key that downstream consumers can
ignore if not needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix,
)
from xgboost import XGBClassifier


@dataclass
class XGBConfig:
    """Hyperparameters from community defaults."""
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


def extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute 8 classifier metrics + confusion matrix counts at given threshold.

    Reports threshold-free AUC alongside threshold-dependent metrics.
    Returns a JSON-serializable dict (all floats/ints).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "n_test": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "AUC_ROC": float(roc_auc_score(y_true, y_proba)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall_Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "threshold": float(threshold),
    }


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
        'predictions' (np.ndarray), 'proba' (np.ndarray),
        'extended_metrics' (dict of calibration + threshold metrics).
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
        "extended_metrics": extended_metrics(y_test.values, preds, proba, threshold=0.5),
    }
