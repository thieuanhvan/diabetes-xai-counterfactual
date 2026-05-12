"""DiCE-ML wrapper for XGBoost counterfactual generation.

Usage:
    runner = DiCERunner(model, X_train, y_train)
    cf_explanations = runner.generate(query_instances)

Notes on DiCE API quirks (dice-ml 0.11):
    1. proximity_weight / diversity_weight: ONLY supported by method="genetic".
    2. Categorical features stringified internally then written back to float64
       column -> TypeError. Workaround: pass ALL features as continuous; round
       integer-valued features post-hoc in main pipeline (see get_discrete_features).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from dice_ml import Data, Dice, Model

from src.counterfactual.feature_taxonomy import (
    get_actionable_features,
    get_feature_ranges,
)


@dataclass
class DiCEConfig:
    method: str = "random"           # "random" | "genetic" | "kdtree"
    n_counterfactuals: int = 5
    desired_class: int = 0            # flip to "no diabetes"
    proximity_weight: float = 0.5     # genetic only
    diversity_weight: float = 1.0     # genetic only
    features_to_vary: Optional[List[str]] = None  # None -> use actionable from taxonomy


class DiCERunner:
    """Wraps DiCE-ML for counterfactual generation on a trained XGBoost model."""

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_col: str = "Diabetes_binary",
        config: Optional[DiCEConfig] = None,
    ):
        self.config = config or DiCEConfig()
        self.target_col = target_col

        if self.config.features_to_vary is None:
            self.config.features_to_vary = get_actionable_features()

        # DiCE expects training data with target column
        train_df = X_train.copy()
        train_df[target_col] = y_train.values

        # Pass ALL features as continuous to bypass dice-ml 0.11 categorical
        # stringify TypeError. Integer-valued features round post-hoc in main.
        self.dice_data = Data(
            dataframe=train_df,
            continuous_features=list(X_train.columns),
            outcome_name=target_col,
        )
        self.dice_model = Model(model=model, backend="sklearn", model_type="classifier")
        self.dice = Dice(self.dice_data, self.dice_model, method=self.config.method)

    def generate(
        self,
        query_instances: pd.DataFrame,
        n_cfs: Optional[int] = None,
        desired_class: Optional[int] = None,
    ):
        """Generate CFs for a batch of query instances."""
        n = n_cfs if n_cfs is not None else self.config.n_counterfactuals
        cls = desired_class if desired_class is not None else self.config.desired_class

        permitted_range = {
            name: [float(lo), float(hi)] for name, (lo, hi) in get_feature_ranges().items()
        }

        kwargs = {
            "total_CFs": n,
            "desired_class": cls,
            "features_to_vary": self.config.features_to_vary,
            "permitted_range": permitted_range,
        }
        if self.config.method == "genetic":
            kwargs["proximity_weight"] = self.config.proximity_weight
            kwargs["diversity_weight"] = self.config.diversity_weight

        return self.dice.generate_counterfactuals(query_instances, **kwargs)