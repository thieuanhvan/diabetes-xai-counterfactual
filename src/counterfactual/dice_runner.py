"""DiCE-ML wrapper for XGBoost counterfactual generation.

Modes:
- per_query=False (batch): single DiCE call with GLOBAL features_to_vary.
  Faster setup, but allows wrong-direction CFs.
- per_query=True (recommended): iterate queries; each call uses query-specific
  features_to_vary + permitted_range from feature_taxonomy. Same wall-clock
  (~1.3s/query). Produces ethically constrained CFs.

DiCE-ml 0.11 quirks worked around:
    1. proximity/diversity weights only for method="genetic".
    2. Categorical stringify TypeError: pass all features as continuous,
       round integer features post-hoc in main.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from dice_ml import Data, Dice, Model
from tqdm import tqdm

from src.counterfactual.feature_taxonomy import (
    get_actionable_features,
    get_feature_ranges,
    get_features_to_vary_for_query,
    get_permitted_range_for_query,
)


@dataclass
class DiCEConfig:
    method: str = "random"
    n_counterfactuals: int = 5
    desired_class: int = 0
    proximity_weight: float = 0.5
    diversity_weight: float = 1.0
    features_to_vary: Optional[List[str]] = None
    per_query: bool = True


class DiCERunner:
    """Returns a flat list of cf_example objects (one per query), with
    `.final_cfs_df` preserved. None entries indicate queries with no varyable
    features or DiCE failure."""

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
        self.feature_cols = list(X_train.columns)

        if self.config.features_to_vary is None:
            self.config.features_to_vary = get_actionable_features()

        train_df = X_train.copy()
        train_df[target_col] = y_train.values

        self.dice_data = Data(
            dataframe=train_df,
            continuous_features=self.feature_cols,
            outcome_name=target_col,
        )
        self.dice_model = Model(model=model, backend="sklearn", model_type="classifier")
        self.dice = Dice(self.dice_data, self.dice_model, method=self.config.method)

    def _build_kwargs(self, n: int, cls: int, ftv: List[str], permitted_range: dict) -> dict:
        kwargs = {
            "total_CFs": n,
            "desired_class": cls,
            "features_to_vary": ftv,
            "permitted_range": permitted_range,
        }
        if self.config.method == "genetic":
            kwargs["proximity_weight"] = self.config.proximity_weight
            kwargs["diversity_weight"] = self.config.diversity_weight
        return kwargs

    def generate(
        self,
        query_instances: pd.DataFrame,
        n_cfs: Optional[int] = None,
        desired_class: Optional[int] = None,
    ) -> List:
        n = n_cfs if n_cfs is not None else self.config.n_counterfactuals
        cls = desired_class if desired_class is not None else self.config.desired_class

        if not self.config.per_query:
            # Batch mode (legacy)
            permitted = {
                name: [float(lo), float(hi)]
                for name, (lo, hi) in get_feature_ranges().items()
            }
            kwargs = self._build_kwargs(n, cls, self.config.features_to_vary, permitted)
            result = self.dice.generate_counterfactuals(query_instances, **kwargs)
            return list(result.cf_examples_list)

        # Per-query mode
        cf_examples: List = []
        for i in tqdm(range(len(query_instances)), desc="Per-query CF"):
            query_row = query_instances.iloc[[i]]
            ftv = get_features_to_vary_for_query(query_row.iloc[0])
            if not ftv:
                cf_examples.append(None)
                continue
            permitted = get_permitted_range_for_query(query_row.iloc[0])
            kwargs = self._build_kwargs(n, cls, ftv, permitted)
            try:
                result = self.dice.generate_counterfactuals(query_row, **kwargs)
                cf_examples.extend(result.cf_examples_list)
            except Exception as e:
                print(f"[warn] query {i}: {type(e).__name__}: {e}")
                cf_examples.append(None)
        return cf_examples