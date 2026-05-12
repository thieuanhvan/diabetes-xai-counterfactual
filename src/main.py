"""Pipeline entry point: load -> train -> generate CFs -> evaluate.

Run:
    python -m src.main --config configs/default.yaml
    # or right-click run_main.py in PyCharm (REPO ROOT)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.counterfactual.actionability import actionability_score
from src.counterfactual.dice_runner import DiCEConfig, DiCERunner
from src.counterfactual.feature_taxonomy import (
    get_discrete_features,
    get_feature_ranges,
)
from src.data.load_brfss import TARGET_COL, load_brfss_2021
from src.evaluate.cf_metrics import (
    diversity,
    plausibility,
    proximity_l1,
    sparsity,
    validity,
)
from src.models.xgb_train import XGBConfig, train_xgb
from src.preprocessing.pipeline import get_train_test_split
from src.utils.seed import seed_everything


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["random"]["seed"])

    # ---- 1. Load ----
    print("[1/4] Load BRFSS 2021...")
    X, y = load_brfss_2021(cfg["paths"]["data_csv"])
    print(f"      shape: X={X.shape}, y={y.shape}, prevalence={y.mean():.4f}")

    # ---- 2. Split + train ----
    print("[2/4] Train XGBoost...")
    X_train, X_test, y_train, y_test = get_train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        seed=cfg["random"]["seed"],
        stratify=cfg["split"]["stratify"],
    )
    xgb_cfg = XGBConfig(**cfg["xgboost"])
    result = train_xgb(X_train, y_train, X_test, y_test, xgb_cfg)
    print(f"      test AUC: {result['auc']:.4f}")
    print(f"      (P2 run 12 baseline on BRFSS 2021: AUC=0.8228 -- should match)")

    # ---- 3. Generate CFs for highest-risk patients ----
    print("[3/4] Generate counterfactuals...")
    n_eval = cfg["evaluate"]["n_test_instances"]
    high_risk_idx = np.argsort(result["proba"])[-n_eval:]
    query_instances = X_test.iloc[high_risk_idx].reset_index(drop=True)

    dice_cfg = DiCEConfig(
        method=cfg["dice"]["method"],
        n_counterfactuals=cfg["dice"]["n_counterfactuals"],
        desired_class=cfg["dice"]["desired_class"],
        proximity_weight=cfg["dice"]["proximity_weight"],
        diversity_weight=cfg["dice"]["diversity_weight"],
    )
    runner = DiCERunner(
        model=result["model"],
        X_train=X_train,
        y_train=y_train,
        target_col=TARGET_COL,
        config=dice_cfg,
    )
    cf_explanations = runner.generate(query_instances)

    # ---- 4. Evaluate ----
    print("[4/4] Evaluate CFs...")
    feature_ranges = get_feature_ranges()
    feature_cols = list(X_train.columns)
    discrete_features = get_discrete_features()  # all except BMI

    rows = []
    for i, cf_set in enumerate(cf_explanations.cf_examples_list):
        if cf_set.final_cfs_df is None or len(cf_set.final_cfs_df) == 0:
            continue
        cfs_df = cf_set.final_cfs_df[feature_cols].reset_index(drop=True)

        # Round integer-valued features (DiCE samples floats for all features
        # because we passed continuous_features=all_columns in dice_runner)
        for col in discrete_features:
            if col in cfs_df.columns:
                cfs_df[col] = cfs_df[col].round().astype(int)

        query = query_instances.iloc[i]
        prox = proximity_l1(query, cfs_df, feature_ranges)
        spars = sparsity(query, cfs_df)
        div = diversity(cfs_df)
        plaus = plausibility(cfs_df, X_train, k=cfg["evaluate"]["plausibility_neighbors"])

        act_scores = [
            actionability_score(query, cfs_df.iloc[j])["score"]
            for j in range(len(cfs_df))
        ]
        act_mean = float(np.mean(act_scores)) if act_scores else 0.0

        cf_preds = result["model"].predict(cfs_df.values)
        val = validity(cf_preds, dice_cfg.desired_class)

        rows.append({
            "i": i,
            "n_cfs": len(cfs_df),
            "validity": val,
            "proximity": prox,
            "sparsity": spars,
            "diversity": div,
            "plausibility": plaus,
            "actionability": act_mean,
        })

    summary = pd.DataFrame(rows)
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "cf_metrics_summary.csv"
    summary.to_csv(out_csv, index=False)

    print("      Aggregate metrics (mean across query instances):")
    print(summary.drop(columns=["i", "n_cfs"]).mean().to_string())
    print(f"      Saved per-instance CSV: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)