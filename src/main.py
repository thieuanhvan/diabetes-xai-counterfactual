"""Pipeline entry point with §11.4 logging + dev-mode sampling."""
from __future__ import annotations

import argparse
import logging
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
from src.data.loader import TARGET_COL, load_dataset
from src.evaluate.cf_metrics import (
    diversity, plausibility, proximity_l1, sparsity, validity,
)
from src.models.xgb_train import XGBConfig, train_xgb
from src.preprocessing.pipeline import get_train_test_split
from src.utils.run_logger import setup_run, finalize_run
from src.utils.seed import seed_everything


log = logging.getLogger(__name__)


def main(config_path: str) -> None:
    config_path = Path(config_path).resolve()
    repo_root = config_path.parent.parent

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Dev-mode detection: sampling -> scratch (auto-gitignored per §11.4)
    sample_n = cfg.get("data", {}).get("sample_n", None)
    sample_seed = cfg.get("data", {}).get("sample_seed", 42)
    is_scratch_override = cfg.get("run", {}).get("is_scratch", None)
    is_scratch = is_scratch_override if is_scratch_override is not None else (sample_n is not None)

    run_ctx = setup_run(repo_root, is_scratch=is_scratch)
    seed_everything(cfg["random"]["seed"])

    # ---- 1. Load ----
    log.info("[1/4] Load dataset...")
    X, y = load_dataset(cfg["paths"]["data_csv"], sample_n=sample_n, sample_seed=sample_seed)
    if sample_n is not None:
        log.info(f"      [DEV MODE] sampled n={sample_n} (seed={sample_seed}) -> scratch run")
    log.info(f"      shape: X={X.shape}, y={y.shape}, prevalence={y.mean():.4f}")

    # ---- 2. Split + train ----
    log.info("[2/4] Train XGBoost...")
    X_train, X_test, y_train, y_test = get_train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        seed=cfg["random"]["seed"],
        stratify=cfg["split"]["stratify"],
    )
    xgb_cfg = XGBConfig(**cfg["xgboost"])
    result = train_xgb(X_train, y_train, X_test, y_test, xgb_cfg)
    log.info(f"      test AUC: {result['auc']:.4f}")
    log.info(f"      (P2 run 12 baseline on BRFSS 2021 full: AUC=0.8228)")

    # ---- 3. Generate CFs ----
    log.info("[3/4] Generate counterfactuals...")
    n_eval = min(cfg["evaluate"]["n_test_instances"], len(X_test))
    high_risk_idx = np.argsort(result["proba"])[-n_eval:]
    query_instances = X_test.iloc[high_risk_idx].reset_index(drop=True)

    dice_cfg = DiCEConfig(
        method=cfg["dice"]["method"],
        n_counterfactuals=cfg["dice"]["n_counterfactuals"],
        desired_class=cfg["dice"]["desired_class"],
        proximity_weight=cfg["dice"]["proximity_weight"],
        diversity_weight=cfg["dice"]["diversity_weight"],
        per_query=cfg["dice"].get("per_query", True),
    )
    log.info(f"      mode: per_query={dice_cfg.per_query}, method={dice_cfg.method}, n_queries={n_eval}")
    runner = DiCERunner(
        model=result["model"], X_train=X_train, y_train=y_train,
        target_col=TARGET_COL, config=dice_cfg,
    )
    cf_examples = runner.generate(query_instances)

    # ---- 4. Evaluate ----
    log.info("[4/4] Evaluate CFs...")
    feature_ranges = get_feature_ranges()
    feature_cols = list(X_train.columns)
    discrete_features = get_discrete_features()

    rows = []
    n_skipped = 0
    for i, cf_set in enumerate(cf_examples):
        if cf_set is None or cf_set.final_cfs_df is None or len(cf_set.final_cfs_df) == 0:
            n_skipped += 1
            continue
        cfs_df = cf_set.final_cfs_df[feature_cols].reset_index(drop=True)
        for col in discrete_features:
            if col in cfs_df.columns:
                cfs_df[col] = cfs_df[col].round().astype(int)

        query = query_instances.iloc[i]
        prox = proximity_l1(query, cfs_df, feature_ranges)
        spars = sparsity(query, cfs_df)
        div = diversity(cfs_df)
        plaus = plausibility(cfs_df, X_train, k=cfg["evaluate"]["plausibility_neighbors"])

        act_results = [actionability_score(query, cfs_df.iloc[j]) for j in range(len(cfs_df))]
        act_mean = float(np.mean([a["score"] for a in act_results]))
        wrong_dir_mean = float(np.mean([a["wrong_direction_violations"] for a in act_results]))
        immutable_mean = float(np.mean([a["immutable_violations"] for a in act_results]))

        cf_preds = result["model"].predict(cfs_df.values)
        val = validity(cf_preds, dice_cfg.desired_class)

        rows.append({
            "i": i, "n_cfs": len(cfs_df),
            "validity": val, "proximity": prox, "sparsity": spars,
            "diversity": div, "plausibility": plaus,
            "actionability": act_mean,
            "wrong_dir_violations": wrong_dir_mean,
            "immutable_violations": immutable_mean,
        })

    summary = pd.DataFrame(rows)
    output_dir = Path(cfg["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{run_ctx['run_id']}_cf_metrics.csv"
    summary.to_csv(out_csv, index=False)

    log.info(f"      Queries processed: {len(rows)}/{len(query_instances)} (skipped {n_skipped})")
    log.info("      Aggregate metrics:")
    for metric_name, metric_val in summary.drop(columns=["i", "n_cfs"]).mean().items():
        log.info(f"        {metric_name:<22s} {metric_val:.6f}")
    log.info(f"      Per-instance CSV: {out_csv}")

    # §11.4 finalize
    agg = summary.drop(columns=["i", "n_cfs"]).mean()
    notes = (
        f"per_query={dice_cfg.per_query}; method={dice_cfg.method}; "
        f"sample_n={sample_n}; n_queries={n_eval}; "
        f"validity={agg['validity']:.4f}; actionability={agg['actionability']:.4f}; "
        f"AUC={result['auc']:.4f}"
    )
    finalize_run(
        run_ctx,
        seeds={
            "global": cfg["random"]["seed"],
            "xgboost": xgb_cfg.random_state,
            "sample": sample_seed if sample_n is not None else None,
        },
        dataset={
            "name": "BRFSS_2021",
            "csv_path": str(cfg["paths"]["data_csv"]),
            "rows_used": int(X.shape[0]),
            "sample_n": sample_n,
            "features": int(X.shape[1]),
            "target": TARGET_COL,
            "prevalence": float(y.mean()),
        },
        hyperparameters={
            "split": cfg["split"],
            "xgboost": dict(xgb_cfg.__dict__),
            "dice": {
                "method": dice_cfg.method,
                "n_counterfactuals": dice_cfg.n_counterfactuals,
                "desired_class": dice_cfg.desired_class,
                "proximity_weight": dice_cfg.proximity_weight,
                "diversity_weight": dice_cfg.diversity_weight,
                "per_query": dice_cfg.per_query,
            },
            "evaluate": cfg["evaluate"],
        },
        notes=notes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)