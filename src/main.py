"""Pipeline entry point with §11.4/5/6 logging + compare_modes orchestration.

Modes:
- compare_modes=false (default): single CF eval per cfg.dice.per_query setting.
- compare_modes=true: run BOTH global and per-query back-to-back, save separate
  per-instance CSVs + comparison.csv + per_feature.csv.

Outputs (compare mode):
- outputs/{run_id}_perquery_cf_metrics.csv
- outputs/{run_id}_global_cf_metrics.csv
- outputs/{run_id}_comparison.csv          (aggregate metrics + deltas)
- outputs/{run_id}_per_feature.csv         (per-feature breakdown, both modes)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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
from src.evaluate.per_feature import (
    aggregate_per_feature,
    per_feature_breakdown_one_query,
)
from src.models.xgb_train import XGBConfig, train_xgb
from src.preprocessing.pipeline import get_train_test_split
from src.utils.run_logger import setup_run, finalize_run
from src.utils.seed import seed_everything


log = logging.getLogger(__name__)


def _run_one_eval(
    *,
    mode_label: str,
    per_query_flag: bool,
    cfg: dict,
    model_result: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    query_instances: pd.DataFrame,
    n_eval: int,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Run a single mode of CF generation + eval.

    Returns:
        (per_instance_summary_df, per_query_per_feature_breakdowns)
    """
    log.info(f"  [{mode_label}] generating CFs (per_query={per_query_flag})...")
    dice_cfg = DiCEConfig(
        method=cfg["dice"]["method"],
        n_counterfactuals=cfg["dice"]["n_counterfactuals"],
        desired_class=cfg["dice"]["desired_class"],
        proximity_weight=cfg["dice"]["proximity_weight"],
        diversity_weight=cfg["dice"]["diversity_weight"],
        per_query=per_query_flag,
    )
    runner = DiCERunner(
        model=model_result["model"],
        X_train=X_train, y_train=y_train,
        target_col=TARGET_COL, config=dice_cfg,
    )
    cf_examples = runner.generate(query_instances)

    log.info(f"  [{mode_label}] scoring CFs...")
    feature_ranges = get_feature_ranges()
    feature_cols = list(X_train.columns)
    discrete_features = get_discrete_features()

    rows = []
    per_q_breakdowns: List[Dict] = []
    n_skipped = 0

    for i, cf_set in enumerate(cf_examples):
        if cf_set is None or cf_set.final_cfs_df is None or len(cf_set.final_cfs_df) == 0:
            n_skipped += 1
            per_q_breakdowns.append(None)
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

        # NEW: per-feature breakdown
        per_q_breakdowns.append(per_feature_breakdown_one_query(query, cfs_df))

        cf_preds = model_result["model"].predict(cfs_df.values)
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
    log.info(f"  [{mode_label}] processed {len(rows)}/{len(query_instances)} (skipped {n_skipped})")
    return summary, per_q_breakdowns


def main(config_path: str) -> None:
    config_path = Path(config_path).resolve()
    repo_root = config_path.parent.parent

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Dev-mode detection
    sample_n = cfg.get("data", {}).get("sample_n", None)
    sample_seed = cfg.get("data", {}).get("sample_seed", 42)
    is_scratch_override = cfg.get("run", {}).get("is_scratch", None)
    is_scratch = is_scratch_override if is_scratch_override is not None else (sample_n is not None)

    # NEW: compare_modes flag
    compare_modes = cfg.get("evaluate", {}).get("compare_modes", False)

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

    # ---- 3. Pick high-risk queries (shared across modes) ----
    log.info(f"[3/4] CF generation (compare_modes={compare_modes})...")
    n_eval = min(cfg["evaluate"]["n_test_instances"], len(X_test))
    high_risk_idx = np.argsort(result["proba"])[-n_eval:]
    query_instances = X_test.iloc[high_risk_idx].reset_index(drop=True)
    log.info(f"      n_queries (high-risk): {n_eval}")

    output_dir = Path(cfg["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 4. Run one or two modes ----
    if not compare_modes:
        # Single-mode (legacy) behavior
        per_query_flag = cfg["dice"].get("per_query", True)
        mode_label = "per_query" if per_query_flag else "global"
        summary, breakdowns = _run_one_eval(
            mode_label=mode_label, per_query_flag=per_query_flag,
            cfg=cfg, model_result=result,
            X_train=X_train, y_train=y_train, X_test=X_test,
            query_instances=query_instances, n_eval=n_eval,
        )
        out_csv = output_dir / f"{run_ctx['run_id']}_cf_metrics.csv"
        summary.to_csv(out_csv, index=False)
        log.info(f"      Per-instance CSV: {out_csv}")
        log.info("      Aggregate metrics:")
        agg = summary.drop(columns=["i", "n_cfs"]).mean()
        for k, v in agg.items():
            log.info(f"        {k:<22s} {v:.6f}")

        # NEW: per-feature breakdown for the single mode
        per_feat_df = aggregate_per_feature(breakdowns, mode_label=mode_label)
        per_feat_csv = output_dir / f"{run_ctx['run_id']}_per_feature.csv"
        per_feat_df.to_csv(per_feat_csv, index=False)
        log.info(f"      Per-feature CSV: {per_feat_csv}")

        notes_str = (
            f"compare_modes=False; mode={mode_label}; method={cfg['dice']['method']}; "
            f"sample_n={sample_n}; n_queries={n_eval}; "
            f"validity={agg['validity']:.4f}; actionability={agg['actionability']:.4f}; "
            f"AUC={result['auc']:.4f}"
        )
    else:
        # Compare both modes back-to-back
        log.info("[4a/4] Mode 1: GLOBAL (features_to_vary='all', no per-query constraints)")
        summary_g, breakdowns_g = _run_one_eval(
            mode_label="global", per_query_flag=False,
            cfg=cfg, model_result=result,
            X_train=X_train, y_train=y_train, X_test=X_test,
            query_instances=query_instances, n_eval=n_eval,
        )
        out_g = output_dir / f"{run_ctx['run_id']}_global_cf_metrics.csv"
        summary_g.to_csv(out_g, index=False)
        log.info(f"      Global CSV: {out_g}")

        log.info("[4b/4] Mode 2: PER-QUERY (taxonomy-constrained)")
        summary_pq, breakdowns_pq = _run_one_eval(
            mode_label="per_query", per_query_flag=True,
            cfg=cfg, model_result=result,
            X_train=X_train, y_train=y_train, X_test=X_test,
            query_instances=query_instances, n_eval=n_eval,
        )
        out_pq = output_dir / f"{run_ctx['run_id']}_perquery_cf_metrics.csv"
        summary_pq.to_csv(out_pq, index=False)
        log.info(f"      Per-query CSV: {out_pq}")

        # Aggregate comparison
        agg_g = summary_g.drop(columns=["i", "n_cfs"]).mean()
        agg_pq = summary_pq.drop(columns=["i", "n_cfs"]).mean()
        comparison = pd.DataFrame({
            "metric": agg_g.index,
            "global": agg_g.values,
            "per_query": agg_pq.values,
            "delta": (agg_pq.values - agg_g.values),
            "rel_delta_pct": ((agg_pq.values - agg_g.values) / np.where(np.abs(agg_g.values) > 1e-9, agg_g.values, 1.0)) * 100,
        })
        out_cmp = output_dir / f"{run_ctx['run_id']}_comparison.csv"
        comparison.to_csv(out_cmp, index=False)
        log.info(f"      Comparison CSV: {out_cmp}")
        log.info("      Comparison summary:")
        for _, row in comparison.iterrows():
            log.info(f"        {row['metric']:<22s} global={row['global']:.4f}  per_query={row['per_query']:.4f}  Δ={row['delta']:+.4f}")

        # Per-feature breakdown for both modes
        pf_g = aggregate_per_feature(breakdowns_g, mode_label="global")
        pf_pq = aggregate_per_feature(breakdowns_pq, mode_label="per_query")
        per_feat_df = pd.concat([pf_g, pf_pq], ignore_index=True)
        per_feat_csv = output_dir / f"{run_ctx['run_id']}_per_feature.csv"
        per_feat_df.to_csv(per_feat_csv, index=False)
        log.info(f"      Per-feature CSV: {per_feat_csv}")

        notes_str = (
            f"compare_modes=True; method={cfg['dice']['method']}; "
            f"sample_n={sample_n}; n_queries={n_eval}; "
            f"validity[g={agg_g['validity']:.4f}|pq={agg_pq['validity']:.4f}]; "
            f"actionability[g={agg_g['actionability']:.4f}|pq={agg_pq['actionability']:.4f}]; "
            f"AUC={result['auc']:.4f}"
        )

    # §11.4 finalize
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
            "dice": dict(cfg["dice"]),
            "evaluate": cfg["evaluate"],
        },
        notes=notes_str,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
