"""Pipeline orchestrator (Counterfactual + Actionability on BRFSS 2021).

Runs the full experiment end-to-end:
  data load -> preprocessing -> XGBoost training -> DiCE CF generation in
  global + per-query modes -> CF metric computation -> per-feature breakdown
  -> JSON sidecar + CSVs to outputs/ -> figures + topk_violations (via wrappers).

Config-driven via configs/default.yaml. Two optional knobs support the
ablation suite:
  cfg["evaluate"]["risk_threshold_min"] (default 0.0): predicted-probability
    cutoff applied before high-risk selection (Ablation 4 risk cohort sweep).
  cfg["run"]["notes_suffix"] (default ""): appended to notes_str so that
    ablation.aggregate can group runs by ablation type via the marker.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from src.pipelines.counterfactual.actionability import actionability_score
from src.pipelines.counterfactual.dice_runner import DiCEConfig, DiCERunner
from src.pipelines.counterfactual.feature_taxonomy import (
    get_discrete_features,
    get_feature_ranges,
    set_conditional_disabled,
    set_socioeconomic_proxies_immutable,
)
from src.pipelines.data.loader import TARGET_COL, load_dataset
from src.pipelines.evaluate.cf_metrics import (
    diversity, plausibility, proximity_l1, sparsity, validity,
)
from src.pipelines.evaluate.per_feature import (
    aggregate_per_feature,
    per_feature_breakdown_one_query,
)
from src.pipelines.models.xgb_train import XGBConfig, train_xgb
from src.pipelines.preprocessing.pipeline import get_train_test_split
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

    # DiCERunner.generate() returns List[Optional[cf_example]]
    # NOT a CounterfactualExamples wrapper. Iterate the list directly, handle None entries.
    log.info(f"  [{mode_label}] scoring CFs...")
    per_instance_rows = []
    per_feature_per_query: List[Dict] = []
    n_skipped = 0

    for i in range(len(query_instances)):
        q = query_instances.iloc[i]
        cf_obj = cf_examples[i]
        cfs = cf_obj.final_cfs_df if cf_obj is not None else None
        if cfs is None or len(cfs) == 0:
            n_skipped += 1
            per_instance_rows.append({
                "i": i, "n_cfs": 0,
                "validity": 0.0, "proximity_L1": np.nan,
                "sparsity": np.nan, "diversity": np.nan,
                "plausibility_kNN50": np.nan,
                "actionability": np.nan,
                "wrong_direction_violations": np.nan,
                "immutable_violations": np.nan,
            })
            per_feature_per_query.append({})
            continue
        cfs_df = cfs.drop(columns=[TARGET_COL]) if TARGET_COL in cfs.columns else cfs
        # round discrete features post-DiCE.
        # dice_runner.py passes all features as continuous (DiCE-ml 0.12 quirk workaround).
        # Without rounding, binary features (e.g. AnyHealthcare 0.001 vs 0) register as
        # spurious "changes" → inflate per_feature counts. BMI is only true continuous feature.
        cfs_df = cfs_df.copy()
        for c in get_discrete_features():
            if c in cfs_df.columns:
                cfs_df[c] = cfs_df[c].round().astype(int)
        cf_preds = model_result["model"].predict(cfs_df.values)
        ranges = get_feature_ranges()
        v = validity(cf_preds, cfg["dice"]["desired_class"])
        prox = proximity_l1(q, cfs_df, ranges)
        spar = sparsity(q, cfs_df)
        div = diversity(cfs_df) if len(cfs_df) > 1 else 0.0
        plaus = plausibility(cfs_df, X_train, k=cfg["evaluate"]["plausibility_neighbors"])
        # actionability_score takes single CF (pd.Series),
        # returns Dict[str, float]. Aggregate across CFs in cfs_df via mean (matches
        # convention of validity/proximity/sparsity which return per-query scalars).
        action_dicts = [actionability_score(q, cfs_df.iloc[j]) for j in range(len(cfs_df))]
        act = float(np.mean([d["score"] for d in action_dicts]))
        wd_v = float(np.mean([d["wrong_direction_violations"] for d in action_dicts]))
        imm_v = float(np.mean([d["immutable_violations"] for d in action_dicts]))
        per_instance_rows.append({
            "i": i, "n_cfs": int(len(cfs_df)),
            "validity": float(v), "proximity_L1": float(prox),
            "sparsity": float(spar), "diversity": float(div),
            "plausibility_kNN50": float(plaus),
            "actionability": float(act),
            "wrong_direction_violations": float(wd_v),
            "immutable_violations": float(imm_v),
        })
        # per_feature_breakdown_one_query takes 2 args, not 3
        per_feature_per_query.append(
            per_feature_breakdown_one_query(q, cfs_df)
        )

    log.info(f"  [{mode_label}] processed {len(query_instances)}/{len(query_instances)} (skipped {n_skipped})")
    return pd.DataFrame(per_instance_rows), per_feature_per_query


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

    # conditional class flag for Ablation 5 taxonomy.
    # Default False → 5-class taxonomy (CONDITIONAL preserved). When True,
    # CONDITIONAL collapses to MONOTONIC_DOWN (DiffWalk treated as monotonic_down).
    conditional_disabled = bool(cfg.get("taxonomy", {}).get("conditional_class_disabled", False))
    set_conditional_disabled(conditional_disabled)
    if conditional_disabled:
        log.info("      [taxonomy] CONDITIONAL class DISABLED → 4-class taxonomy (DiffWalk → MONOTONIC_DOWN)")
    else:
        log.info("      [taxonomy] CONDITIONAL class enabled → 5-class taxonomy (default)")

    # socioeconomic proxies flag for Ablation 5b (3-class conservative variant).
    # Default False → Income/Education/AnyHealthcare retain MONOTONIC_UP. When True,
    # those three SOCIOECONOMIC_PROXY_FEATURES collapse to IMMUTABLE.
    se_proxies_immutable = bool(cfg.get("taxonomy", {}).get("socioeconomic_proxies_immutable", False))
    set_socioeconomic_proxies_immutable(se_proxies_immutable)
    if se_proxies_immutable:
        log.info("      [taxonomy] SOCIOECONOMIC PROXIES IMMUTABLE → 3-class conservative (Income/Education/AnyHealthcare → IMMUTABLE)")

    # ---- 1. Load ----
    log.info("[Load Dataset]")
    X, y = load_dataset(cfg["paths"]["data_csv"], sample_n=sample_n, sample_seed=sample_seed)
    if sample_n is not None:
        log.info(f"      [DEV MODE] sampled n={sample_n} (seed={sample_seed}) -> scratch run")
    log.info(f"      shape: X={X.shape}, y={y.shape}, prevalence={y.mean():.4f}")

    # ---- 2. Split + train ----
    log.info("[Train XGBoost]")
    X_train, X_test, y_train, y_test = get_train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        seed=cfg["random"]["seed"],
        stratify=cfg["split"]["stratify"],
    )
    xgb_cfg = XGBConfig(**cfg["xgboost"])
    result = train_xgb(X_train, y_train, X_test, y_test, xgb_cfg)
    log.info(f"      test AUC: {result['auc']:.4f}")

    # ---- 3. Pick high-risk queries (shared across modes) ----
    log.info(f"[CF Generation] compare_modes={compare_modes}")

    # risk_threshold_min support for Ablation 4 (class balance).
    # When > 0, restricts query pool to patients with predicted P(diabetes=1) >= threshold
    # BEFORE selecting top-N. Default 0.0 preserves original behavior (all patients eligible,
    # top-N by ranking).
    risk_threshold_min = float(cfg.get("evaluate", {}).get("risk_threshold_min", 0.0))
    if risk_threshold_min > 0.0:
        eligible_mask = result["proba"] >= risk_threshold_min
        eligible_count = int(eligible_mask.sum())
        log.info(f"      risk_threshold_min={risk_threshold_min:.2f} → eligible pool: {eligible_count}/{len(X_test)} test patients")
        if eligible_count == 0:
            raise ValueError(
                f"No test patients meet risk_threshold_min={risk_threshold_min}. "
                f"Max proba in test = {float(result['proba'].max()):.4f}. "
                f"Lower threshold or check pipeline."
            )
        eligible_idx = np.where(eligible_mask)[0]
        # Rank eligible patients by proba descending, take top-N
        eligible_proba = result["proba"][eligible_idx]
        order_within_eligible = np.argsort(eligible_proba)
        n_eval = min(cfg["evaluate"]["n_test_instances"], eligible_count)
        # Pick top-N highest proba within eligible
        top_in_eligible = order_within_eligible[-n_eval:]
        high_risk_idx = eligible_idx[top_in_eligible]
    else:
        n_eval = min(cfg["evaluate"]["n_test_instances"], len(X_test))
        high_risk_idx = np.argsort(result["proba"])[-n_eval:]

    query_instances = X_test.iloc[high_risk_idx].reset_index(drop=True)
    log.info(f"      n_queries (high-risk): {n_eval}")

    output_dir = Path(cfg["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dump test predictions for downstream calibration/threshold analysis
    pd.DataFrame({"y_true": np.asarray(y_test).ravel(), "y_prob": result["proba"]}).to_csv(output_dir / "test_predictions.csv", index=False)
    log.info(f"      Test predictions CSV: {output_dir / 'test_predictions.csv'}")

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
        out_csv = output_dir / "cf_metrics.csv"
        summary.to_csv(out_csv, index=False)
        log.info(f"      Per-instance CSV: {out_csv}")
        log.info("      Aggregate metrics:")
        agg = summary.drop(columns=["i", "n_cfs"]).mean()
        for k, v in agg.items():
            log.info(f"        {k:<22s} {v:.6f}")

        # NEW: per-feature breakdown for the single mode
        per_feat_df = aggregate_per_feature(breakdowns, mode_label=mode_label)
        per_feat_csv = output_dir / "per_feature.csv"
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
        log.info("[Global Mode] features_to_vary='all', no per-query constraints")
        summary_g, breakdowns_g = _run_one_eval(
            mode_label="global", per_query_flag=False,
            cfg=cfg, model_result=result,
            X_train=X_train, y_train=y_train, X_test=X_test,
            query_instances=query_instances, n_eval=n_eval,
        )
        out_g = output_dir / "global_cf_metrics.csv"
        summary_g.to_csv(out_g, index=False)
        log.info(f"      Global CSV: {out_g}")

        log.info("[Per-Query Mode] taxonomy-constrained")
        summary_pq, breakdowns_pq = _run_one_eval(
            mode_label="per_query", per_query_flag=True,
            cfg=cfg, model_result=result,
            X_train=X_train, y_train=y_train, X_test=X_test,
            query_instances=query_instances, n_eval=n_eval,
        )
        out_pq = output_dir / "perquery_cf_metrics.csv"
        summary_pq.to_csv(out_pq, index=False)
        log.info(f"      Per-query CSV: {out_pq}")

        # NEW: per-feature breakdown comparison
        # aggregate_per_feature signature is (per_query_results, mode_label).
        # Call once per mode, concatenate. topk_violations.py expects long-form
        # with 'mode' column to pivot on.
        df_g = aggregate_per_feature(breakdowns_g, mode_label="global")
        df_pq = aggregate_per_feature(breakdowns_pq, mode_label="per_query")
        per_feat_df = pd.concat([df_g, df_pq], ignore_index=True)
        per_feat_csv = output_dir / "per_feature.csv"
        per_feat_df.to_csv(per_feat_csv, index=False)
        log.info(f"      Per-feature CSV: {per_feat_csv}")

        # Aggregate metrics for notes
        agg_g = summary_g.drop(columns=["i", "n_cfs"]).mean()
        agg_pq = summary_pq.drop(columns=["i", "n_cfs"]).mean()

        # rename verbose metric names to short form
        # matching make_figures.py + analysis scripts expectations.
        metric_rename = {
            "proximity_L1": "proximity",
            "plausibility_kNN50": "plausibility",
            "wrong_direction_violations": "wrong_dir_violations",
        }
        agg_g.index = [metric_rename.get(n, n) for n in agg_g.index]
        agg_pq.index = [metric_rename.get(n, n) for n in agg_pq.index]

        # Comparison CSV (aggregate + delta) — suppress divide-by-zero warning
        # when global=0 (e.g. immutable_violations always 0 in well-behaved runs).
        with np.errstate(divide='ignore', invalid='ignore'):
            comparison = pd.DataFrame({
                "metric": agg_g.index,
                "global": agg_g.values,
                "per_query": agg_pq.values,
                "delta_abs": agg_pq.values - agg_g.values,
                "rel_delta_pct": 100.0 * (agg_pq.values - agg_g.values) / agg_g.values,
            })
        comp_csv = output_dir / "comparison.csv"
        comparison.to_csv(comp_csv, index=False)
        log.info(f"      Comparison CSV: {comp_csv}")

        # emit Comparison summary log block
        log.info("      Comparison summary:")
        for metric in agg_g.index:
            g_val = float(agg_g[metric])
            pq_val = float(agg_pq[metric])
            delta = pq_val - g_val
            log.info(f"        {metric:<22s} global={g_val:.4f}  per_query={pq_val:.4f}  Δ={delta:+.4f}")

        notes_str = (
            f"compare_modes=True; method={cfg['dice']['method']}; "
            f"sample_n={sample_n}; n_queries={n_eval}; "
            f"validity[g={agg_g['validity']:.4f}|pq={agg_pq['validity']:.4f}]; "
            f"actionability[g={agg_g['actionability']:.4f}|pq={agg_pq['actionability']:.4f}]; "
            f"AUC={result['auc']:.4f}"
        )

    # append notes_suffix for ablation grouping
    # aggregate.py parses 'class_threshold=' and 'taxonomy_n_classes=' markers
    notes_suffix = cfg.get("run", {}).get("notes_suffix", "").strip()
    if notes_suffix:
        notes_str = f"{notes_str}; {notes_suffix}"

    # Finalize run: emit log + config sidecar
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
            # capture taxonomy config in hyperparameters JSON for reproducibility
            "taxonomy": cfg.get("taxonomy", {}),
        },
        # Record extended classifier metrics (calibration, threshold sweep)
        classifier_metrics=result.get("extended_metrics"),
        notes=notes_str,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)