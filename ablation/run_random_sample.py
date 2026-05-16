"""Random-sample ablation: 100 patients drawn uniformly at random from the
test set, instead of the top-200 high-risk cohort used by the authoritative
pipeline.

Addresses the reviewer concern that the headline gains may hold only at the
high-confidence tail of the risk distribution. The cohort is selected with a
dedicated seed (COHORT_SEED) so the random draw is reproducible.

Run:    python ablation/run_random_sample.py
Output: outputs/random_sample/{comparison, per_feature,
        global_cf_metrics, perquery_cf_metrics}.csv
Wall-clock: ~5 min (similar to the 200-patient main run, per-CF latency
            dominated by DiCE search).

This wrapper lives at <repo>/ablation/, so the repo root resolves to its
parent's parent directory. Paths from configs/default.yaml (data CSV,
output folder) are interpreted relative to the repo root, not the current
working directory, so the script works regardless of where Python is invoked.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.pipelines.counterfactual.actionability import actionability_score
from src.pipelines.counterfactual.dice_runner import DiCEConfig, DiCERunner
from src.pipelines.counterfactual.feature_taxonomy import (
    get_discrete_features,
    get_feature_ranges,
    set_conditional_disabled,
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
from src.utils.seed import seed_everything


N_RANDOM = 100        # number of randomly-sampled patients
COHORT_SEED = 42      # RNG seed for cohort selection (NOT the pipeline seed)


def _resolve_repo_path(path_str: str) -> str:
    """If path_str is relative, anchor it at the repo root. Returns absolute string."""
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def _run_one_eval(
    *,
    mode_label: str,
    per_query_flag: bool,
    cfg: dict,
    model_result: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    query_instances: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Mirror of the per-mode evaluation loop in the main pipeline, simplified
    for ablation use. Generates DiCE counterfactuals for query_instances under
    the given mode, then scores each query's CFs and returns per-instance
    summary plus per-feature breakdown."""
    log = logging.getLogger(__name__)
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
    per_instance_rows: List[Dict] = []
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
        cfs_df = cfs_df.copy()
        # Discrete-feature rounding: dice_runner passes all features as
        # continuous (DiCE-ml quirk); rounding restores binary/ordinal support
        # before metric computation so per-feature counts are not inflated by
        # spurious near-zero changes. BMI is the only genuinely continuous
        # feature and is unaffected.
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
        per_feature_per_query.append(
            per_feature_breakdown_one_query(q, cfs_df)
        )

    log.info(f"  [{mode_label}] processed {len(query_instances)-n_skipped}/{len(query_instances)} (skipped {n_skipped})")
    return pd.DataFrame(per_instance_rows), per_feature_per_query


def main() -> None:
    # Load default config
    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info(f"Random-sample ablation: n={N_RANDOM}, seed={COHORT_SEED}")
    log.info("=" * 60)

    seed_everything(cfg["random"]["seed"])

    # Taxonomy mode (default: 5-class with conditional)
    conditional_disabled = bool(cfg.get("taxonomy", {}).get("conditional_class_disabled", False))
    set_conditional_disabled(conditional_disabled)
    if conditional_disabled:
        log.info("      [taxonomy] CONDITIONAL class DISABLED -> 4-class taxonomy")
    else:
        log.info("      [taxonomy] CONDITIONAL class enabled -> 5-class taxonomy (default)")

    # Load + split + train. Resolve data CSV path against the repo root so the
    # script works regardless of the current working directory at invocation.
    log.info("[Load Dataset]")
    data_csv_path = _resolve_repo_path(cfg["paths"]["data_csv"])
    X, y = load_dataset(data_csv_path)
    log.info(f"      shape: X={X.shape}, y={y.shape}, prevalence={y.mean():.4f}")

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

    # Random cohort selection (NOT top-N by predicted probability)
    rng = np.random.default_rng(COHORT_SEED)
    random_idx = rng.choice(len(X_test), size=N_RANDOM, replace=False)
    query_instances = X_test.iloc[random_idx].reset_index(drop=True)

    cohort_probs = result["proba"][random_idx]
    log.info(f"[Cohort] {N_RANDOM} random patients (seed={COHORT_SEED})")
    log.info(f"      Predicted prob distribution:")
    log.info(f"        mean={cohort_probs.mean():.3f}, median={float(np.median(cohort_probs)):.3f}")
    log.info(f"        q1={float(np.quantile(cohort_probs, 0.25)):.3f}, "
             f"q3={float(np.quantile(cohort_probs, 0.75)):.3f}")
    log.info(f"        min={cohort_probs.min():.3f}, max={cohort_probs.max():.3f}")
    log.info(f"      (Authoritative cohort uses top-200 high-risk; mean prob >0.7)")

    # Run both modes back-to-back on the same random cohort
    log.info("[Global Mode]")
    summary_g, breakdowns_g = _run_one_eval(
        mode_label="global", per_query_flag=False,
        cfg=cfg, model_result=result,
        X_train=X_train, y_train=y_train,
        query_instances=query_instances,
    )
    log.info("[Per-Query Mode]")
    summary_pq, breakdowns_pq = _run_one_eval(
        mode_label="per_query", per_query_flag=True,
        cfg=cfg, model_result=result,
        X_train=X_train, y_train=y_train,
        query_instances=query_instances,
    )

    # Aggregate + comparison
    agg_g = summary_g.drop(columns=["i", "n_cfs"]).mean()
    agg_pq = summary_pq.drop(columns=["i", "n_cfs"]).mean()
    metric_rename = {
        "proximity_L1": "proximity",
        "plausibility_kNN50": "plausibility",
        "wrong_direction_violations": "wrong_dir_violations",
    }
    agg_g.index = [metric_rename.get(n, n) for n in agg_g.index]
    agg_pq.index = [metric_rename.get(n, n) for n in agg_pq.index]

    with np.errstate(divide='ignore', invalid='ignore'):
        comparison = pd.DataFrame({
            "metric": agg_g.index,
            "global": agg_g.values,
            "per_query": agg_pq.values,
            "delta_abs": agg_pq.values - agg_g.values,
            "rel_delta_pct": 100.0 * (agg_pq.values - agg_g.values) / agg_g.values,
        })

    df_g = aggregate_per_feature(breakdowns_g, mode_label="global")
    df_pq = aggregate_per_feature(breakdowns_pq, mode_label="per_query")
    per_feat = pd.concat([df_g, df_pq], ignore_index=True)

    # Write to outputs/random_sample/ subfolder so we do NOT overwrite the
    # main authoritative compare-modes run that lives directly under outputs/.
    out_dir = REPO_ROOT / "outputs" / "random_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_g.to_csv(out_dir / "global_cf_metrics.csv", index=False)
    summary_pq.to_csv(out_dir / "perquery_cf_metrics.csv", index=False)
    per_feat.to_csv(out_dir / "per_feature.csv", index=False)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    log.info("=" * 60)
    log.info("[Random Sample Ablation] Summary:")
    log.info("=" * 60)
    for m, g, p in zip(agg_g.index, agg_g.values, agg_pq.values):
        delta = p - g
        log.info(f"  {m:<22s} global={g:.4f}  per_query={p:.4f}  delta={delta:+.4f}")

    log.info("")
    log.info(f"Outputs written to {out_dir.relative_to(REPO_ROOT)}/:")
    log.info(f"  comparison.csv, per_feature.csv,")
    log.info(f"  global_cf_metrics.csv, perquery_cf_metrics.csv")


if __name__ == "__main__":
    main()
