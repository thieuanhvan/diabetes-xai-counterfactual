"""External validation: train BRFSS 2021, test BRFSS 2015.

Out-of-distribution evaluation of the trained classifier and the CF pipeline
on an external cohort. Reuses `src.pipelines.main._run_one_eval` for CF
generation (no logic duplication, no changes to existing pipeline code).

Prerequisite:
    Place BRFSS 2015 cleaned CSV at `data/cdc_brfss_diabetes_2015.csv`
    (same 21-feature schema as BRFSS 2021; see data/PROVENANCE.md).

Outputs (written to outputs/external_2015/):
    test_predictions.csv              — y_true + y_prob on 2015
    global_cf_metrics.csv             — per-query metrics, global mode
    perquery_cf_metrics.csv           — per-query metrics, per-query mode
    comparison.csv                    — aggregate global vs per-query
    external_validation_summary.json  — headline numbers for manuscript

Compute time: ~5–7 min (XGB ~6s on 2021 train, prediction on 2015 instant,
DiCE compare-modes ~5 min on top-200 high-risk).

Run:
    Right-click in PyCharm, OR
    python analysis/run_external_validation_brfss2015.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss

# Resolve repo root regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.data.loader import load_dataset
from src.pipelines.main import _run_one_eval
from src.pipelines.models.xgb_train import XGBConfig, train_xgb
from src.pipelines.preprocessing.pipeline import get_train_test_split
from src.utils.seed import seed_everything


def main() -> int:
    # ---- 0. Setup -----------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("external_validation")
    log.info("=" * 70)
    log.info("External validation: train BRFSS 2021, test BRFSS 2015")
    log.info("=" * 70)

    config_path = REPO_ROOT / "configs" / "default.yaml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed_everything(cfg["random"]["seed"])

    # ---- 1. Load 2021 (training cohort) -------------------------------------
    train_csv = REPO_ROOT / "data" / "cdc_brfss_diabetes_2021.csv"
    log.info(f"[1/6] Load TRAIN cohort: {train_csv}")
    X_2021, y_2021 = load_dataset(train_csv)
    log.info(f"      2021 shape: X={X_2021.shape}, prevalence={y_2021.mean():.4f}")

    # Use the same 80/20 split as authoritative pipeline to keep training set
    # bit-identical (model is bit-identical to manuscript v33/v38 results).
    X_train, _X_test_2021, y_train, _y_test_2021 = get_train_test_split(
        X_2021, y_2021,
        test_size=cfg["split"]["test_size"],
        seed=cfg["random"]["seed"],
        stratify=cfg["split"]["stratify"],
    )
    log.info(f"      train split: X={X_train.shape}, prevalence={y_train.mean():.4f}")

    # ---- 2. Load 2015 (external test cohort) --------------------------------
    test_csv = REPO_ROOT / "data" / "cdc_brfss_diabetes_2015.csv"
    log.info(f"[2/6] Load EXTERNAL TEST cohort: {test_csv}")
    if not test_csv.exists():
        log.error(f"BRFSS 2015 CSV not found at: {test_csv}")
        log.error("Place the BRFSS 2015 cleaned CSV at this path. "
                  "See data/README.md and data/PROVENANCE.md for acquisition. "
                  "Expected 21 features, same schema as BRFSS 2021.")
        return 1
    X_test, y_test = load_dataset(test_csv)
    log.info(f"      2015 shape: X={X_test.shape}, prevalence={y_test.mean():.4f}")

    # Income encoding caveat: 2015 Income range is 1-8, 2021 is 1-11.
    # Tree-based classifier handles out-of-range values gracefully (decision
    # rules on numerical splits do not extrapolate). Reported in manuscript.
    income_max_2021 = int(X_train["Income"].max())
    income_max_2015 = int(X_test["Income"].max())
    log.info(f"      Income encoding: train(2021) max={income_max_2021}, "
             f"test(2015) max={income_max_2015}  "
             f"(tree handles out-of-range gracefully)")

    # ---- 3. Train XGB on 2021 train split, evaluate on 2015 -----------------
    log.info("[3/6] Train XGBoost on 2021 train split, evaluate on 2015...")
    xgb_cfg = XGBConfig(**cfg["xgboost"])
    result = train_xgb(X_train, y_train, X_test, y_test, xgb_cfg)
    external_auc = float(result["auc"])
    y_prob = result["proba"]
    external_brier = float(brier_score_loss(y_test.values, y_prob))
    log.info(f"      External (2015) AUC:   {external_auc:.4f} "
             f"(compare 2021 internal: 0.8233)")
    log.info(f"      External (2015) Brier: {external_brier:.4f} "
             f"(compare 2021 internal: 0.0991)")

    # ---- 4. Pick top-200 high-risk in 2015 ----------------------------------
    log.info("[4/6] Select high-risk queries from 2015 cohort...")
    n_eval = min(cfg["evaluate"]["n_test_instances"], len(X_test))
    high_risk_idx = np.argsort(y_prob)[-n_eval:]
    query_instances = X_test.iloc[high_risk_idx].reset_index(drop=True)
    log.info(f"      Top-{n_eval} high-risk queries by predicted probability")
    log.info(f"      Predicted-prob range in queries: "
             f"[{y_prob[high_risk_idx].min():.4f}, {y_prob[high_risk_idx].max():.4f}]")

    # ---- 5. Output dir + dump predictions ----------------------------------
    output_dir = REPO_ROOT / "outputs" / "external_2015"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "y_true": np.asarray(y_test).ravel(),
        "y_prob": y_prob,
    }).to_csv(output_dir / "test_predictions.csv", index=False)
    log.info(f"      Predictions: {output_dir / 'test_predictions.csv'}")

    # ---- 6. Run two modes (reuse _run_one_eval, no code duplication) -------
    log.info("[5/6] Run global mode CF generation on 2015 queries...")
    global_summary, _ = _run_one_eval(
        mode_label="global", per_query_flag=False,
        cfg=cfg, model_result=result,
        X_train=X_train, y_train=y_train, X_test=X_test,
        query_instances=query_instances, n_eval=n_eval,
    )
    global_summary.to_csv(output_dir / "global_cf_metrics.csv", index=False)

    log.info("[6/6] Run per-query mode CF generation on 2015 queries...")
    perquery_summary, _ = _run_one_eval(
        mode_label="per_query", per_query_flag=True,
        cfg=cfg, model_result=result,
        X_train=X_train, y_train=y_train, X_test=X_test,
        query_instances=query_instances, n_eval=n_eval,
    )
    perquery_summary.to_csv(output_dir / "perquery_cf_metrics.csv", index=False)

    # ---- 7. Aggregate comparison + JSON summary ----------------------------
    metrics = [
        "validity", "proximity_L1", "sparsity", "diversity",
        "plausibility_kNN50", "actionability",
        "wrong_direction_violations", "immutable_violations",
    ]
    comparison = []
    for m in metrics:
        g = float(global_summary[m].mean())
        p = float(perquery_summary[m].mean())
        comparison.append({"metric": m, "global": g, "per_query": p, "delta": p - g})
    pd.DataFrame(comparison).to_csv(output_dir / "comparison.csv", index=False)

    summary = {
        "train_cohort": "BRFSS_2021",
        "test_cohort": "BRFSS_2015",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_eval": int(n_eval),
        "prevalence_train": float(y_train.mean()),
        "prevalence_test": float(y_test.mean()),
        "income_max_2021": income_max_2021,
        "income_max_2015": income_max_2015,
        "external_auc": external_auc,
        "external_brier": external_brier,
        "comparison": {row["metric"]: {"global": row["global"],
                                       "per_query": row["per_query"],
                                       "delta": row["delta"]} for row in comparison},
    }
    with open(output_dir / "external_validation_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- 8. Console summary block ready for manuscript ---------------------
    log.info("=" * 70)
    log.info("READY-TO-PASTE MANUSCRIPT NUMBERS (External validation §4.X):")
    log.info("=" * 70)
    log.info(f"  Train: BRFSS 2021 (n_train={len(X_train):,}, "
             f"prev={y_train.mean():.4f})")
    log.info(f"  Test:  BRFSS 2015 (n_test={len(X_test):,}, "
             f"prev={y_test.mean():.4f})")
    log.info(f"")
    log.info(f"  Classifier transfer:")
    log.info(f"    External AUC:   {external_auc:.4f}  "
             f"(2021 internal 0.8233, Δ={external_auc-0.8233:+.4f})")
    log.info(f"    External Brier: {external_brier:.4f}  "
             f"(2021 internal 0.0991, Δ={external_brier-0.0991:+.4f})")
    log.info(f"")
    log.info(f"  CF pipeline transfer (top-{n_eval} high-risk 2015):")
    for m in ["validity", "actionability", "wrong_direction_violations",
              "immutable_violations"]:
        g = summary["comparison"][m]["global"]
        p = summary["comparison"][m]["per_query"]
        d = summary["comparison"][m]["delta"]
        log.info(f"    {m:<28s} global={g:.4f}  per_query={p:.4f}  Δ={d:+.4f}")
    log.info("=" * 70)
    log.info(f"  Summary JSON: {output_dir / 'external_validation_summary.json'}")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())