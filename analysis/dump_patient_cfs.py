"""Per-patient counterfactual dump for the clinical case-study table.

Selects the K patients with the most wrong-direction violations under global
mode, regenerates their counterfactuals in both global and per-query modes,
and writes the full counterfactual feature values to a single CSV. The output
is intended to populate a manuscript table that contrasts, for one or two
illustrative patients, the baseline feature values against the CFs proposed
under each mode --- with the wrong-direction recommendations under global mode
shown side-by-side with the taxonomy-corrected per-query CFs.

Run:    python analysis/dump_patient_cfs.py
Output: <output_dir>/patient_cfs_example.csv  (output_dir from
        configs/default.yaml, default outputs/scratch/; rows: baseline +
        5 global CFs + 5 per-query CFs for each of the K selected patients)
        and a small markdown summary printed to stdout.
Wall-clock: ~30-60 seconds (K=3 patients, both modes).

This wrapper lives at <repo>/analysis/, so the repo root resolves to its
parent's parent directory. Paths from configs/default.yaml are interpreted
relative to the repo root, not the current working directory, so the script
works regardless of where Python is invoked.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

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
    set_conditional_disabled,
)
from src.pipelines.data.loader import TARGET_COL, load_dataset
from src.pipelines.models.xgb_train import XGBConfig, train_xgb
from src.pipelines.preprocessing.pipeline import get_train_test_split
from src.utils.seed import seed_everything


K_PATIENTS = 3   # number of worst-case patients to dump


def _resolve_repo_path(path_str: str) -> str:
    """If path_str is relative, anchor it at the repo root."""
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def _generate_cfs(
    *,
    per_query_flag: bool,
    cfg: dict,
    model_result: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    query_instances: pd.DataFrame,
):
    """Run DiCE for the given mode on the supplied query_instances. Returns
    the raw DiCE cf_examples object (one entry per query)."""
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
    return runner.generate(query_instances)


def _score_cfs(cfs_df: pd.DataFrame, model, desired_class: int) -> np.ndarray:
    """Return predicted P(diabetes=1) for each CF row."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(cfs_df.values)[:, 1]
    else:
        proba = model.predict(cfs_df.values).astype(float)
    return proba


def _row_dict(
    *,
    patient_id: int,
    row_type: str,
    cf_rank: int,
    features: pd.Series,
    pred_proba: float,
    n_wrong_dir: int = -1,
    n_immutable: int = -1,
) -> dict:
    """Assemble one CSV row."""
    row = {
        "patient_id": patient_id,
        "row_type": row_type,
        "cf_rank": cf_rank,
        "pred_proba": round(float(pred_proba), 4),
        "diabetes_pred": int(pred_proba >= 0.5),
        "wrong_dir_violations": n_wrong_dir,
        "immutable_violations": n_immutable,
    }
    for col, val in features.items():
        if col == TARGET_COL:
            continue
        if isinstance(val, (int, np.integer)):
            row[col] = int(val)
        else:
            row[col] = round(float(val), 3)
    return row


def main() -> None:
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
    log.info(f"Patient CF dump: K={K_PATIENTS} worst-case patients")
    log.info("=" * 60)

    seed_everything(cfg["random"]["seed"])

    # Taxonomy mode (default: 5-class with conditional)
    conditional_disabled = bool(cfg.get("taxonomy", {}).get("conditional_class_disabled", False))
    set_conditional_disabled(conditional_disabled)

    # Load + split + train
    log.info("[Load Dataset]")
    data_csv_path = _resolve_repo_path(cfg["paths"]["data_csv"])
    X, y = load_dataset(data_csv_path)
    log.info(f"      shape: X={X.shape}, y={y.shape}")

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

    # Reproduce the authoritative high-risk cohort selection
    n_eval = min(cfg["evaluate"]["n_test_instances"], len(X_test))
    high_risk_idx = np.argsort(result["proba"])[-n_eval:]
    cohort = X_test.iloc[high_risk_idx].reset_index(drop=True)
    cohort_probs = result["proba"][high_risk_idx]
    log.info(f"[Cohort] top-{n_eval} high-risk patients (authoritative cohort)")

    # Read the existing global-mode per-patient summary to identify the K
    # patients with the highest wrong-direction violations.
    summary_path = Path(_resolve_repo_path(cfg["paths"]["output_dir"])) / "global_cf_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Not found: {summary_path}. "
            "Run the main pipeline first (python run_main.py) so the global-mode "
            "per-patient metrics are available."
        )
    summary = pd.read_csv(summary_path)
    summary = summary.sort_values("wrong_direction_violations", ascending=False)
    top_k = summary.head(K_PATIENTS)
    log.info("[Selected worst-case patients under global mode]")
    log.info(f"  patient_id |  pred_proba | wrong_dir_violations | actionability")
    for _, row in top_k.iterrows():
        i = int(row["i"])
        log.info(
            f"  {i:10d} | {cohort_probs[i]:10.3f} | "
            f"{row['wrong_direction_violations']:19.3f} | {row['actionability']:13.3f}"
        )

    selected_indices = [int(r) for r in top_k["i"].values]
    queries_subset = cohort.iloc[selected_indices].reset_index(drop=True)

    # Generate counterfactuals for the selected patients under both modes
    log.info("[Generating CFs for selected patients]")
    log.info("  global mode...")
    cf_global = _generate_cfs(
        per_query_flag=False, cfg=cfg, model_result=result,
        X_train=X_train, y_train=y_train, query_instances=queries_subset,
    )
    log.info("  per-query mode...")
    cf_perquery = _generate_cfs(
        per_query_flag=True, cfg=cfg, model_result=result,
        X_train=X_train, y_train=y_train, query_instances=queries_subset,
    )

    # Build the long-format CSV: one row per (patient, row_type, cf_rank)
    rows: List[dict] = []
    discrete = get_discrete_features()

    for local_i, patient_id in enumerate(selected_indices):
        baseline = queries_subset.iloc[local_i].copy()
        baseline_proba = float(result["proba"][high_risk_idx[patient_id]])
        rows.append(_row_dict(
            patient_id=patient_id,
            row_type="baseline",
            cf_rank=0,
            features=baseline,
            pred_proba=baseline_proba,
        ))

        for mode_label, cf_examples in [("cf_global", cf_global), ("cf_perquery", cf_perquery)]:
            cf_obj = cf_examples[local_i]
            cfs = cf_obj.final_cfs_df if cf_obj is not None else None
            if cfs is None or len(cfs) == 0:
                log.warning(f"  patient {patient_id} ({mode_label}): no CFs returned")
                continue
            cfs_df = cfs.drop(columns=[TARGET_COL]) if TARGET_COL in cfs.columns else cfs.copy()
            # Discrete-feature rounding: DiCE-ml accepts all continuous, so we
            # round integer-valued features post hoc to recover their support
            # before the row is written to CSV. BMI stays continuous.
            for c in discrete:
                if c in cfs_df.columns:
                    cfs_df[c] = cfs_df[c].round().astype(int)
            cf_probas = _score_cfs(cfs_df, result["model"], cfg["dice"]["desired_class"])
            for rank in range(len(cfs_df)):
                cf_row = cfs_df.iloc[rank]
                action = actionability_score(baseline, cf_row)
                rows.append(_row_dict(
                    patient_id=patient_id,
                    row_type=mode_label,
                    cf_rank=rank + 1,
                    features=cf_row,
                    pred_proba=float(cf_probas[rank]),
                    n_wrong_dir=int(action.get("wrong_direction_violations", 0)),
                    n_immutable=int(action.get("immutable_violations", 0)),
                ))

    out_df = pd.DataFrame(rows)
    out_path = Path(_resolve_repo_path(cfg["paths"]["output_dir"])) / "patient_cfs_example.csv"
    out_df.to_csv(out_path, index=False)

    log.info("=" * 60)
    log.info(f"Wrote {len(out_df)} rows to {out_path.relative_to(REPO_ROOT)}")
    log.info("=" * 60)
    log.info("Column layout:")
    log.info(f"  patient_id, row_type, cf_rank, pred_proba, diabetes_pred,")
    log.info(f"  wrong_dir_violations, immutable_violations, [21 feature columns]")
    log.info("")
    log.info("Row types per patient:")
    log.info(f"  baseline (1 row)  -> the patient's original feature values")
    log.info(f"  cf_global (up to 5 rows)  -> CFs from unconstrained DiCE search")
    log.info(f"  cf_perquery (up to 5 rows) -> CFs from per-query taxonomy constraints")
    log.info("")
    log.info("For the manuscript table, pick one patient (typically the most")
    log.info("extreme case) and one CF per mode that best illustrates the contrast.")


if __name__ == "__main__":
    main()
