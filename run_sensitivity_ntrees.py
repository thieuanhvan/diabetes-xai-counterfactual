"""
Tree-count sensitivity (n_estimators 300 vs 500) for the IJMI revision (R1-4).
Standalone, additive: writes only to outputs/sens_ntrees_*, never touches the
authoritative run or the ablation outputs. Right-click -> Run in PyCharm.

Reports (a) model-level agreement on the full test set and (b) per-query CF
metrics on a matched high-risk cohort, for 300 vs 500 trees.
"""
import sys, yaml, tempfile, os
import numpy as np
sys.path.insert(0, ".")
from src.pipelines.data.loader import load_dataset
from src.pipelines.preprocessing.pipeline import get_train_test_split
from src.pipelines.models.xgb_train import XGBConfig, train_xgb
from src.pipelines.main import main as run_pipeline
from sklearn.metrics import roc_auc_score
import pandas as pd

BASE = "configs/default.yaml"
N_COHORT = 100  # matched cohort for the CF-level comparison

def model_level():
    cfg = yaml.safe_load(open(BASE))
    X, y = load_dataset(cfg["paths"]["data_csv"])
    Xtr, Xte, ytr, yte = get_train_test_split(
        X, y, test_size=cfg["split"]["test_size"], seed=cfg["random"]["seed"],
        stratify=cfg["split"]["stratify"])
    proba = {}
    print("== Model level (full test set) ==")
    for n in (500, 300):
        c = dict(cfg["xgboost"]); c["n_estimators"] = n
        out = train_xgb(Xtr, ytr, Xte, yte, XGBConfig(**c))
        proba[n] = np.asarray(out["proba"])
        print(f"  n_estimators={n}: AUC={roc_auc_score(yte, proba[n]):.4f}")
    p5, p3 = proba[500], proba[300]
    print(f"  proba Pearson r = {np.corrcoef(p3, p5)[0, 1]:.5f}")
    print(f"  class agreement @0.5 = {np.mean((p3 >= .5) == (p5 >= .5)) * 100:.3f}%")

def cf_level():
    cfg = yaml.safe_load(open(BASE))
    print(f"\n== CF level (per-query, matched {N_COHORT}-patient high-risk cohort) ==")
    for n in (500, 300):
        c = dict(cfg)
        c["xgboost"] = dict(cfg["xgboost"]); c["xgboost"]["n_estimators"] = n
        c["paths"] = dict(cfg["paths"]); c["paths"]["output_dir"] = f"outputs/sens_ntrees_{n}"
        c["evaluate"] = dict(cfg["evaluate"]); c["evaluate"]["compare_modes"] = False
        c["evaluate"]["n_test_instances"] = N_COHORT
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(c, f); tmp = f.name
        run_pipeline(tmp); os.unlink(tmp)
        d = pd.read_csv(f"outputs/sens_ntrees_{n}/cf_metrics.csv")
        print(f"  n_estimators={n}: validity={d.validity.mean():.4f} "
              f"actionability={np.nanmean(d.actionability):.4f} "
              f"wrong_dir={d.wrong_direction_violations.mean():.4f} "
              f"immutable={d.immutable_violations.mean():.4f}")

if __name__ == "__main__":
    model_level()
    cf_level()
