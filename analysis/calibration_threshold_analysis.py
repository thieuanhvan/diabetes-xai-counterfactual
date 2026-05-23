"""Calibration + threshold sensitivity analysis for IJMI §4.2.1.

Reads outputs/test_predictions.csv (y_true, y_prob) produced by the
authoritative pipeline run, computes:

    1. Brier score + log loss
    2. 10-bin quantile calibration curve (saves PNG)
    3. Sensitivity/specificity/BAcc/PPV/NPV/F1 at thresholds
       {0.30, 0.35, 0.40, 0.45, 0.50}
    4. Youden's J optimal threshold (full ROC grid search)

Outputs:
    outputs/fig_calibration_curve.png
    outputs/threshold_sensitivity.csv
    Console summary block ready to paste into manuscript.

No model retraining. Reads predictions only.

Right-click run in PyCharm (REPO_ROOT auto-detected) or:
    python analysis/calibration_threshold_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss, confusion_matrix, log_loss, roc_curve,
)

# Resolve repo root regardless of cwd
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
PREDS_CSV = OUTPUTS_DIR / "test_predictions.csv"


def main() -> int:
    if not PREDS_CSV.exists():
        print(f"ERROR: {PREDS_CSV} not found.")
        print("Re-run the authoritative pipeline (run_main.py) first;")
        print("the patched src/pipelines/main.py will dump test_predictions.csv.")
        return 1

    preds = pd.read_csv(PREDS_CSV)
    y_true = preds["y_true"].values
    y_prob = preds["y_prob"].values
    print(f"Loaded {len(y_true)} test predictions from {PREDS_CSV}")
    print(f"Prevalence: {np.mean(y_true):.4f}")
    print()

    # ---- 1. Brier score + log loss ----------------------------------------
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    print(f"Brier score: {brier:.4f}")
    print(f"Log loss:    {ll:.4f}")
    print()

    # ---- 2. Calibration curve (10-bin quantile) ---------------------------
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", lw=2, ms=8, label="XGBoost")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve, BRFSS 2021 test set")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig_path = OUTPUTS_DIR / "fig_calibration_curve.png"
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")
    print()

    # ---- 3. Threshold sweep -----------------------------------------------
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
        rows.append({
            "threshold": t,
            "sens": sens,
            "spec": spec,
            "bacc": (sens + spec) / 2,
            "ppv": ppv,
            "npv": npv,
            "f1": f1,
            "youden": sens + spec - 1,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })

    # ---- 4. Youden's J optimal threshold (full grid search) ---------------
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    opt_idx = int(np.argmax(youden))
    opt_t = float(thr[opt_idx])
    opt_sens = float(tpr[opt_idx])
    opt_spec = float(1 - fpr[opt_idx])
    rows.append({
        "threshold": round(opt_t, 4),
        "sens": opt_sens,
        "spec": opt_spec,
        "bacc": (opt_sens + opt_spec) / 2,
        "ppv": float("nan"),
        "npv": float("nan"),
        "f1": float("nan"),
        "youden": float(youden[opt_idx]),
        "tp": -1, "fp": -1, "tn": -1, "fn": -1,
    })

    df = pd.DataFrame(rows)
    print("=== Threshold sensitivity sweep ===")
    print(df.round(4).to_string(index=False))
    print()
    sweep_csv = OUTPUTS_DIR / "threshold_sensitivity.csv"
    df.to_csv(sweep_csv, index=False)
    print(f"Saved: {sweep_csv}")
    print()

    # ---- 5. Summary block ready for manuscript ----------------------------
    print("=" * 60)
    print("Summary for manuscript §4.2.1:")
    print("=" * 60)
    print(f"Brier score:                    {brier:.4f}")
    print(f"Log loss:                       {ll:.4f}")
    print(f"Youden optimal threshold:       {opt_t:.4f}")
    print(f"  Sensitivity at optimal:       {opt_sens:.4f}")
    print(f"  Specificity at optimal:       {opt_spec:.4f}")
    print(f"  Youden's J:                   {youden[opt_idx]:.4f}")
    print(f"Sensitivity at threshold 0.30:  {rows[0]['sens']:.4f}")
    print(f"Sensitivity at threshold 0.50:  {rows[4]['sens']:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())