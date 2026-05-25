"""
One-time artifact preparation for the P4 demo.

Reuses P4's own pipeline functions (no logic duplicated):
    src.pipelines.data.loader.load_dataset
    src.pipelines.preprocessing.pipeline.get_train_test_split
    src.pipelines.models.xgb_train.train_xgb, XGBConfig

Produces:
    demo/models/xgb_brfss2021.joblib    Trained XGBClassifier
    demo/models/X_train_sample.parquet  Stratified 5000-row train subset
                                        (used by DiCE Data() — full 189k is
                                         overkill for neighbor lookups)
    demo/models/y_train_sample.parquet  Targets aligned with X_train_sample
    demo/models/X_test.parquet          Full ~47k test set (for preset
                                        archetype selection by risk bin)
    demo/models/proba_test.parquet      Test-set predicted probabilities
    demo/models/metadata.json           AUC, versions, timestamp, hyperparams

Run from repo root:
    python demo/prepare_demo_artifacts.py

Or with custom config:
    python demo/prepare_demo_artifacts.py --config configs/default.yaml

Runtime: ~30-60 s on a modern laptop (XGB hist tree_method, n_estimators=500).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

# Ensure repo root is on sys.path so `import src.pipelines...` works
# regardless of cwd. demo/ sits at repo_root/demo/ so parent of this
# file's parent is the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.data.loader import TARGET_COL, load_dataset  # noqa: E402
from src.pipelines.models.xgb_train import XGBConfig, train_xgb  # noqa: E402
from src.pipelines.preprocessing.pipeline import get_train_test_split  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).resolve().parent
MODELS_DIR = DEMO_DIR / "models"

OUT_MODEL = MODELS_DIR / "xgb_brfss2021.joblib"
OUT_XTRAIN = MODELS_DIR / "X_train_sample.parquet"
OUT_YTRAIN = MODELS_DIR / "y_train_sample.parquet"
OUT_XTEST = MODELS_DIR / "X_test.parquet"
OUT_PROBA = MODELS_DIR / "proba_test.parquet"
OUT_META = MODELS_DIR / "metadata.json"

# DiCE Data() init benefits from broad feature coverage but does NOT need
# the full 189k train rows. A stratified 5000-row sample gives effectively
# identical neighbor lookups at <1 s init cost.
TRAIN_SAMPLE_SIZE = 5000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "default.yaml"),
        help="P4 config YAML (default: configs/default.yaml)",
    )
    p.add_argument(
        "--train-sample-size",
        type=int,
        default=TRAIN_SAMPLE_SIZE,
        help=f"Stratified train subsample size for DiCE (default: {TRAIN_SAMPLE_SIZE})",
    )
    return p.parse_args()


def stratified_sample(
    X: pd.DataFrame, y: pd.Series, n: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample preserving class balance.

    Implemented via sklearn.train_test_split (train_size = n, stratify = y).
    Avoids the pandas 3.0 groupby(group_keys=False) edge case where the
    group key column is dropped from the result.
    """
    if len(X) <= n:
        return X.reset_index(drop=True).copy(), y.reset_index(drop=True).copy()
    from sklearn.model_selection import train_test_split
    X_s, _, y_s, _ = train_test_split(
        X, y,
        train_size=n,
        stratify=y,
        random_state=seed,
    )
    return X_s.reset_index(drop=True), y_s.reset_index(drop=True)


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("P4 Demo Artifact Preparation")
    print("=" * 60)

    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(cfg_path.read_text())
    print(f"Config:        {cfg_path}")

    # Resolve data CSV (relative to repo root if not absolute)
    csv_path = Path(cfg["paths"]["data_csv"])
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path
    if not csv_path.exists():
        print(f"ERROR: data CSV not found: {csv_path}", file=sys.stderr)
        return 1
    print(f"Data CSV:      {csv_path}")

    # Ensure output dir
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir:    {MODELS_DIR}")
    print()

    # ────────────────────────────────────────────────────────────────
    # Step 1: load dataset
    # ────────────────────────────────────────────────────────────────
    print("[1/4] Loading BRFSS 2021 dataset...")
    X, y = load_dataset(
        csv_path=csv_path,
        sample_n=cfg["data"].get("sample_n"),
        sample_seed=cfg["data"].get("sample_seed", 42),
    )
    print(f"      → {len(X):,} rows × {len(X.columns)} features, "
          f"prevalence = {y.mean():.4f}")

    # ────────────────────────────────────────────────────────────────
    # Step 2: train/test split (matches P4 main.py)
    # ────────────────────────────────────────────────────────────────
    print("[2/4] Stratified train/test split (80/20, seed=42)...")
    X_train, X_test, y_train, y_test = get_train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        seed=cfg["random"]["seed"],
        stratify=cfg["split"]["stratify"],
    )
    print(f"      → train: {len(X_train):,} | test: {len(X_test):,}")

    # ────────────────────────────────────────────────────────────────
    # Step 3: train XGBoost (matches default.yaml hyperparameters)
    # ────────────────────────────────────────────────────────────────
    print("[3/4] Training XGBoost (hist tree, 500 estimators)...")
    xgb_cfg = XGBConfig(**cfg["xgboost"])
    result = train_xgb(X_train, y_train, X_test, y_test, config=xgb_cfg)
    model = result["model"]
    test_auc = result["auc"]
    proba_test = result["proba"]
    print(f"      → test AUC = {test_auc:.4f}")

    # ────────────────────────────────────────────────────────────────
    # Step 4: persist artifacts
    # ────────────────────────────────────────────────────────────────
    print("[4/4] Persisting artifacts...")

    # Model
    joblib.dump(model, OUT_MODEL)
    print(f"      → {OUT_MODEL.name} ({OUT_MODEL.stat().st_size / 1024:.1f} KB)")

    # Stratified train sample for DiCE Data()
    X_train_s, y_train_s = stratified_sample(
        X_train, y_train,
        n=args.train_sample_size,
        seed=cfg["random"]["seed"],
    )
    X_train_s.to_parquet(OUT_XTRAIN, index=False)
    y_train_s.to_frame(name=TARGET_COL).to_parquet(OUT_YTRAIN, index=False)
    print(f"      → {OUT_XTRAIN.name} ({len(X_train_s):,} rows, "
          f"prevalence = {y_train_s.mean():.4f})")

    # Full test set + probabilities (for preset archetype selection)
    X_test.to_parquet(OUT_XTEST, index=False)
    pd.DataFrame({"proba": proba_test}).to_parquet(OUT_PROBA, index=False)
    print(f"      → {OUT_XTEST.name} ({len(X_test):,} rows)")

    # Metadata
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_source": str(cfg_path.relative_to(REPO_ROOT)),
        "data_csv": str(csv_path.relative_to(REPO_ROOT)),
        "n_train": int(len(X_train)),
        "n_train_sample": int(len(X_train_s)),
        "n_test": int(len(X_test)),
        "n_features": int(len(X.columns)),
        "test_auc": round(float(test_auc), 4),
        "prevalence_train": round(float(y_train.mean()), 4),
        "prevalence_test": round(float(y_test.mean()), 4),
        "xgb_config": cfg["xgboost"],
        "split": cfg["split"],
        "seed": cfg["random"]["seed"],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model_file": OUT_MODEL.name,
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"      → {OUT_META.name}")

    print()
    print("=" * 60)
    print("✓ Done. Launch the demo with:")
    print("    streamlit run demo/app.py")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
