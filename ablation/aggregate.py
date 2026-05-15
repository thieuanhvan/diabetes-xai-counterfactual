"""Post-ablation aggregation — read run JSONs, build comparison CSVs.

After each ablation finishes (e.g. multi-seed = 5 runs), call the matching
build_table_*() function here to compress the N outputs/run_*.json files into
a single comparison CSV ready for Bảng 4.5.x in main_vi v7.

Discovery pattern:
- Each run produces outputs/run_<YYYYMMDD_HHMM>.json with config in 'hyperparameters'
- aggregate functions filter runs by 'ablation=<type>' marker in notes_suffix
  (PRIMARY) with fallback to config-field filter + dedup-by-latest-timestamp
  (SECONDARY, for runs predating the marker convention)
- Output CSVs go to outputs/ablation_<name>_table.csv

Per Generalrule v37 §11.4 — output paths back into outputs/ so they live
alongside the data they summarize.

Usage examples (run from repo root):

    python -m ablation.aggregate seed       # build Bảng 4.5.1
    python -m ablation.aggregate method     # build Bảng 4.5.2
    python -m ablation.aggregate n_cf       # build Bảng 4.5.3
    python -m ablation.aggregate class      # build Bảng 4.5.4
    python -m ablation.aggregate taxonomy   # build Bảng 4.5.5

Or import + call programmatically from a wrapper.

NOTE: aggregator reads CF metric CSVs (run_*_cf_metrics.csv) for per-instance
metrics + JSON sidecars for run config. JSON-only would miss aggregate metric
values which live in the per-instance CSV not the config dict.

v3 changes (16/05/2026): primary filter via 'ablation=<type>' marker added to
seed/n_cf/method grids in run_ablation_all.py. For each builder:
1. PRIMARY pass: filter runs where notes contains 'ablation=<type>'. If found,
   use only those.
2. FALLBACK pass: if no marker-tagged runs found, filter by config field +
   dedup by latest timestamp per unique parameter value. Preserves backward
   compat with runs predating the marker convention (e.g. 15/05/2026 session).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"


# ──────────────────────────────────────────────────────────────────────
# Discovery + loading
# ──────────────────────────────────────────────────────────────────────
def list_recent_runs(limit: int = 100) -> List[Path]:
    """Return list of run JSONs sorted oldest → newest by filename timestamp."""
    runs = sorted(OUTPUTS_DIR.glob("run_*.json"))
    runs = [r for r in runs if "_scratch_" not in r.stem]
    return runs[-limit:]


def load_run(json_path: Path) -> Dict[str, Any]:
    """Load JSON sidecar."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def find_matching_cf_csv(json_path: Path, mode: str = "perquery") -> Optional[Path]:
    """Given outputs/run_X.json, find matching outputs/run_X_<mode>_cf_metrics.csv."""
    run_id = json_path.stem
    candidate = OUTPUTS_DIR / f"{run_id}_{mode}_cf_metrics.csv"
    if candidate.exists():
        return candidate
    fallback = OUTPUTS_DIR / f"{run_id}_cf_metrics.csv"
    if fallback.exists():
        return fallback
    return None


def aggregate_metrics(cf_csv: Path) -> Dict[str, float]:
    """Mean of all per-instance metric columns in a cf_metrics CSV."""
    df = pd.read_csv(cf_csv)
    drop_cols = [c for c in ("i", "n_cfs") if c in df.columns]
    return df.drop(columns=drop_cols).mean().to_dict()


def _has_ablation_marker(notes: str, ablation_type: str) -> bool:
    """Return True if notes contains 'ablation=<type>' marker."""
    return f"ablation={ablation_type}" in notes


def _filter_by_marker_or_fallback(
    runs: List[Path],
    ablation_type: str,
    config_filter_fn,
    dedup_key_fn,
) -> List[tuple]:
    """Two-pass filter: marker first, dedup-by-config-field as fallback.

    Returns list of (json_path, cfg) tuples ready for row construction.

    config_filter_fn(cfg) -> bool: True if run matches the ablation parameter family.
    dedup_key_fn(cfg) -> hashable: parameter value to dedup on (e.g. seed value).
    """
    # Pass 1: marker-tagged runs
    marker_runs = []
    for jp in runs:
        cfg = load_run(jp)
        if _has_ablation_marker(cfg.get("notes", ""), ablation_type):
            marker_runs.append((jp, cfg))

    if marker_runs:
        return marker_runs

    # Pass 2 (fallback): config-field filter + dedup by latest timestamp per
    # unique parameter value. Runs sorted oldest→newest, so later runs in
    # iteration overwrite earlier — same parameter value keeps latest.
    by_param = {}
    for jp in runs:
        cfg = load_run(jp)
        if not config_filter_fn(cfg):
            continue
        key = dedup_key_fn(cfg)
        if key is None:
            continue
        by_param[key] = (jp, cfg)
    return list(by_param.values())


# ──────────────────────────────────────────────────────────────────────
# Per-ablation table builders
# ──────────────────────────────────────────────────────────────────────
def build_table_seed(runs: List[Path]) -> pd.DataFrame:
    """Bảng 4.5.1 — multi-seed: 1 row per seed, columns = 4-8 metrics + AUC.

    Primary filter: 'ablation=seed' marker. Fallback: method=random (default) +
    config xgboost.random_state in seed grid + dedup by seed value (keep latest).
    method=random filter prevents method-ablation runs (genetic/kdtree at seed=42)
    from overriding the canonical seed-ablation run_1811.
    """
    SEED_GRID = {42, 123, 2024, 7, 31337}

    def cf(cfg):
        hyp = cfg.get("hyperparameters", {})
        method = hyp.get("dice", {}).get("method")
        s = hyp.get("xgboost", {}).get("random_state")
        return method == "random" and s in SEED_GRID

    def dk(cfg):
        return cfg.get("hyperparameters", {}).get("xgboost", {}).get("random_state")

    selected = _filter_by_marker_or_fallback(runs, "seed", cf, dk)

    rows = []
    for jp, cfg in selected:
        hyp = cfg.get("hyperparameters", {})
        clf = cfg.get("classifier_metrics") or {}
        seed = hyp.get("xgboost", {}).get("random_state")
        cf_csv = find_matching_cf_csv(jp, mode="perquery")
        if cf_csv is None:
            continue
        agg = aggregate_metrics(cf_csv)
        rows.append({
            "run_id": cfg["run_id"],
            "seed": seed,
            "AUC": clf.get("AUC_ROC"),
            "validity": agg.get("validity"),
            "actionability": agg.get("actionability"),
            "wrong_dir_violations": agg.get("wrong_direction_violations"),
            "immutable_violations": agg.get("immutable_violations"),
            "sparsity": agg.get("sparsity"),
            "proximity_L1": agg.get("proximity_L1"),
            "diversity": agg.get("diversity"),
        })
    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    if len(df) >= 2:
        numeric = df.select_dtypes(include="number").columns.drop("seed", errors="ignore")
        summary = pd.DataFrame({
            "run_id": ["MEAN", "STD"],
            "seed": ["—", "—"],
            **{c: [df[c].mean(), df[c].std()] for c in numeric},
        })
        df = pd.concat([df, summary], ignore_index=True)
    return df


def build_table_method(runs: List[Path]) -> pd.DataFrame:
    """Bảng 4.5.2 — method comparison: 1 row per method (random/genetic/kdtree).

    Primary filter: 'ablation=method' marker. Fallback: dedup by method value.
    """
    METHODS = {"random", "genetic", "kdtree"}

    def cf(cfg):
        m = cfg.get("hyperparameters", {}).get("dice", {}).get("method")
        return m in METHODS

    def dk(cfg):
        return cfg.get("hyperparameters", {}).get("dice", {}).get("method")

    selected = _filter_by_marker_or_fallback(runs, "method", cf, dk)

    rows = []
    for jp, cfg in selected:
        method = cfg.get("hyperparameters", {}).get("dice", {}).get("method")
        clf = cfg.get("classifier_metrics") or {}
        cf_csv = find_matching_cf_csv(jp, mode="perquery")
        if cf_csv is None:
            continue
        agg = aggregate_metrics(cf_csv)
        rows.append({
            "run_id": cfg["run_id"],
            "method": method,
            "AUC": clf.get("AUC_ROC"),
            "validity": agg.get("validity"),
            "actionability": agg.get("actionability"),
            "wrong_dir_violations": agg.get("wrong_direction_violations"),
            "sparsity": agg.get("sparsity"),
            "proximity_L1": agg.get("proximity_L1"),
            "diversity": agg.get("diversity"),
            "plausibility_kNN50": agg.get("plausibility_kNN50"),
        })
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def build_table_n_cf(runs: List[Path]) -> pd.DataFrame:
    """Bảng 4.5.3 — n_counterfactuals sweep: 1 row per n_cf value.

    Primary filter: 'ablation=n_cf' marker. Fallback: method=random (default) +
    dedup by n_counterfactuals value.
    """
    N_CF_GRID = {1, 3, 5, 10}

    def cf(cfg):
        hyp = cfg.get("hyperparameters", {})
        method = hyp.get("dice", {}).get("method")
        n = hyp.get("dice", {}).get("n_counterfactuals")
        return method == "random" and n in N_CF_GRID

    def dk(cfg):
        return cfg.get("hyperparameters", {}).get("dice", {}).get("n_counterfactuals")

    selected = _filter_by_marker_or_fallback(runs, "n_cf", cf, dk)

    rows = []
    for jp, cfg in selected:
        n_cf = cfg.get("hyperparameters", {}).get("dice", {}).get("n_counterfactuals")
        cf_csv = find_matching_cf_csv(jp, mode="perquery")
        if cf_csv is None:
            continue
        agg = aggregate_metrics(cf_csv)
        rows.append({
            "run_id": cfg["run_id"],
            "n_counterfactuals": n_cf,
            "validity": agg.get("validity"),
            "actionability": agg.get("actionability"),
            "wrong_dir_violations": agg.get("wrong_direction_violations"),
            "sparsity": agg.get("sparsity"),
            "diversity": agg.get("diversity"),
        })
    return pd.DataFrame(rows).sort_values("n_counterfactuals").reset_index(drop=True)


def build_table_class(runs: List[Path]) -> pd.DataFrame:
    """Bảng 4.5.4 — class balance: 1 row per risk-threshold cohort.

    Filter via 'class_threshold=' marker in notes (existing convention).
    """
    rows = []
    for jp in runs:
        cfg = load_run(jp)
        notes = cfg.get("notes", "")
        marker = "class_threshold="
        if marker not in notes:
            continue
        threshold = notes.split(marker)[1].split(";")[0].strip()
        cf_csv = find_matching_cf_csv(jp, mode="perquery")
        if cf_csv is None:
            continue
        agg = aggregate_metrics(cf_csv)
        rows.append({
            "run_id": cfg["run_id"],
            "risk_threshold": threshold,
            "validity": agg.get("validity"),
            "actionability": agg.get("actionability"),
            "wrong_dir_violations": agg.get("wrong_direction_violations"),
            "sparsity": agg.get("sparsity"),
        })
    return pd.DataFrame(rows).sort_values("risk_threshold").reset_index(drop=True)


def build_table_taxonomy(runs: List[Path]) -> pd.DataFrame:
    """Bảng 4.5.5 — 5-class vs 4-class taxonomy: 2 rows (paired comparison).

    Filter via 'taxonomy_n_classes=' marker in notes (existing convention).
    """
    rows = []
    for jp in runs:
        cfg = load_run(jp)
        notes = cfg.get("notes", "")
        marker = "taxonomy_n_classes="
        if marker not in notes:
            continue
        n_classes = notes.split(marker)[1].split(";")[0].strip()
        cf_csv = find_matching_cf_csv(jp, mode="perquery")
        if cf_csv is None:
            continue
        agg = aggregate_metrics(cf_csv)
        rows.append({
            "run_id": cfg["run_id"],
            "taxonomy_n_classes": n_classes,
            "validity": agg.get("validity"),
            "actionability": agg.get("actionability"),
            "wrong_dir_violations": agg.get("wrong_direction_violations"),
            "immutable_violations": agg.get("immutable_violations"),
        })
    return pd.DataFrame(rows).sort_values("taxonomy_n_classes").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# CLI dispatcher
# ──────────────────────────────────────────────────────────────────────
BUILDERS = {
    "seed": build_table_seed,
    "method": build_table_method,
    "n_cf": build_table_n_cf,
    "class": build_table_class,
    "taxonomy": build_table_taxonomy,
}


def main(ablation_type: str, limit: int = 100) -> Path:
    """Build the table for one ablation type, save CSV to outputs/."""
    if ablation_type not in BUILDERS:
        raise ValueError(f"Unknown ablation type: {ablation_type}. Valid: {list(BUILDERS)}")
    runs = list_recent_runs(limit=limit)
    df = BUILDERS[ablation_type](runs)
    if df.empty:
        print(f"⚠ No matching runs found for ablation_type={ablation_type}")
        print("  Check that runs were executed and config fields are tagged correctly.")
        return None
    out_csv = OUTPUTS_DIR / f"ablation_{ablation_type}_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"✓ Built table: {out_csv}")
    print(df.to_string(index=False))
    return out_csv


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None:
        print(f"Usage: python -m ablation.aggregate {'|'.join(BUILDERS)}")
        sys.exit(1)
    main(arg)
