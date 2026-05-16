"""Post-ablation aggregation — read per-cell archives, build comparison CSVs.

After each ablation grid finishes (e.g. multi-seed = 5 runs), call the matching
build_table_*() function here to compress N archived cell directories into a
single comparison CSV ready for Bảng 4.5.x in main_vi v7.

Source of truth:
- outputs/_ablation_archive/<cell_name>/ created by ablation.core._snapshot_outputs
- Each archive contains:
    - manifest.json     (cell_name, run_id, ablation_type, notes_kv)
    - config.json       (copy of logs/run_<TS>.json — full §11.4 sidecar)
    - comparison.csv    (pre-aggregated global vs per_query metrics)
    - perquery_cf_metrics.csv  (per-instance metrics; aggregator means it)
    - global_cf_metrics.csv
    - per_feature.csv

Output CSVs go to outputs/ablation_<name>_table.csv per Generalrule v38 §11.4.

v4 (16/05/2026): rewritten against _ablation_archive/. Drops the 2-pass
marker-vs-fallback filter — the manifest.ablation_type field is now the single
source of truth, written at snapshot time by core.py.

Backward compat: NONE. The old aggregator (reading outputs/run_*.json) never
worked because the pipeline writes JSONs to logs/ not outputs/. Anyone with
historical outputs/run_*.json files predating the snapshot system is on the
v3 codepath and should re-run their ablations.

Usage (run from repo root):

    python -m ablation.aggregate seed       # build Bảng 4.5.1
    python -m ablation.aggregate method     # build Bảng 4.5.2
    python -m ablation.aggregate n_cf       # build Bảng 4.5.3
    python -m ablation.aggregate class      # build Bảng 4.5.4
    python -m ablation.aggregate taxonomy   # build Bảng 4.5.5
    python -m ablation.aggregate all        # build all 5 in one go
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
ARCHIVE_ROOT = OUTPUTS_DIR / "_ablation_archive"


# ──────────────────────────────────────────────────────────────────────
# Archive discovery + loading
# ──────────────────────────────────────────────────────────────────────
ArchivedCell = Tuple[str, Path, Dict[str, Any]]
# (cell_name, archive_dir, manifest_dict)


def list_archived_cells() -> List[ArchivedCell]:
    """Scan outputs/_ablation_archive/* for per-cell snapshots.

    Returns list sorted by cell_name. Cells without a manifest.json are skipped
    with a warning (they may be partial snapshots from a crashed run).
    """
    if not ARCHIVE_ROOT.exists():
        return []
    cells: List[ArchivedCell] = []
    for cell_dir in sorted(ARCHIVE_ROOT.iterdir()):
        if not cell_dir.is_dir():
            continue
        manifest_path = cell_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"  [WARN] skipping {cell_dir.name}: no manifest.json")
            continue
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"  [WARN] skipping {cell_dir.name}: manifest unreadable ({e})")
            continue
        cells.append((cell_dir.name, cell_dir, manifest))
    return cells


def filter_by_ablation_type(cells: List[ArchivedCell], ablation_type: str) -> List[ArchivedCell]:
    """Keep only cells whose manifest.ablation_type matches."""
    return [c for c in cells if c[2].get("ablation_type") == ablation_type]


def load_cell_config(cell_dir: Path) -> Dict[str, Any]:
    """Read the cell's copied §11.4 config sidecar (config.json)."""
    config_path = cell_dir / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def aggregate_perquery_metrics(cell_dir: Path) -> Dict[str, float]:
    """Mean of all per-instance metric columns in perquery_cf_metrics.csv.

    Drops the 'i' and 'n_cfs' index/count columns. NaN entries (failed CFs)
    are skipped via pandas default mean behavior.
    """
    csv = cell_dir / "perquery_cf_metrics.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    drop_cols = [c for c in ("i", "n_cfs") if c in df.columns]
    return df.drop(columns=drop_cols).mean(numeric_only=True).to_dict()


# ──────────────────────────────────────────────────────────────────────
# Per-ablation table builders
# ──────────────────────────────────────────────────────────────────────
def build_table_seed(cells: List[ArchivedCell]) -> pd.DataFrame:
    """Bảng 4.5.1 — multi-seed: 1 row per seed + MEAN/STD summary rows."""
    selected = filter_by_ablation_type(cells, "seed")
    rows = []
    for cell_name, cell_dir, manifest in selected:
        cfg = load_cell_config(cell_dir)
        hyp = cfg.get("hyperparameters", {})
        clf = cfg.get("classifier_metrics") or {}
        agg = aggregate_perquery_metrics(cell_dir)
        rows.append({
            "run_id": cfg.get("run_id"),
            "cell_name": cell_name,
            "seed": hyp.get("xgboost", {}).get("random_state"),
            "AUC": clf.get("AUC_ROC"),
            "validity": agg.get("validity"),
            "actionability": agg.get("actionability"),
            "wrong_dir_violations": agg.get("wrong_direction_violations"),
            "immutable_violations": agg.get("immutable_violations"),
            "sparsity": agg.get("sparsity"),
            "proximity_L1": agg.get("proximity_L1"),
            "diversity": agg.get("diversity"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    if len(df) >= 2:
        numeric_cols = df.select_dtypes(include="number").columns.drop("seed", errors="ignore")
        summary = pd.DataFrame({
            "run_id": ["MEAN", "STD"],
            "cell_name": ["—", "—"],
            "seed": ["—", "—"],
            **{c: [df[c].mean(), df[c].std()] for c in numeric_cols},
        })
        df = pd.concat([df, summary], ignore_index=True)
    return df


def build_table_method(cells: List[ArchivedCell]) -> pd.DataFrame:
    """Bảng 4.5.2 — method comparison: 1 row per method (random/genetic/kdtree)."""
    selected = filter_by_ablation_type(cells, "method")
    rows = []
    for cell_name, cell_dir, manifest in selected:
        cfg = load_cell_config(cell_dir)
        hyp = cfg.get("hyperparameters", {})
        clf = cfg.get("classifier_metrics") or {}
        agg = aggregate_perquery_metrics(cell_dir)
        rows.append({
            "run_id": cfg.get("run_id"),
            "cell_name": cell_name,
            "method": hyp.get("dice", {}).get("method"),
            "AUC": clf.get("AUC_ROC"),
            "validity": agg.get("validity"),
            "actionability": agg.get("actionability"),
            "wrong_dir_violations": agg.get("wrong_direction_violations"),
            "sparsity": agg.get("sparsity"),
            "proximity_L1": agg.get("proximity_L1"),
            "diversity": agg.get("diversity"),
            "plausibility_kNN50": agg.get("plausibility_kNN50"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def build_table_n_cf(cells: List[ArchivedCell]) -> pd.DataFrame:
    """Bảng 4.5.3 — n_counterfactuals sweep: 1 row per n_cf value."""
    selected = filter_by_ablation_type(cells, "n_cf")
    rows = []
    for cell_name, cell_dir, manifest in selected:
        cfg = load_cell_config(cell_dir)
        hyp = cfg.get("hyperparameters", {})
        agg = aggregate_perquery_metrics(cell_dir)
        rows.append({
            "run_id": cfg.get("run_id"),
            "cell_name": cell_name,
            "n_counterfactuals": hyp.get("dice", {}).get("n_counterfactuals"),
            "validity": agg.get("validity"),
            "actionability": agg.get("actionability"),
            "wrong_dir_violations": agg.get("wrong_direction_violations"),
            "sparsity": agg.get("sparsity"),
            "diversity": agg.get("diversity"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("n_counterfactuals").reset_index(drop=True)


def build_table_class(cells: List[ArchivedCell]) -> pd.DataFrame:
    """Bảng 4.5.4 — class balance: 1 row per risk-threshold cohort.

    risk_threshold sourced from manifest.notes_kv['class_threshold'] (string).
    """
    selected = filter_by_ablation_type(cells, "class")
    rows = []
    for cell_name, cell_dir, manifest in selected:
        cfg = load_cell_config(cell_dir)
        notes_kv = manifest.get("notes_kv", {}) or {}
        threshold = notes_kv.get("class_threshold", "?")
        agg = aggregate_perquery_metrics(cell_dir)
        rows.append({
            "run_id": cfg.get("run_id"),
            "cell_name": cell_name,
            "risk_threshold": threshold,
            "validity": agg.get("validity"),
            "actionability": agg.get("actionability"),
            "wrong_dir_violations": agg.get("wrong_direction_violations"),
            "sparsity": agg.get("sparsity"),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("risk_threshold").reset_index(drop=True)


def build_table_taxonomy(cells: List[ArchivedCell]) -> pd.DataFrame:
    """Bảng 4.5.5 — taxonomy granularity: rows = taxonomy variants.

    Supports 3 known variants:
    - 5-class default (taxonomy_n_classes=5)
    - 4-class collapsed (taxonomy_n_classes=4, DiffWalk→MONOTONIC_DOWN)
    - 3-class conservative (taxonomy_n_classes=3, Income/Education/AnyHealthcare→IMMUTABLE)

    Reads ablation parameter values from manifest.notes_kv (parsed at snapshot
    time by core.py from the pipeline's notes_suffix string).
    """
    selected = filter_by_ablation_type(cells, "taxonomy")
    rows = []
    for cell_name, cell_dir, manifest in selected:
        cfg = load_cell_config(cell_dir)
        notes_kv = manifest.get("notes_kv", {}) or {}
        n_classes = notes_kv.get("taxonomy_n_classes", "?")
        variant = notes_kv.get("variant", "default")
        agg = aggregate_perquery_metrics(cell_dir)
        rows.append({
            "run_id": cfg.get("run_id"),
            "cell_name": cell_name,
            "taxonomy_n_classes": n_classes,
            "variant": variant,
            "validity": agg.get("validity"),
            "actionability": agg.get("actionability"),
            "wrong_dir_violations": agg.get("wrong_direction_violations"),
            "immutable_violations": agg.get("immutable_violations"),
        })
    if not rows:
        return pd.DataFrame()
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


def main(ablation_type: str) -> Optional[Path]:
    """Build the table for one ablation type, save CSV to outputs/."""
    if ablation_type not in BUILDERS:
        raise ValueError(f"Unknown ablation type: {ablation_type}. Valid: {list(BUILDERS)}")
    cells = list_archived_cells()
    if not cells:
        print(f"⚠ No archived cells found in {ARCHIVE_ROOT}")
        print("  Run `python run_ablation_all.py` (or --smoke) first to populate the archive.")
        return None
    df = BUILDERS[ablation_type](cells)
    if df.empty:
        print(f"⚠ No matching cells found for ablation_type={ablation_type}")
        print(f"  Found {len(cells)} archived cells; types present: "
              f"{sorted(set((c[2].get('ablation_type') or '?') for c in cells))}")
        return None
    out_csv = OUTPUTS_DIR / f"ablation_{ablation_type}_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"✓ Built table: {out_csv.relative_to(REPO_ROOT)}")
    print(df.to_string(index=False))
    return out_csv


def main_all() -> Dict[str, Optional[Path]]:
    """Build all 5 tables. Best-effort each. Returns map ablation_type → out path."""
    results = {}
    for atype in BUILDERS:
        print()
        print("=" * 60)
        print(f"=== Building table: {atype}")
        print("=" * 60)
        try:
            results[atype] = main(atype)
        except Exception as e:
            print(f"  [WARN] {atype} failed: {type(e).__name__}: {e}")
            results[atype] = None
    return results


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is None:
        print(f"Usage: python -m ablation.aggregate {'|'.join(BUILDERS)}|all")
        sys.exit(1)
    if arg == "all":
        main_all()
    else:
        main(arg)
