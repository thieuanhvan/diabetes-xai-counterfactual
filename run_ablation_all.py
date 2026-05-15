"""Master wrapper: run ALL 5 ablations sequentially in one go.

Order optimized cheap-first → expensive-last so partial completion still
produces useful output. Each ablation grid is best-effort: a failure in one
does not block subsequent ablations.

Pre-flight: detects whether the taxonomy disable flag is available
(set_conditional_disabled in feature_taxonomy). If absent (e.g. someone has
forked main and removed the optional ablation hooks), the taxonomy and
class-balance ablations are skipped with a warning rather than crashing.

After all grids finish, builds 5 ablation_*_table.csv tables via
ablation.aggregate (best-effort each).

Total wall-clock estimate (Vân hardware, all 5 ablations):
- Ablation 5 taxonomy (2 runs):  ~20 min
- Ablation 4 class    (3 runs):  ~30 min
- Ablation 3 n_cf     (4 runs):  ~40 min
- Ablation 1 seed     (5 runs):  ~50 min
- Ablation 2 method   (3 runs):  ~3-4h  (genetic + kdtree slow)
─────────────────────────────────────
Total                  17 runs:  ~5-6h

Run overnight or while doing other work. Smoke-test 1 grid first if uncertain:
    python run_ablation_seed.py   # ~10 min single seed if SEEDS=[42] in that file

§11.5 wrapper.

v3 changes (16/05/2026): all 5 grids now set notes_suffix with 'ablation=<type>'
marker so ablation.aggregate can filter strictly. Existing 'class_threshold='
and 'taxonomy_n_classes=' markers preserved as secondary markers in their grids.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, List


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from ablation.core import run_grid


# ──────────────────────────────────────────────────────────────────────
# Pre-flight: optional ablation hooks (taxonomy disable + risk threshold)
# ──────────────────────────────────────────────────────────────────────
def _taxonomy_flag_available() -> bool:
    """Return True if the taxonomy disable flag is importable."""
    try:
        from src.pipelines.counterfactual.feature_taxonomy import set_conditional_disabled  # noqa: F401
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────
# Grid definitions — every grid sets 'ablation=<type>' notes_suffix marker
# ──────────────────────────────────────────────────────────────────────
def _grid_taxonomy():
    return [
        ("taxonomy_5class", {
            "taxonomy": {"conditional_class_disabled": False},
            "run": {"notes_suffix": "ablation=taxonomy; taxonomy_n_classes=5"},
        }),
        ("taxonomy_4class", {
            "taxonomy": {"conditional_class_disabled": True},
            "run": {"notes_suffix": "ablation=taxonomy; taxonomy_n_classes=4"},
        }),
    ]


def _grid_class():
    return [
        (f"class_{label}", {
            "evaluate": {
                "risk_threshold_min": threshold,
                "n_test_instances": 200,
            },
            "run": {"notes_suffix": f"ablation=class; class_threshold={threshold}"},
        })
        for label, threshold in [
            ("general",   0.0),
            ("at_risk",   0.5),
            ("high_risk", 0.7),
        ]
    ]


def _grid_n_cf():
    return [
        (f"n_cf_{n}", {
            "dice": {"n_counterfactuals": n},
            "run": {"notes_suffix": f"ablation=n_cf; n_cf_value={n}"},
        })
        for n in [1, 3, 5, 10]
    ]


def _grid_seed():
    return [
        (f"seed_{s}", {
            "random": {"seed": s},
            "xgboost": {"random_state": s},
            "run": {"notes_suffix": f"ablation=seed; seed_value={s}"},
        })
        for s in [42, 123, 2024, 7, 31337]
    ]


def _grid_method():
    return [
        (f"method_{m}", {
            "dice": {"method": m},
            "run": {"notes_suffix": f"ablation=method; method_value={m}"},
        })
        for m in ["random", "genetic", "kdtree"]
    ]


# Cheap-first ordering: partial completion still produces useful output.
ABLATIONS: List[Dict] = [
    {"name": "taxonomy", "needs_taxonomy_flag": True,  "grid_fn": _grid_taxonomy, "est_min": 20},
    {"name": "class",    "needs_taxonomy_flag": True,  "grid_fn": _grid_class,    "est_min": 30},
    {"name": "n_cf",     "needs_taxonomy_flag": False, "grid_fn": _grid_n_cf,     "est_min": 40},
    {"name": "seed",     "needs_taxonomy_flag": False, "grid_fn": _grid_seed,     "est_min": 50},
    {"name": "method",   "needs_taxonomy_flag": False, "grid_fn": _grid_method,   "est_min": 240},
]


def _safe_run_ablation(spec: Dict, taxonomy_ok: bool) -> Dict:
    """Run one ablation grid. Returns status dict with name + outcome + elapsed."""
    name = spec["name"]
    if spec["needs_taxonomy_flag"] and not taxonomy_ok:
        print(f"\n[run_ablation_all] SKIP {name} — taxonomy disable flag not available")
        print(f"[run_ablation_all]   (ensure feature_taxonomy.set_conditional_disabled exists)")
        return {"name": name, "status": "skipped_no_taxonomy_flag", "elapsed_min": 0}

    print()
    print("█" * 70)
    print(f"█  ABLATION: {name.upper()}  (est. ~{spec['est_min']} min)")
    print("█" * 70)
    t0 = time.time()
    try:
        run_grid(name, spec["grid_fn"]())
        elapsed_min = (time.time() - t0) / 60
        print(f"\n[run_ablation_all] ✓ {name} complete in {elapsed_min:.1f} min")
        return {"name": name, "status": "ok", "elapsed_min": elapsed_min}
    except Exception as e:
        elapsed_min = (time.time() - t0) / 60
        print(f"\n[run_ablation_all] ✗ {name} FAILED after {elapsed_min:.1f} min")
        print(f"[run_ablation_all]   {type(e).__name__}: {e}")
        print(f"[run_ablation_all]   continuing to next ablation...")
        return {"name": name, "status": "failed", "elapsed_min": elapsed_min, "error": str(e)}


def _safe_aggregate(name: str) -> bool:
    """Best-effort aggregate for one ablation. Returns True if table built."""
    try:
        from ablation.aggregate import main as build_table
        result = build_table(name)
        return result is not None
    except Exception as e:
        print(f"[run_ablation_all] aggregate {name} failed: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    t_start = time.time()
    taxonomy_ok = _taxonomy_flag_available()

    print("█" * 70)
    print("█  run_ablation_all.py — Master Ablation Wrapper")
    print("█" * 70)
    print(f"  Taxonomy disable flag: {'✓ available' if taxonomy_ok else '✗ unavailable — ablation 4+5 will skip'}")
    print(f"  Estimated total wall-clock: ~{sum(s['est_min'] for s in ABLATIONS if s['needs_taxonomy_flag'] is False or taxonomy_ok)} min")
    print(f"  Output: outputs/run_<timestamp>.{{log,json,csv}} per cell + outputs/ablation_<type>_table.csv per ablation")
    print()
    print(f"  Execution order (cheap-first):")
    for i, spec in enumerate(ABLATIONS, start=1):
        will_skip = spec["needs_taxonomy_flag"] and not taxonomy_ok
        marker = "SKIP" if will_skip else f"~{spec['est_min']} min"
        print(f"    {i}. {spec['name']:<10} ({marker})")
    print()

    results = []
    for spec in ABLATIONS:
        results.append(_safe_run_ablation(spec, taxonomy_ok))

    print()
    print("█" * 70)
    print("█  AGGREGATING TABLES")
    print("█" * 70)
    for r in results:
        if r["status"] == "ok":
            print(f"\n[run_ablation_all] Aggregating {r['name']} table...")
            _safe_aggregate(r["name"])

    total_elapsed_min = (time.time() - t_start) / 60
    print()
    print("█" * 70)
    print("█  FINAL SUMMARY")
    print("█" * 70)
    print(f"  Total wall-clock: {total_elapsed_min:.1f} min")
    print()
    print(f"  {'Ablation':<12} {'Status':<22} {'Wall-clock':>12}")
    print(f"  {'-' * 12} {'-' * 22} {'-' * 12}")
    for r in results:
        status_str = {"ok": "✓ complete", "failed": "✗ failed", "skipped_no_taxonomy_flag": "○ skipped (no taxonomy flag)"}[r["status"]]
        elapsed_str = f"{r['elapsed_min']:.1f} min" if r["elapsed_min"] > 0 else "—"
        print(f"  {r['name']:<12} {status_str:<22} {elapsed_str:>12}")
    print()
    print("  Output tables:")
    for r in results:
        if r["status"] == "ok":
            print(f"    outputs/ablation_{r['name']}_table.csv")
    print()
    print("[run_ablation_all] Done. Ready to integrate vào main_vi v7+ §4.5.x.")
