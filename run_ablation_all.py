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

Total wall-clock estimate (reference hardware, all 5 ablations, n_test_instances=200):
- Ablation 5 taxonomy (2 runs):  ~20 min
- Ablation 4 class    (3 runs):  ~30 min
- Ablation 3 n_cf     (4 runs):  ~40 min
- Ablation 1 seed     (5 runs):  ~50 min
- Ablation 2 method   (3 runs):  ~3-4h  (genetic + kdtree slow)
─────────────────────────────────────
Total                  17 runs:  ~5-6h

Top-level wrapper.

v3 (16/05/2026): all 5 grids set notes_suffix with 'ablation=<type>' marker so
ablation.aggregate can filter strictly. Existing 'class_threshold=' and
'taxonomy_n_classes=' markers preserved as secondary markers in their grids.

v4 (16/05/2026): added --smoke flag (~5-10 min total). Runs only ablation
TAXONOMY (2 cells: 5class + 4class) with n_test_instances=20 override.
Verifies the end-to-end pipeline + per-cell snapshot + aggregator before
committing to the full ~5h run. Also added --clean to wipe
outputs/_ablation_archive/ before starting.

USAGE:
    python run_ablation_all.py --smoke           # ~5-10 min, taxonomy only
    python run_ablation_all.py --smoke --clean   # wipe archive first
    python run_ablation_all.py                   # full ~5-6h run
    python run_ablation_all.py --clean           # full run, fresh archive
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from ablation.core import run_grid, ARCHIVE_ROOT


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
            "taxonomy": {
                "conditional_class_disabled": False,
                "socioeconomic_proxies_immutable": False,
            },
            "run": {"notes_suffix": "ablation=taxonomy; taxonomy_n_classes=5"},
        }),
        ("taxonomy_4class", {
            "taxonomy": {
                "conditional_class_disabled": True,
                "socioeconomic_proxies_immutable": False,
            },
            "run": {"notes_suffix": "ablation=taxonomy; taxonomy_n_classes=4"},
        }),
        ("taxonomy_3class_conservative", {
            "taxonomy": {
                "conditional_class_disabled": False,
                "socioeconomic_proxies_immutable": True,
            },
            "run": {"notes_suffix": "ablation=taxonomy; taxonomy_n_classes=3; variant=conservative"},
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
    {"name": "taxonomy", "needs_taxonomy_flag": True,  "grid_fn": _grid_taxonomy, "est_min": 30},
    {"name": "class",    "needs_taxonomy_flag": True,  "grid_fn": _grid_class,    "est_min": 30},
    {"name": "n_cf",     "needs_taxonomy_flag": False, "grid_fn": _grid_n_cf,     "est_min": 40},
    {"name": "seed",     "needs_taxonomy_flag": False, "grid_fn": _grid_seed,     "est_min": 50},
    {"name": "method",   "needs_taxonomy_flag": False, "grid_fn": _grid_method,   "est_min": 240},
]


# ──────────────────────────────────────────────────────────────────────
# Smoke-mode helpers (v4)
# ──────────────────────────────────────────────────────────────────────
def _inject_smoke_override(grid):
    """Inject evaluate.n_test_instances=20 into each cell's overrides.

    Deep-merges with the cell's existing overrides — does NOT replace any
    other `evaluate.*` keys that were already set (e.g. class grid's
    risk_threshold_min would survive even if smoke ran the class ablation).
    """
    out = []
    for name, ov in grid:
        ov_new = dict(ov)
        ev_merged = dict(ov_new.get("evaluate", {}))
        ev_merged["n_test_instances"] = 20
        ov_new["evaluate"] = ev_merged
        out.append((name, ov_new))
    return out


def _clean_archive():
    """Wipe outputs/_ablation_archive/. Prints summary."""
    if ARCHIVE_ROOT.exists():
        n_cells = sum(1 for p in ARCHIVE_ROOT.iterdir() if p.is_dir())
        shutil.rmtree(ARCHIVE_ROOT)
        print(f"[run_ablation_all] cleaned archive: removed {n_cells} cell dir(s)")
    else:
        print(f"[run_ablation_all] no archive to clean ({ARCHIVE_ROOT} doesn't exist)")


# ──────────────────────────────────────────────────────────────────────
# Per-ablation dispatch + aggregate
# ──────────────────────────────────────────────────────────────────────
def _safe_run_ablation(spec: Dict, taxonomy_ok: bool, smoke: bool = False) -> Dict:
    """Run one ablation grid. Returns status dict with name + outcome + elapsed."""
    name = spec["name"]
    if spec["needs_taxonomy_flag"] and not taxonomy_ok:
        print(f"\n[run_ablation_all] SKIP {name} — taxonomy disable flag not available")
        print(f"[run_ablation_all]   (ensure feature_taxonomy.set_conditional_disabled exists)")
        return {"name": name, "status": "skipped_no_taxonomy_flag", "elapsed_min": 0}

    print()
    print("█" * 70)
    label = f"{name.upper()} [SMOKE]" if smoke else name.upper()
    print(f"█  ABLATION: {label}  (est. ~{spec['est_min']} min)")
    print("█" * 70)
    t0 = time.time()
    try:
        grid = spec["grid_fn"]()
        if smoke:
            grid = _inject_smoke_override(grid)
        run_grid(name, grid)
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


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--smoke", action="store_true",
        help="Quick 5-10 min sanity run: taxonomy ablation only (2 cells), n_test_instances=20.",
    )
    p.add_argument(
        "--clean", action="store_true",
        help="Wipe outputs/_ablation_archive/ before running.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    t_start = time.time()
    taxonomy_ok = _taxonomy_flag_available()

    if args.clean:
        _clean_archive()

    # Select which ablations to run: full = all 5; smoke = taxonomy only
    if args.smoke:
        ablations = [s for s in ABLATIONS if s["name"] == "taxonomy"]
        mode_banner = "SMOKE TEST (taxonomy only, n_test_instances=20)"
        est_total = "~5-10"
    else:
        ablations = list(ABLATIONS)
        mode_banner = "FULL RUN"
        est_total = f"~{sum(s['est_min'] for s in ablations if s['needs_taxonomy_flag'] is False or taxonomy_ok)}"

    print("█" * 70)
    print(f"█  run_ablation_all.py — Master Ablation Wrapper [{mode_banner}]")
    print("█" * 70)
    print(f"  Taxonomy disable flag: {'✓ available' if taxonomy_ok else '✗ unavailable — taxonomy/class will skip'}")
    print(f"  Estimated total wall-clock: {est_total} min")
    print(f"  Output: outputs/_ablation_archive/<cell>/ + outputs/ablation_<type>_table.csv")
    print()
    print(f"  Execution order (cheap-first):")
    for i, spec in enumerate(ablations, start=1):
        will_skip = spec["needs_taxonomy_flag"] and not taxonomy_ok
        marker = "SKIP" if will_skip else f"~{spec['est_min']} min"
        print(f"    {i}. {spec['name']:<10} ({marker})")
    print()

    results = []
    for spec in ablations:
        results.append(_safe_run_ablation(spec, taxonomy_ok, smoke=args.smoke))

    print()
    print("█" * 70)
    print("█  AGGREGATING TABLES")
    print("█" * 70)
    aggregate_ok: Dict[str, bool] = {}
    for r in results:
        if r["status"] == "ok":
            print(f"\n[run_ablation_all] Aggregating {r['name']} table...")
            aggregate_ok[r["name"]] = _safe_aggregate(r["name"])

    total_elapsed_min = (time.time() - t_start) / 60
    print()
    print("█" * 70)
    print(f"█  FINAL SUMMARY [{mode_banner}]")
    print("█" * 70)
    print(f"  Total wall-clock: {total_elapsed_min:.1f} min")
    print()
    print(f"  {'Ablation':<12} {'Run status':<22} {'Wall-clock':>12}  {'Aggregator':>12}")
    print(f"  {'-' * 12} {'-' * 22} {'-' * 12}  {'-' * 12}")
    for r in results:
        status_str = {"ok": "✓ complete", "failed": "✗ failed", "skipped_no_taxonomy_flag": "○ skipped (no flag)"}[r["status"]]
        elapsed_str = f"{r['elapsed_min']:.1f} min" if r["elapsed_min"] > 0 else "—"
        agg_str = "✓ table built" if aggregate_ok.get(r["name"]) else ("✗ no table" if r["status"] == "ok" else "—")
        print(f"  {r['name']:<12} {status_str:<22} {elapsed_str:>12}  {agg_str:>12}")
    print()
    print("  Output tables:")
    for r in results:
        if aggregate_ok.get(r["name"]):
            print(f"    outputs/ablation_{r['name']}_table.csv")
    print()
    print("  Archived cells:")
    if ARCHIVE_ROOT.exists():
        for cell_dir in sorted(ARCHIVE_ROOT.iterdir()):
            if cell_dir.is_dir():
                print(f"    outputs/_ablation_archive/{cell_dir.name}/")
    print()

    if args.smoke:
        # Smoke-test pass/fail summary
        all_ok = (
            len(results) > 0
            and all(r["status"] == "ok" for r in results)
            and all(aggregate_ok.get(r["name"]) for r in results)
        )
        if all_ok:
            print("[SMOKE] ✓ PASS — pipeline + snapshot + aggregator all working.")
            print("[SMOKE]   You can now run the full ablation: `python run_ablation_all.py`")
        else:
            print("[SMOKE] ✗ FAIL — fix issues above before committing to the full run.")
            sys.exit(1)
    else:
        print("[run_ablation_all] Done. Ablation summary tables written to outputs/.")
