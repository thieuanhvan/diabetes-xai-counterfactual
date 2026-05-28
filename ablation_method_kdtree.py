"""Rerun ONLY the kdtree cell of the method ablation, then rebuild the method
table. Use after applying the float64 upcast workaround in
src/pipelines/counterfactual/dice_runner.py (DiCE 0.12 + pandas 3.x kdtree
dtype bug).

The random and genetic cells of the method ablation completed successfully in
the 2026-05-15 16:48 ablation_all session, so we don't rerun them — their
JSON sidecars in outputs/ are still picked up by build_table_method which
filters by cfg.hyperparameters.dice.method.

Wall-clock estimate: ~20-30 min (kdtree is fast per DiCE; 200 queries x ~1.3s
per query x 2 modes ≈ 8-9 min + setup + posthoc).

Top-level wrapper.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from ablation.core import run_grid
from ablation.aggregate import main as build_table


def _grid_method_kdtree_only():
    return [
        ("method_kdtree", {"dice": {"method": "kdtree"}}),
    ]


if __name__ == "__main__":
    print("█" * 70)
    print("█  ablation_method_kdtree.py — Rerun kdtree cell only")
    print("█" * 70)
    print("  Reason: dice_runner.py float64 upcast workaround applied for")
    print("          DiCE 0.12 + pandas 3.x kdtree dtype bug.")
    print("  Random + genetic cells already complete (re-used from prior run).")
    print()

    t0 = time.time()
    try:
        run_grid("method", _grid_method_kdtree_only())
        elapsed_min = (time.time() - t0) / 60
        print(f"\n[ablation_method_kdtree] ✓ kdtree complete in {elapsed_min:.1f} min")
    except Exception as e:
        elapsed_min = (time.time() - t0) / 60
        print(f"\n[ablation_method_kdtree] ✗ kdtree FAILED after {elapsed_min:.1f} min")
        print(f"[ablation_method_kdtree]   {type(e).__name__}: {e}")
        sys.exit(1)

    print()
    print("█" * 70)
    print("█  REBUILD method table")
    print("█" * 70)
    try:
        build_table("method")
        print(f"\n[ablation_method_kdtree] ✓ outputs/ablation_method_table.csv rebuilt")
    except Exception as e:
        print(f"[ablation_method_kdtree] aggregate method failed: {type(e).__name__}: {e}")
        sys.exit(2)

    total_elapsed_min = (time.time() - t0) / 60
    print()
    print(f"[ablation_method_kdtree] Done. Total wall-clock: {total_elapsed_min:.1f} min")
