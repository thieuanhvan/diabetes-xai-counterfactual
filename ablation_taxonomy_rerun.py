"""Rerun 3 taxonomy ablation cells at n=200 with bug-fixed scoring code.

Right-click -> Run in PyCharm (no CLI args, no walk-up).

Each cell auto-snapshots into outputs/_ablation_archive/<cell_name>/,
overwriting the previous snapshot. The other 14 cells (class, n_cf, seed,
method) are left untouched -- they remain in _ablation_archive/ from the
previous run (the bug fix does not affect those cells because it is not
triggered by their flags).

Wall-clock estimate: ~30 min total (3 cells x ~10 min on an 8-core machine).

Sequence:
  1. taxonomy_5class              — sanity check (numbers must match manuscript)
  2. taxonomy_4class              — numbers change (actionability up, wrong_dir -> 0)
  3. taxonomy_3class_conservative — aggregate unchanged, per_feature labels change
"""
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from ablation.core import run_grid


def _grid_taxonomy_rerun():
    """3 taxonomy cells with n=200 (override default ablation_taxonomy_*.yaml).

    Each cell rebuilds configs/ablation_<name>.yaml from default.yaml + these
    overrides (default.yaml already has n_test_instances=200), so n=20 in the
    existing taxonomy ablation configs is implicitly overridden by the default.
    """
    return [
        ("taxonomy_5class", {
            "taxonomy": {
                "conditional_class_disabled": False,
                "socioeconomic_proxies_immutable": False,
            },
            "run": {
                "notes_suffix": "ablation=taxonomy; taxonomy_n_classes=5; rerun_post_bugfix",
                "is_scratch": False,
            },
        }),
        ("taxonomy_4class", {
            "taxonomy": {
                "conditional_class_disabled": True,
                "socioeconomic_proxies_immutable": False,
            },
            "run": {
                "notes_suffix": "ablation=taxonomy; taxonomy_n_classes=4; rerun_post_bugfix",
                "is_scratch": False,
            },
        }),
        ("taxonomy_3class_conservative", {
            "taxonomy": {
                "conditional_class_disabled": False,
                "socioeconomic_proxies_immutable": True,
            },
            "run": {
                "notes_suffix": "ablation=taxonomy; taxonomy_n_classes=3; variant=conservative; rerun_post_bugfix",
                "is_scratch": False,
            },
        }),
    ]


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("[rerun] Taxonomy ablation rerun — post bug fix (17/05/2026)")
    print("=" * 60)
    print("Cells: taxonomy_5class, taxonomy_4class, taxonomy_3class_conservative")
    print("Each cell snapshots to outputs/_ablation_archive/<cell>/")
    print("Wall-clock estimate: ~30 min total (3 × ~10 min)")
    print("=" * 60)
    run_grid("taxonomy", _grid_taxonomy_rerun())
    print()
    print("=" * 60)
    print("[rerun] Done. Next: right-click run_ablation_aggregate.py to regenerate")
    print("        outputs/ablation_taxonomy_table.csv with updated numbers.")
    print("=" * 60)
