#!/usr/bin/env python3
"""
cleanup_outputs.py - repo outputs/ hygiene one-shot.

Resolves the inconsistency between top-level outputs/{comparison,*}.csv
(broken numbers that contradict the manuscript) and the authoritative
archive at outputs/_ablation_archive/class_general/ (matches manuscript
Tables 4-5: validity 0.752->0.808, actionability 0.6655->0.9880).

Actions (idempotent, non-destructive):
0. If outputs/archive/manuscript_v16/ exists (from an earlier revision of
   this script), migrate it to outputs/archive/manuscript/.
1. Create outputs/archive/manuscript/ and populate from class_general
   archive cell.
2. Quarantine broken top-level CSVs/figures to outputs/_quarantine_pre_cleanup/.
3. Quarantine outputs_pre_bugfix_20260517_1536/ folder (redundant).
4. Leave _ablation_archive/ intact (all 18 cells preserved).

Run from repo root:
    python cleanup_outputs.py

Nothing is deleted; everything questionable is moved into
outputs/_quarantine_pre_cleanup/<timestamp>/, recoverable.
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS = REPO_ROOT / "outputs"
ARCHIVE = OUTPUTS / "_ablation_archive"
CANONICAL_CELL = ARCHIVE / "class_general"
FINAL_DIR = OUTPUTS / "archive" / "manuscript"
# Legacy path from earlier cleanup script revision (no longer canonical).
# If present, migrated to FINAL_DIR at script start.
LEGACY_FINAL_DIR = OUTPUTS / "archive" / "manuscript_v16"
QUARANTINE = OUTPUTS / "_quarantine_pre_cleanup"
PRE_BUGFIX = REPO_ROOT / "outputs_pre_bugfix_20260517_1536"

CANONICAL_FILES = [
    "comparison.csv",
    "global_cf_metrics.csv",
    "perquery_cf_metrics.csv",
    "per_feature.csv",
    "config.json",
    "manifest.json",
]

# top-level outputs files to quarantine (broken or non-canonical scratch)
# ablation_*_table.csv files are KEPT at top level: they aggregate _ablation_archive/
# and are valid summaries, not scratch artefacts.
TOP_LEVEL_TO_QUARANTINE = [
    "comparison.csv",
    "global_cf_metrics.csv",
    "perquery_cf_metrics.csv",
    "per_feature.csv",
    "topk_violations.csv",
    "topk_violations.md",
    "fig_comparison_metrics.png",
    "fig_comparison_metrics.pdf",
    "fig_per_feature.png",
    "fig_per_feature.pdf",
]


def main():
    if not OUTPUTS.exists():
        print(f"ERROR: {OUTPUTS} not found. Run from repo root.", file=sys.stderr)
        sys.exit(1)
    if not CANONICAL_CELL.exists():
        print(f"ERROR: canonical cell {CANONICAL_CELL} not found.", file=sys.stderr)
        print(f"       Expected the class_general ablation cell to exist.", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # 0. migrate legacy outputs/archive/manuscript_v16/ if present
    if LEGACY_FINAL_DIR.exists() and not FINAL_DIR.exists():
        print(f"[0/3] Migrating legacy outputs/archive/manuscript_v16/ -> outputs/archive/manuscript/")
        FINAL_DIR.parent.mkdir(parents=True, exist_ok=True)
        LEGACY_FINAL_DIR.rename(FINAL_DIR)
        print(f"        renamed (atomic)")
    elif LEGACY_FINAL_DIR.exists() and FINAL_DIR.exists():
        # both exist — keep new canonical, quarantine old
        print(f"[0/3] Quarantining redundant outputs/archive/manuscript_v16/ (new canonical already present)")
        legacy_target = QUARANTINE / f"manuscript_v16_legacy_{timestamp}"
        QUARANTINE.mkdir(parents=True, exist_ok=True)
        shutil.move(str(LEGACY_FINAL_DIR), str(legacy_target))
        print(f"        moved to outputs/_quarantine_pre_cleanup/{legacy_target.name}/")

    # 1. populate outputs/archive/manuscript/ from class_general
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/3] Populating outputs/archive/manuscript/ from _ablation_archive/class_general/")
    for fname in CANONICAL_FILES:
        src = CANONICAL_CELL / fname
        dst = FINAL_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"        copied  {fname}")
        else:
            print(f"        SKIP    {fname} (not in canonical cell)")

    readme_path = FINAL_DIR / "README.md"
    readme_path.write_text(
        "# outputs/archive/manuscript\n\n"
        "Authoritative results reported in the manuscript "
        "(Tables 4 and 5, Sections 4.3 and 4.4) — frozen snapshot of the "
        "main experimental configuration.\n\n"
        "Source cell: `outputs/_ablation_archive/class_general/` "
        "(run_id `run_20260517_1716`, seed 42, BRFSS 2021 full cohort, "
        "main experimental configuration). The specific manuscript revision "
        "this snapshot corresponds to is recorded in `manifest.json` and "
        "`config.json` (library versions, hardware, git commit when "
        "applicable).\n\n"
        "Do not overwrite. Working/scratch runs from `python run_main.py` "
        "should write to `outputs/scratch/` (configure via "
        "`configs/default.yaml: paths.output_dir`).\n",
        encoding="utf-8",
    )
    print(f"        wrote   README.md")

    # 2. quarantine broken top-level files
    quarantine_dir = QUARANTINE / f"top_level_{timestamp}"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/3] Quarantining broken top-level files")
    print(f"        target: outputs/_quarantine_pre_cleanup/top_level_{timestamp}/")
    moved_count = 0
    for fname in TOP_LEVEL_TO_QUARANTINE:
        src = OUTPUTS / fname
        if src.exists():
            shutil.move(str(src), str(quarantine_dir / fname))
            print(f"        moved   {fname}")
            moved_count += 1
    print(f"        ({moved_count} files moved)")

    # 3. quarantine outputs_pre_bugfix folder
    print(f"\n[3/3] Quarantining outputs_pre_bugfix_20260517_1536/")
    if PRE_BUGFIX.exists():
        target = QUARANTINE / PRE_BUGFIX.name
        shutil.move(str(PRE_BUGFIX), str(target))
        print(f"        moved to outputs/_quarantine_pre_cleanup/{PRE_BUGFIX.name}/")
    else:
        print(f"        not present, skip")

    print()
    print("DONE. Repo outputs/ state:")
    print("  outputs/archive/manuscript/          <- authoritative, manuscript-aligned")
    print("  outputs/_ablation_archive/           <- 18 ablation cells, untouched")
    print("  outputs/ablation_*_table.csv         <- aggregated tables, untouched")
    print("  outputs/_quarantine_pre_cleanup/     <- recoverable; delete after verify")
    print()
    print("Verify:")
    print("  diff outputs/archive/manuscript/comparison.csv outputs/_ablation_archive/class_general/comparison.csv")
    print("  (expect: zero diff)")
    print()
    print("Recommended next steps:")
    print("  1. Replace README.md with the updated version (points at outputs/archive/manuscript/).")
    print("  2. Replace REPRODUCIBILITY.md with the updated version (KBS title, real env).")
    print("  3. Update configs/default.yaml: paths.output_dir = 'outputs/scratch'")
    print("     so future `python run_main.py` writes to scratch, not top level.")
    print("  4. After verifying steps 1-3, delete outputs/_quarantine_pre_cleanup/ manually.")


if __name__ == "__main__":
    main()
