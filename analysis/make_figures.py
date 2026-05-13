"""Generate all figures from a run's outputs.

Two ways to invoke:
1. Auto-called by run_main.py at end of pipeline.
2. Standalone: `python analysis/make_figures.py` (right-click → Run trong
   PyCharm cũng được). Regen figures cho LATEST run mà không re-pipeline.

Không nhận tham số CLI — luôn dùng run mới nhất trong outputs/.
Nếu cần target run cụ thể, edit RUN_ID_OVERRIDE bên dưới.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Override: hardcode 1 run_id cụ thể nếu cần regen figure cho run cũ.
# None = dùng run mới nhất (default).
# ──────────────────────────────────────────────────────────────────────
RUN_ID_OVERRIDE: Optional[str] = None
# Ví dụ: RUN_ID_OVERRIDE = "run_20260513_1710"


# ──────────────────────────────────────────────────────────────────────
# §11.5 — file này ở analysis/ (1 level deep) nên repo root = parent.parent
# ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from analysis.figures import per_feature_actionability


def find_latest_per_feature_csv(outputs_dir: Path) -> Path:
    """Return path to most recent run_*_per_feature.csv (lexicographic sort
    on filename = datetime sort, since YYYYMMDD_HHMM is sortable)."""
    candidates = sorted(outputs_dir.glob("run_*_per_feature.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No run_*_per_feature.csv in {outputs_dir}. "
            "Cần chạy pipeline với compare_modes=true trước (configs/default.yaml)."
        )
    return candidates[-1]


def find_csv_for_run(outputs_dir: Path, run_id: str) -> Path:
    """Look up the per_feature CSV for an explicit run_id."""
    csv_path = outputs_dir / f"{run_id}_per_feature.csv"
    if not csv_path.exists():
        available = sorted(outputs_dir.glob("run_*_per_feature.csv"))
        msg = f"Not found: {csv_path}"
        if available:
            msg += f"\nAvailable runs in {outputs_dir}:\n  "
            msg += "\n  ".join(p.name for p in available)
        raise FileNotFoundError(msg)
    return csv_path


def main() -> None:
    """Generate figures from latest run (or RUN_ID_OVERRIDE if set)."""
    outputs_dir = REPO_ROOT / "outputs"

    if RUN_ID_OVERRIDE is not None:
        csv_path = find_csv_for_run(outputs_dir, RUN_ID_OVERRIDE)
    else:
        csv_path = find_latest_per_feature_csv(outputs_dir)

    print(f"[make_figures] Repo root: {REPO_ROOT}")
    print(f"[make_figures] Input CSV: {csv_path}")
    print()

    # ── Figure: per-feature actionability ──
    print("[make_figures] Per-feature actionability...")
    result = per_feature_actionability.generate(csv_path)
    print(f"  PNG: {result['png']}")
    print(f"  PDF: {result['pdf']}")

    # Future figures: add invocation here.
    # Mỗi figure module export generate(csv_path) -> {'png': Path, 'pdf': Path}
    # e.g.
    #   from analysis.figures import comparison_metrics
    #   comparison_metrics.generate(comparison_csv)

    print()
    print("[make_figures] Done.")


if __name__ == "__main__":
    main()
