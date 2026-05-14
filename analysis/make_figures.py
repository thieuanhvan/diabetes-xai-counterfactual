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


from analysis.figures import comparison_metrics, per_feature_actionability


def find_latest_csv(outputs_dir: Path, suffix: str) -> Path:
    """Lexicographic sort on filename = datetime sort (YYYYMMDD_HHMM sortable).

    Args:
        outputs_dir: outputs/ folder.
        suffix: e.g. '_per_feature.csv' or '_comparison.csv'.
    """
    candidates = sorted(outputs_dir.glob(f"run_*{suffix}"))
    if not candidates:
        raise FileNotFoundError(
            f"No run_*{suffix} in {outputs_dir}. "
            "Cần chạy pipeline với compare_modes=true trước (configs/default.yaml)."
        )
    return candidates[-1]


def find_csv_for_run(outputs_dir: Path, run_id: str, suffix: str) -> Path:
    """Look up a specific CSV by run_id and suffix."""
    csv_path = outputs_dir / f"{run_id}{suffix}"
    if not csv_path.exists():
        available = sorted(outputs_dir.glob(f"run_*{suffix}"))
        msg = f"Not found: {csv_path}"
        if available:
            msg += f"\nAvailable runs in {outputs_dir}:\n  "
            msg += "\n  ".join(p.name for p in available)
        raise FileNotFoundError(msg)
    return csv_path


def _resolve_csv(outputs_dir: Path, suffix: str) -> Path:
    """Resolve a CSV path honouring RUN_ID_OVERRIDE."""
    if RUN_ID_OVERRIDE is not None:
        return find_csv_for_run(outputs_dir, RUN_ID_OVERRIDE, suffix)
    return find_latest_csv(outputs_dir, suffix)


def main() -> None:
    """Generate all figures from latest run (or RUN_ID_OVERRIDE if set)."""
    outputs_dir = REPO_ROOT / "outputs"
    per_feature_csv = _resolve_csv(outputs_dir, "_per_feature.csv")
    comparison_csv  = _resolve_csv(outputs_dir, "_comparison.csv")

    print(f"[make_figures] Repo root: {REPO_ROOT}")
    print(f"[make_figures] Per-feature CSV: {per_feature_csv}")
    print(f"[make_figures] Comparison CSV:  {comparison_csv}")
    print()

    # ── Figure: per-feature actionability ──
    print("[make_figures] Per-feature actionability...")
    result = per_feature_actionability.generate(per_feature_csv)
    print(f"  PNG: {result['png']}")
    print(f"  PDF: {result['pdf']}")
    print()

    # ── Figure: comparison metrics (rel_delta + abs side-by-side) ──
    print("[make_figures] Comparison metrics...")
    result = comparison_metrics.generate(comparison_csv)
    print(f"  PNG: {result['png']}")
    print(f"  PDF: {result['pdf']}")
    print()

    # Future figures: add invocation here.
    # Mỗi figure module export generate(csv_path) -> {'png': Path, 'pdf': Path}

    print("[make_figures] Done.")


if __name__ == "__main__":
    main()
