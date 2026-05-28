"""Generate all figures from the latest run's outputs.

Two ways to invoke:
1. Auto-called by run_main.py at end of pipeline.
2. Standalone: `python analysis/make_figures.py` (right-click -> Run in
   PyCharm cũng được). Regen figures cho LATEST run mà không re-pipeline.

Convention: outputs/ reflects the latest run
only — KHÔNG có run_id prefix trên filename. Mỗi run overwrites previous
contents. Audit trail của các run cũ nằm ở logs/run_{ts}.{log,json}.
"""
from __future__ import annotations

import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# This file lives in analysis/ (one level deep), so repo root = parent.parent
# ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from analysis.figures import comparison_metrics, per_feature_actionability


def _require_csv(outputs_dir: Path, name: str) -> Path:
    """Resolve a CSV in outputs/ by exact filename."""
    csv_path = outputs_dir / name
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Not found: {csv_path}. "
            "Run the pipeline with compare_modes=true first (configs/default.yaml)."
        )
    return csv_path


def main() -> None:
    """Generate all figures from the latest run in outputs/."""
    outputs_dir = REPO_ROOT / "outputs"
    per_feature_csv = _require_csv(outputs_dir, "per_feature.csv")
    comparison_csv  = _require_csv(outputs_dir, "comparison.csv")

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
