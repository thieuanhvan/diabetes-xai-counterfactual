"""Generate all figures from the latest run's outputs.

Two ways to invoke:
1. Auto-called by run_main.py at the end of the pipeline.
2. Standalone: `python analysis/make_figures.py` (right-click -> Run in
   PyCharm also works). Regenerates figures for the latest run without
   re-running the pipeline.

Convention: the pipeline output directory reflects the latest run only --
no run_id prefix on filenames; each run overwrites the previous contents.
The audit trail of past runs lives in logs/run_{ts}.{log,json}.
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


import yaml

from analysis.figures import comparison_metrics, per_feature_actionability


def _resolve_output_dir() -> Path:
    """Resolve the pipeline output directory from configs/default.yaml so
    figures are read from wherever the pipeline actually wrote them
    (matches src/pipelines/main.py). Falls back to outputs/scratch."""
    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        out = Path(cfg["paths"]["output_dir"])
        return out if out.is_absolute() else REPO_ROOT / out
    except Exception:
        return REPO_ROOT / "outputs" / "scratch"


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
    outputs_dir = _resolve_output_dir()
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
    # Each figure module exports generate(csv_path) -> {'png': Path, 'pdf': Path}

    print("[make_figures] Done.")


if __name__ == "__main__":
    main()
