"""Pipeline entry-point wrapper. Right-click -> Run in PyCharm.

After the pipeline finishes, two analysis steps are invoked automatically:
1. analysis/make_figures.py    — generate figures (PNG + PDF) for the latest run.
2. analysis/topk_violations.py — rank features by violation-rate reduction
   (CSV + Markdown table).

Each analysis step is best-effort: the pipeline outputs remain valid even if a
step fails (e.g. if matplotlib is not installed, figure generation is skipped
but topk still runs).

No CLI arguments are accepted — all configuration is fixed in code.
- Pipeline config: configs/default.yaml
- Figure generation: always-on (best-effort, skipped if matplotlib missing)
- Topk violations:   always-on (best-effort, skipped if pandas missing)
"""
from __future__ import annotations

import sys
from pathlib import Path


# This wrapper lives at the repo root, so repo root = the file's own parent.
# No walk-up is needed; we do not check for src/ or outputs/ existence here.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.pipelines.main import main as run_pipeline


def _maybe_run_figures() -> None:
    """Best-effort figure generation after the pipeline. Pipeline outputs stay
    valid if figure generation fails (e.g. matplotlib not installed)."""
    try:
        from analysis.make_figures import main as make_figures
    except ImportError as e:
        print(f"\n[run_main] Skip figures (import failed): {e}")
        print("[run_main]   -> Install matplotlib: pip install matplotlib")
        return

    print()
    print("=" * 60)
    print("[run_main] Generating figures from latest run...")
    print("=" * 60)
    try:
        make_figures()
    except Exception as e:
        print(f"\n[run_main] Figure generation failed: {type(e).__name__}: {e}")
        print("[run_main] Pipeline outputs (CSV/JSON/log) still valid.")
        print("[run_main] Regenerate figures: python analysis/make_figures.py")


def _maybe_run_topk() -> None:
    """Best-effort topk_violations ranking after the pipeline. Pipeline outputs
    stay valid if the step fails."""
    try:
        from analysis.topk_violations import main as run_topk
    except ImportError as e:
        print(f"\n[run_main] Skip topk_violations (import failed): {e}")
        return

    print()
    print("=" * 60)
    print("[run_main] Generating topk_violations ranking from latest run...")
    print("=" * 60)
    try:
        run_topk()
    except Exception as e:
        print(f"\n[run_main] Topk_violations failed: {type(e).__name__}: {e}")
        print("[run_main] Pipeline outputs (CSV/JSON/log) still valid.")
        print("[run_main] Regenerate ranking: python analysis/topk_violations.py")


if __name__ == "__main__":
    run_pipeline("configs/default.yaml")
    _maybe_run_figures()
    _maybe_run_topk()
