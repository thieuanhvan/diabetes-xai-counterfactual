"""Pipeline entry §11.5 wrapper. Right-click → Run trong PyCharm.

Sau khi pipeline xong, tự động gọi 2 analysis steps:
1. analysis/make_figures.py — gen figures (PNG + PDF) cho run vừa chạy.
2. analysis/topk_violations.py — rank features by violation_rate reduction
   (CSV + Markdown table).

Mỗi analysis step best-effort: pipeline outputs vẫn valid nếu step thất bại
(e.g. matplotlib chưa cài → skip figures, vẫn chạy topk).

Không nhận tham số CLI — mọi config hardcoded.
- Pipeline config: configs/default.yaml
- Figure generation:  always-on (best-effort, skip nếu matplotlib missing)
- Topk violations:    always-on (best-effort, skip nếu pandas missing)
"""
from __future__ import annotations

import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# §11.5 — wrapper ở REPO ROOT nên repo root = parent của file này.
# Không walk-up: không cần check src/ hoặc outputs/ tồn tại.
# ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.pipelines.main import main as run_pipeline


def _maybe_run_figures() -> None:
    """Best-effort: gen figures sau pipeline. Pipeline outputs vẫn valid
    nếu figure generation thất bại (e.g. matplotlib chưa cài)."""
    try:
        from analysis.make_figures import main as make_figures
    except ImportError as e:
        print(f"\n[run_main] Skip figures (import failed): {e}")
        print("[run_main]   → Cài matplotlib: pip install matplotlib")
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
        print("[run_main] Regen figures: python analysis/make_figures.py")


def _maybe_run_topk() -> None:
    """Best-effort: gen topk_violations ranking sau pipeline. Pipeline outputs
    vẫn valid nếu step thất bại."""
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
        print("[run_main] Regen ranking: python analysis/topk_violations.py")


if __name__ == "__main__":
    run_pipeline("configs/default.yaml")
    _maybe_run_figures()
    _maybe_run_topk()
