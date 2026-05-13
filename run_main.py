"""Wrapper to launch the diabetes-xai-counterfactual pipeline.

Per generalrule §11.5: run_*.py wrapper at REPO ROOT.
Defensive: find_repo_root walks up from __file__ until configs/default.yaml found.
"""
from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Walk up from `start` to find directory containing configs/default.yaml."""
    cur = start.resolve()
    for _ in range(10):  # safety bound
        if (cur / "configs" / "default.yaml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise RuntimeError(
        f"Cannot find repo root with configs/default.yaml starting from {start}"
    )


ROOT = find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(ROOT))

from src.main import main

if __name__ == "__main__":
    main(str(ROOT / "configs" / "default.yaml"))