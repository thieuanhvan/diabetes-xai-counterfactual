"""Right-click runnable entry for PyCharm.

Equivalent to: python -m src.main --config configs/default.yaml

Place at REPO ROOT (cùng cấp configs/, src/, README.md).
Auto-detects repo root by walking up to find configs/default.yaml,
so still works if accidentally placed in a subdirectory.
"""
from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Walk up from `start` until a directory containing configs/default.yaml is found."""
    cur = start.resolve()
    while True:
        if (cur / "configs" / "default.yaml").exists():
            return cur
        if cur == cur.parent:  # filesystem root reached
            raise RuntimeError(
                f"Could not find repo root (no configs/default.yaml) starting from {start}"
            )
        cur = cur.parent


ROOT = find_repo_root(Path(__file__).parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import main

if __name__ == "__main__":
    main(str(ROOT / "configs" / "default.yaml"))