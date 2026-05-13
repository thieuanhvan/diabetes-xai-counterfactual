"""Run logger + config JSON sidecar per generalrule §11.4 + §11.6 (v37).

§11.4 (v37): log + config JSON sidecar both emit to outputs/ (configs/ là INPUT
       only; logs/ retired). Schema: runtime/libraries/seeds/dataset/
       hyperparameters/git_commit/timestamps/notes.
§11.6: runtime field expanded with hardware sub-dict (cpu/cpu_count/ram_gb/gpu).
       Best-effort detection; degrades gracefully if psutil or nvidia-smi unavailable.

Scratch runs (is_scratch=True) get '_scratch' marker → auto-gitignored.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


VN_TZ = timezone(timedelta(hours=7))

_TRACKED_LIBS = [
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost",
    "shap", "dice-ml", "pyyaml", "tqdm", "joblib", "psutil",
]


def _git_commit_short(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _library_versions() -> Dict[str, str]:
    versions = {}
    for lib in _TRACKED_LIBS:
        try:
            versions[lib] = md.version(lib)
        except md.PackageNotFoundError:
            versions[lib] = "not-installed"
    return versions


def _hardware_info() -> Dict[str, Any]:
    """Best-effort hardware detection per §11.6. Degrade gracefully on failure."""
    info: Dict[str, Any] = {
        "cpu": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "ram_gb": None,
        "gpu": "none",
    }

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        gpu_names = [line.strip() for line in out.decode().splitlines() if line.strip()]
        if gpu_names:
            info["gpu"] = "; ".join(gpu_names)
    except Exception:
        pass

    return info


def _next_run_number(outputs_dir: Path) -> int:
    """Count existing run logs to determine next number. v37: single outputs_dir."""
    if not outputs_dir.exists():
        return 1
    return len(list(outputs_dir.glob("run*.log"))) + 1


def setup_run(repo_root: Path, is_scratch: bool = False) -> Dict[str, Any]:
    """Configure logging + return run context. v37: log + config both in outputs/."""
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    n = _next_run_number(outputs_dir)
    ts = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M")
    scratch_tag = "_scratch" if is_scratch else ""
    run_id = f"run{n}{scratch_tag}_{ts}"

    log_path = outputs_dir / f"{run_id}.log"
    config_path = outputs_dir / f"{run_id}.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.info(f"=== Run started: {run_id} (scratch={is_scratch}) ===")
    logging.info(f"Log:    {log_path}")
    logging.info(f"Config: {config_path}")

    return {
        "run_id": run_id,
        "log_path": log_path,
        "config_path": config_path,
        "timestamp_start": datetime.now(VN_TZ),
        "repo_root": repo_root,
        "is_scratch": is_scratch,
    }


def finalize_run(
    run_ctx: Dict[str, Any],
    *,
    seeds: Dict[str, Any],
    dataset: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    notes: str = "",
) -> None:
    """Emit config JSON sidecar per §11.4 + §11.6 schema."""
    config = {
        "run_id": run_ctx["run_id"],
        "is_scratch": run_ctx["is_scratch"],
        "runtime": {
            "python": platform.python_version(),
            "os": platform.system(),
            "platform": platform.platform(),
            "hardware": _hardware_info(),
        },
        "libraries": _library_versions(),
        "seeds": seeds,
        "dataset": dataset,
        "hyperparameters": hyperparameters,
        "git_commit": _git_commit_short(run_ctx["repo_root"]),
        "timestamps": {
            "start": run_ctx["timestamp_start"].isoformat(),
            "end": datetime.now(VN_TZ).isoformat(),
        },
        "notes": notes,
    }

    with open(run_ctx["config_path"], "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logging.info(f"Config JSON saved: {run_ctx['config_path']}")
    logging.info(f"=== Run finished: {run_ctx['run_id']} ===")