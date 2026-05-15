"""Run logger + config JSON sidecar per generalrule §11.4/§11.5/§11.6.

Each authoritative run emits 2 artifacts cùng base name trong outputs/:
- outputs/run_{YYYYMMDD_HHMM}.log       (plaintext, Python logging module)
- outputs/run_{YYYYMMDD_HHMM}.json      (UTF-8, indent=2, §11.4 schema)

Scratch runs (is_scratch=True) thêm '_scratch' marker:
- outputs/run_scratch_{YYYYMMDD_HHMM}.{log,json}
→ auto-gitignored per pattern 'outputs/run*_scratch_*'.

Schema fields (§11.4 + §11.6):
- run_id, is_scratch
- runtime: python, os, platform, hardware (cpu/cpu_count/ram_gb/gpu)
- libraries: tracked package versions
- seeds, dataset, hyperparameters, git_commit
- timestamps (ISO 8601 + VN offset), notes

Naming convention (v38, chốt 13/05/2026): pure datetime, KHÔNG dùng sequential
counter. Datetime granularity phút đủ collision-free trong workflow thực tế.
Counter cũ ('runN_') stateful, redundant với datetime, cognitive overhead khi
review. Removed entirely from v38.
"""
from __future__ import annotations

import importlib.metadata as md
import json
import logging
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


# ──────────────────────────────────────────────────────────────────────
# §11.4 helpers — git commit, library versions
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# §11.6 helpers — hardware capture (best-effort, cross-platform)
# ──────────────────────────────────────────────────────────────────────
def _detect_cpu() -> Dict[str, Any]:
    """Best-effort CPU info via platform module + WMI on Windows."""
    info: Dict[str, Any] = {
        "model": platform.processor() or platform.machine(),
        "arch": platform.machine(),
    }
    # Try Windows WMI for more detail
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode(errors="ignore")
            lines = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
            if lines:
                info["model"] = lines[0]
        except Exception:
            pass
    return info


def _detect_cpu_count() -> Optional[int]:
    try:
        import os
        return os.cpu_count()
    except Exception:
        return None


def _detect_ram_gb() -> Optional[float]:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return None


def _detect_gpu() -> Optional[str]:
    """Try nvidia-smi for GPU model. Returns None if not present."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        return out if out else None
    except Exception:
        return None


def _hardware_info() -> Dict[str, Any]:
    return {
        "cpu": _detect_cpu(),
        "cpu_count": _detect_cpu_count(),
        "ram_gb": _detect_ram_gb(),
        "gpu": _detect_gpu(),
    }


# ──────────────────────────────────────────────────────────────────────
# §11.4 main API — setup_run / finalize_run
# ──────────────────────────────────────────────────────────────────────
def setup_run(repo_root: Path, is_scratch: bool = False) -> Dict[str, Any]:
    """Configure logging + return run context.

    Args:
        repo_root: project root for outputs/ folder.
        is_scratch: if True, filename gets '_scratch' marker → auto-gitignored.

    Returns context dict with run_id, paths, timestamp, repo_root, is_scratch.
    """
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M")
    scratch_tag = "_scratch" if is_scratch else ""
    # v38: drop counter, pure datetime → run_20260513_1654 or run_scratch_20260513_1146
    run_id = f"run{scratch_tag}_{ts}"

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
    classifier_metrics: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> None:
    """Emit config JSON sidecar per §11.4 + §11.6 schema.

    v4 update (15/05/2026): added optional classifier_metrics field for §4.2
    main_vi v4 extended baseline metrics (Precision/Recall/F1/Specificity/MCC).
    Backward compatible — old callers without classifier_metrics still work.
    """
    config = {
        "run_id": run_ctx["run_id"],
        "is_scratch": run_ctx["is_scratch"],
        "runtime": {
            "python": platform.python_version(),
            "os": platform.system(),
            "platform": platform.platform(),
            "hardware": _hardware_info(),  # §11.6
        },
        "libraries": _library_versions(),
        "seeds": seeds,
        "dataset": dataset,
        "hyperparameters": hyperparameters,
        "classifier_metrics": classifier_metrics,  # v4: §4.2 main_vi extended metrics
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