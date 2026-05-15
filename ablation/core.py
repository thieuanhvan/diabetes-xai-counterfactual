"""Shared ablation harness — config override + run dispatch.

Three responsibilities:
1. Load configs/default.yaml as dict (ground truth — never modified).
2. Deep-merge with per-ablation overrides into a fresh dict.
3. Write configs/ablation_<name>.yaml + dispatch via src.pipelines.main.

CONSTRAINT (read first):
src.pipelines.main computes `repo_root = config_path.parent.parent`, so config
files MUST live ONE level deep under repo root, i.e. configs/<name>.yaml — NOT
configs/ablation/<name>.yaml. We use prefix `ablation_<type>_<value>.yaml` to
namespace them while staying flat (e.g. configs/ablation_seed_42.yaml).

Override syntax: dict subset that gets recursively merged into the default
config. Example:
    run_with_override("seed_42", {
        "random": {"seed": 42},
        "xgboost": {"random_state": 42},
        "evaluate": {"compare_modes": True},
    })

Each call produces a standard run_<YYYYMMDD_HHMM>.{log,json,_cf_metrics.csv}
in outputs/ — no special ablation namespace. aggregate.py post-filters by
run_id timestamp + ablation config file mapping (logged in JSON).

Notes capture: every ablation run sets cfg["run"]["is_scratch"] = False so the
JSONs are NOT filtered by .gitignore — these are paper-authoritative.

Per ablation_vi v2 §8 + Generalrule v37 §11.4/§11.5/§11.6.
"""
from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


# Repo root = parent of "ablation/" folder (i.e. /path/to/diabetes-xai-counterfactual/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


# Ensure repo root on sys.path for src.* imports when called from wrappers
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────
def load_default_config() -> Dict[str, Any]:
    """Read configs/default.yaml and return a fresh dict."""
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Default config not found: {DEFAULT_CONFIG_PATH}")
    with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `overrides` INTO a copy of `base`. Returns new dict.

    Scalar values in overrides replace base; nested dicts are merged key-wise.
    Lists are replaced wholesale (not extended) to keep override semantics simple.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def write_ablation_config(name: str, overrides: Dict[str, Any]) -> Path:
    """Write configs/ablation_<name>.yaml = default ∪ overrides. Returns path."""
    cfg = deep_merge(load_default_config(), overrides)
    # Always non-scratch (paper-authoritative)
    cfg.setdefault("run", {})["is_scratch"] = False

    cfg_path = REPO_ROOT / "configs" / f"ablation_{name}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return cfg_path


# ──────────────────────────────────────────────────────────────────────
# Run dispatch
# ──────────────────────────────────────────────────────────────────────
def run_with_override(name: str, overrides: Dict[str, Any]) -> Path:
    """Write per-ablation config + dispatch via src.pipelines.main.

    Returns the path to the config file used (caller may want to log or
    archive it). The actual run outputs go to outputs/run_<timestamp>.{log,json,csv}.
    """
    cfg_path = write_ablation_config(name, overrides)
    print()
    print("=" * 60)
    print(f"[ablation/core] Dispatch: name={name}")
    print(f"[ablation/core] Config:   {cfg_path.relative_to(REPO_ROOT)}")
    print(f"[ablation/core] Overrides: {overrides}")
    print("=" * 60)
    # Lazy import — keeps core.py importable even if src.pipelines.main has
    # heavy deps not yet installed in a freshly cloned env
    from src.pipelines.main import main as run_pipeline
    run_pipeline(str(cfg_path))
    return cfg_path


def run_grid(
    name_prefix: str,
    grid: List[Tuple[str, Dict[str, Any]]],
) -> List[Path]:
    """Run a list of named overrides sequentially. Returns config paths used.

    Example:
        grid = [("seed_42", {"random": {"seed": 42}}),
                ("seed_123", {"random": {"seed": 123}})]
        run_grid("multiseed", grid)
    """
    paths = []
    print()
    print("#" * 60)
    print(f"[ablation/core] Starting grid: {name_prefix} ({len(grid)} runs)")
    print("#" * 60)
    for i, (name, ov) in enumerate(grid, start=1):
        print(f"\n[ablation/core] === Grid {i}/{len(grid)}: {name} ===")
        paths.append(run_with_override(name, ov))
    print()
    print("#" * 60)
    print(f"[ablation/core] Grid complete: {len(paths)} runs")
    print("#" * 60)
    return paths


# ──────────────────────────────────────────────────────────────────────
# Smoke test: importable + finds default config
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        cfg = load_default_config()
        print(f"✓ Loaded default config from {DEFAULT_CONFIG_PATH}")
        print(f"  Keys: {list(cfg.keys())}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
