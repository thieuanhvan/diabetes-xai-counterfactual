"""Shared ablation harness — config override + run dispatch + per-cell snapshot.

Three responsibilities:
1. Load configs/default.yaml as dict (ground truth — never modified).
2. Deep-merge with per-ablation overrides into a fresh dict.
3. Write configs/ablation_<name>.yaml + dispatch via src.pipelines.main.

After each cell runs, snapshot outputs/*.csv + matching logs/run_*.json into
outputs/_ablation_archive/<cell_name>/. This preserves per-cell data despite
the "outputs/ overwrites each run" convention. aggregate.py reads
from these archives, not from outputs/ directly.

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

v4 (16/05/2026): added _snapshot_outputs() called after each pipeline run.
Captures the 4 unprefixed CSVs in outputs/ + matching logs/run_*.json sidecar
into outputs/_ablation_archive/<cell_name>/. Pipeline core untouched (no edits
to src/pipelines/main.py or src/utils/run_logger.py).
"""
from __future__ import annotations

import copy
import json
import logging
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# Repo root = parent of "ablation/" folder (i.e. /path/to/diabetes-xai-counterfactual/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
LOGS_DIR = REPO_ROOT / "logs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
ARCHIVE_ROOT = OUTPUTS_DIR / "_ablation_archive"

# CSV files that the pipeline writes to outputs/ (no run_id prefix).
# Order matters: comparison.csv first so partial snapshots still have the
# headline metrics if something fails mid-copy.
SNAPSHOT_CSVS = (
    "comparison.csv",
    "perquery_cf_metrics.csv",
    "global_cf_metrics.csv",
    "per_feature.csv",
)

VN_TZ = timezone(timedelta(hours=7))


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
# Snapshot helpers (v4) — preserve per-cell artifacts
# ──────────────────────────────────────────────────────────────────────
def _newest_run_json(after_ts: Optional[datetime]) -> Optional[Path]:
    """Find the most-recently-modified non-scratch logs/run_*.json.

    If after_ts is given, require mtime >= after_ts to filter out stale JSONs
    from prior runs. Falls back to absolute newest if no match (defensive).
    """
    if not LOGS_DIR.exists():
        return None
    candidates = [p for p in LOGS_DIR.glob("run_*.json") if "_scratch_" not in p.stem]
    if not candidates:
        return None
    if after_ts is not None:
        fresh = [p for p in candidates if datetime.fromtimestamp(p.stat().st_mtime, tz=VN_TZ) >= after_ts]
        if fresh:
            return max(fresh, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_notes_suffix(notes: str) -> Dict[str, str]:
    """Parse 'key=val; key=val' tokens from notes string. Returns dict.

    Only picks tokens with exactly one '='. Order-preserving via dict insertion.
    """
    out: Dict[str, str] = {}
    if not notes:
        return out
    for token in notes.split(";"):
        token = token.strip()
        if "=" not in token:
            continue
        # Split on first '=' only — values may contain '=' (rare).
        k, v = token.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and v and k not in out:
            out[k] = v
    return out


def _snapshot_outputs(cell_name: str, run_started_at: datetime) -> Path:
    """Copy outputs/*.csv + matching logs/run_*.json into archive dir.

    Returns archive_dir path. Idempotent — overwrites existing files in same
    archive dir if cell is rerun.
    """
    archive_dir = ARCHIVE_ROOT / cell_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy the 4 CSVs (best-effort — log warning if any missing).
    missing = []
    for csv_name in SNAPSHOT_CSVS:
        src = OUTPUTS_DIR / csv_name
        if not src.exists():
            missing.append(csv_name)
            continue
        shutil.copy2(src, archive_dir / csv_name)
    if missing:
        print(f"[ablation/core] WARN: cell {cell_name} missing CSVs in outputs/: {missing}")

    # 2. Find matching logs/run_*.json (newest after run_started_at).
    json_src = _newest_run_json(after_ts=run_started_at)
    run_id: Optional[str] = None
    ablation_type: Optional[str] = None
    notes_kv: Dict[str, str] = {}
    if json_src is not None:
        shutil.copy2(json_src, archive_dir / "config.json")
        try:
            with open(json_src, encoding="utf-8") as f:
                cfg_json = json.load(f)
            run_id = cfg_json.get("run_id")
            notes_kv = _parse_notes_suffix(cfg_json.get("notes", ""))
            ablation_type = notes_kv.get("ablation")
        except Exception as e:
            print(f"[ablation/core] WARN: cell {cell_name} could not parse config.json: {e}")
    else:
        print(f"[ablation/core] WARN: cell {cell_name} no matching logs/run_*.json found")

    # 3. Write manifest.json.
    manifest = {
        "cell_name": cell_name,
        "run_id": run_id,
        "ablation_type": ablation_type,
        "notes_kv": notes_kv,
        "snapshot_timestamp_vn": datetime.now(VN_TZ).isoformat(),
        "csvs_present": [c for c in SNAPSHOT_CSVS if (archive_dir / c).exists()],
        "config_present": (archive_dir / "config.json").exists(),
    }
    with open(archive_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[ablation/core] ✓ snapshot: {archive_dir.relative_to(REPO_ROOT)}")
    print(f"[ablation/core]   run_id={run_id}  ablation_type={ablation_type}  csvs={len(manifest['csvs_present'])}/{len(SNAPSHOT_CSVS)}")
    return archive_dir


# ──────────────────────────────────────────────────────────────────────
# Run dispatch
# ──────────────────────────────────────────────────────────────────────
def run_with_override(name: str, overrides: Dict[str, Any]) -> Path:
    """Write per-ablation config + dispatch via src.pipelines.main + snapshot.

    Returns the path to the config file used (caller may want to log or
    archive it). The actual run outputs go to outputs/ (overwritten each run);
    a snapshot is preserved in outputs/_ablation_archive/<name>/.
    """
    cfg_path = write_ablation_config(name, overrides)
    print()
    print("=" * 60)
    print(f"[ablation/core] Dispatch: name={name}")
    print(f"[ablation/core] Config:   {cfg_path.relative_to(REPO_ROOT)}")
    print(f"[ablation/core] Overrides: {overrides}")
    print("=" * 60)
    # Capture pre-run timestamp so we can disambiguate the matching JSON sidecar
    # even if multiple runs land in the same minute (datetime granularity is min).
    t_start = datetime.now(VN_TZ)
    # Lazy import — keeps core.py importable even if src.pipelines.main has
    # heavy deps not yet installed in a freshly cloned env
    from src.pipelines.main import main as run_pipeline
    run_pipeline(str(cfg_path))
    # Snapshot artifacts AFTER pipeline finishes, BEFORE next cell can overwrite.
    _snapshot_outputs(name, run_started_at=t_start)
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
