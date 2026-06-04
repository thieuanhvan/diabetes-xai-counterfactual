"""run_ablation_aggregate.py — Master aggregation wrapper.

Top-level workflow wrapper. Right-clickable in PyCharm.

Mirrors the run_main → make_figures auto-call pattern: after run_ablation_all.py
finishes the 5 grids, run this script to aggregate each grid into its
ablation_<type>_table.csv. No CLI arguments needed.

Equivalent to running manually:
    python -m ablation.aggregate taxonomy
    python -m ablation.aggregate class
    python -m ablation.aggregate n_cf
    python -m ablation.aggregate seed
    python -m ablation.aggregate method

Cheap-first execution order matches run_ablation_all.py so that if any cell
failed mid-grid, the cheapest aggregations surface first.


"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Cheap-first order, matches run_ablation_all.py
TYPES = ["taxonomy", "class", "n_cf", "seed", "method"]


def aggregate_type(t: str) -> bool:
    """Invoke `python -m ablation.aggregate <t>` and return success."""
    print(f"\n{'=' * 60}")
    print(f"=== Aggregating: {t}")
    print(f"{'=' * 60}")
    try:
        subprocess.run(
            [sys.executable, "-m", "ablation.aggregate", t],
            cwd=str(REPO_ROOT),
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] {t} aggregate failed (exit code {e.returncode})")
        return False
    except FileNotFoundError as e:
        print(f"  [WARN] {t} aggregate failed (interpreter not found): {e}")
        return False
    except Exception as e:
        print(f"  [WARN] {t} aggregate failed: {e}")
        return False


def main():
    print("=" * 70)
    print("run_ablation_aggregate.py — aggregating 5 ablation tables")
    print(f"Repo root: {REPO_ROOT}")
    print("=" * 70)

    succeeded = []
    failed = []
    for t in TYPES:
        ok = aggregate_type(t)
        (succeeded if ok else failed).append(t)

    print("\n" + "=" * 70)
    print(f"[run_ablation_aggregate] Summary:")
    print(f"  Succeeded ({len(succeeded)}/{len(TYPES)}): {', '.join(succeeded) or '(none)'}")
    if failed:
        print(f"  Failed    ({len(failed)}/{len(TYPES)}): {', '.join(failed)}")
        print(f"\n  Expected output files in outputs/:")
        for t in succeeded:
            print(f"    outputs/ablation_{t}_table.csv")
        sys.exit(1)
    else:
        print(f"\n  All 5 aggregation tables written to outputs/:")
        for t in TYPES:
            print(f"    outputs/ablation_{t}_table.csv")
        print("\n[run_ablation_aggregate] Done.")


if __name__ == "__main__":
    main()
