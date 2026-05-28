"""Rank features by reduction in violation rate (global → per-query).

Two ways to invoke:
1. Auto-called by run_main.py at end of pipeline.
2. Standalone: `python analysis/topk_violations.py` (right-click -> Run in
   PyCharm also works). Regenerates the ranking for the latest run without
   re-running the pipeline.

Convention: the pipeline output directory reflects the latest run only --
no run_id prefix on filenames; each run overwrites the previous contents.
The audit trail of past runs lives in logs/run_{ts}.{log,json}.

Output:
1. CSV: outputs/topk_violations.csv — features ranked by violation
   reduction (descending). Includes both modes' counts so reviewers can
   see whether per-query mode suppressed the feature entirely or just
   redirected its usage.
2. Markdown table: outputs/topk_violations.md
   for easy paste into the manuscript or supplementary materials.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


# ──────────────────────────────────────────────────────────────────────
# This file lives in analysis/ (one level deep), so repo root = parent.parent
# ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve_output_dir() -> Path:
    """Resolve the pipeline output directory from configs/default.yaml so the
    ranking is read from wherever the pipeline actually wrote it (matches
    src/pipelines/main.py). Falls back to outputs/scratch."""
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
            "Run the pipeline with compare_modes=true first."
        )
    return csv_path


def compute_ranking(per_feature_csv: Path) -> pd.DataFrame:
    """Read per_feature.csv, pivot global/per_query side-by-side, rank by reduction.

    Returns DataFrame with columns:
        rank, feature, taxonomy_class,
        global_violation_rate, perquery_violation_rate, violation_reduction,
        global_n_changes, perquery_n_changes, suppression_pattern
    Sorted by violation_reduction DESC, ties broken by global_violation_rate DESC.

    suppression_pattern ∈ {
        'suppressed_entirely'   : per_query n_changes = 0 (feature dropped)
        'redirected'            : per_query n_changes > 0, violations went to 0
        'partial_redirect'      : per_query n_changes > 0, violations reduced but > 0
        'no_global_violations'  : global violation_rate = 0 (feature never violating)
        'no_changes'            : both modes have n_changes = 0 (e.g. immutables)
    }
    """
    df = pd.read_csv(per_feature_csv)
    g = df[df['mode'] == 'global'].set_index('feature')
    p = df[df['mode'] == 'per_query'].set_index('feature')

    features = sorted(set(g.index) & set(p.index))
    rows = []
    for f in features:
        g_viol = g.loc[f, 'violation_rate']
        p_viol = p.loc[f, 'violation_rate']
        g_n    = int(g.loc[f, 'n_total_cf_changes'])
        p_n    = int(p.loc[f, 'n_total_cf_changes'])

        reduction = g_viol - p_viol

        if g_n == 0 and p_n == 0:
            pattern = 'no_changes'
        elif g_viol == 0:
            pattern = 'no_global_violations'
        elif p_n == 0:
            pattern = 'suppressed_entirely'
        elif p_viol == 0:
            pattern = 'redirected'
        else:
            pattern = 'partial_redirect'

        rows.append({
            'feature':                  f,
            'taxonomy_class':           g.loc[f, 'taxonomy_class'],
            'global_violation_rate':    g_viol,
            'perquery_violation_rate':  p_viol,
            'violation_reduction':      reduction,
            'global_n_changes':         g_n,
            'perquery_n_changes':       p_n,
            'suppression_pattern':      pattern,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=['violation_reduction', 'global_violation_rate'],
        ascending=[False, False],
    ).reset_index(drop=True)
    out.insert(0, 'rank', range(1, len(out) + 1))
    return out


def to_markdown(ranking: pd.DataFrame, top_k: Optional[int] = None) -> str:
    """Format ranking as Markdown table for manuscript paste."""
    if top_k is not None:
        ranking = ranking.head(top_k)

    header = (
        "| Rank | Feature | Taxonomy class | Global viol. | Per-query viol. | "
        "Δ reduction | Global n | Per-query n | Pattern |\n"
        "|---:|:---|:---|---:|---:|---:|---:|---:|:---|"
    )
    lines = [header]
    for _, row in ranking.iterrows():
        lines.append(
            f"| {row['rank']} "
            f"| {row['feature']} "
            f"| {row['taxonomy_class']} "
            f"| {row['global_violation_rate']:.3f} "
            f"| {row['perquery_violation_rate']:.3f} "
            f"| {row['violation_reduction']:+.3f} "
            f"| {row['global_n_changes']} "
            f"| {row['perquery_n_changes']} "
            f"| {row['suppression_pattern']} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Generate ranking from the latest run in outputs/."""
    outputs_dir = _resolve_output_dir()
    csv_path = _require_csv(outputs_dir, "per_feature.csv")

    print(f"[topk_violations] Repo root: {REPO_ROOT}")
    print(f"[topk_violations] Input CSV: {csv_path}")

    ranking = compute_ranking(csv_path)

    csv_out = outputs_dir / "topk_violations.csv"
    md_out  = outputs_dir / "topk_violations.md"

    ranking.to_csv(csv_out, index=False)
    print(f"[topk_violations] CSV saved: {csv_out}")

    md_text = to_markdown(ranking)
    md_out.write_text(md_text, encoding='utf-8')
    print(f"[topk_violations] Markdown saved: {md_out}")

    print()
    print("=" * 70)
    print("Ranking (all features, sorted by violation_reduction DESC):")
    print("=" * 70)
    print(md_text)
    print()
    print("[topk_violations] Done.")


if __name__ == "__main__":
    main()
