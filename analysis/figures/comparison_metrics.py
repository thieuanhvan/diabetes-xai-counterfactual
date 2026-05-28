"""Generate Figure: Comparison of CF quality metrics (global vs per-query).

Module-level implementation. Called by analysis/make_figures.py.

Reads outputs/run_*_comparison.csv (output of main.py compare_modes mode)
and produces PNG + PDF figure with two panels:
- Top:    relative delta (%) of per-query vs global, one bar per metric,
          color-coded by direction of improvement (green = good, red = bad).
- Bottom: absolute values side-by-side (global vs per-query), grouped by
          metric. Y-axis is log-scale because metrics span [0, 15+].

Figure conventions:
- All in-image text in English (axes, labels, legends).
- No 'Figure N' in title — caption in manuscript handles numbering.
- Title is descriptive of content.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


# Display order chosen to group metrics by interpretation:
#   - Quality of CFs (validity, actionability)
#   - Failure modes (wrong_dir, immutable)
#   - Cost of CFs (proximity, sparsity, plausibility, diversity)
ORDERED_METRICS = [
    'validity',
    'actionability',
    'wrong_dir_violations',
    'immutable_violations',
    'proximity',
    'sparsity',
    'plausibility',
    'diversity',
]

# Display labels (more readable than CSV column names)
METRIC_LABELS = {
    'validity':             'Validity',
    'actionability':        'Actionability',
    'wrong_dir_violations': 'Wrong-dir\nviolations',
    'immutable_violations': 'Immutable\nviolations',
    'proximity':            'Proximity\n(L1)',
    'sparsity':             'Sparsity',
    'plausibility':         'Plausibility\n(kNN dist)',
    'diversity':            'Diversity',
}

# Direction of "good": +1 = higher is better; -1 = lower is better.
# Used to color rel_delta bars: green when delta points in good direction.
GOOD_DIRECTION = {
    'validity':             +1,
    'actionability':        +1,
    'wrong_dir_violations': -1,
    'immutable_violations': -1,
    'proximity':            -1,   # lower L1 dist = closer CF
    'sparsity':             -1,   # lower = fewer features changed
    'plausibility':         -1,   # lower = closer to data manifold
    'diversity':            +1,   # higher diversity = better
}

# Colors
GREEN = '#388e3c'
RED   = '#d32f2f'
NEUTRAL_GREY = '#9e9e9e'


def _rel_delta_color(metric: str, rel_delta_pct: float) -> str:
    """Green if rel_delta points in the good direction, red if bad, grey if 0."""
    if abs(rel_delta_pct) < 1e-6:
        return NEUTRAL_GREY
    good_dir = GOOD_DIRECTION[metric]
    # delta is positive ⇒ per_query > global. If good_dir is +1 ⇒ green.
    if (rel_delta_pct > 0 and good_dir > 0) or (rel_delta_pct < 0 and good_dir < 0):
        return GREEN
    return RED


def generate(
    comparison_csv: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate comparison-metrics figure from a run's comparison.csv.

    Args:
        comparison_csv: Path to outputs/run_*_comparison.csv.
        output_dir:     Where to save PNG + PDF. Defaults to comparison_csv.parent.

    Returns:
        {'png': Path, 'pdf': Path} — paths of saved figure files.
    """
    comparison_csv = Path(comparison_csv)
    if output_dir is None:
        output_dir = comparison_csv.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output base name: fixed, no run_id prefix.
    # outputs/ reflects latest run only; manuscript references fig_*.png
    # unambiguously regardless of when the run happened.
    base = 'fig_comparison_metrics'
    png_path = output_dir / f"{base}.png"
    pdf_path = output_dir / f"{base}.pdf"

    # Load data — set metric as index for label-based lookup
    df = pd.read_csv(comparison_csv).set_index('metric')

    # Sanity-check: all expected metrics present
    missing = [m for m in ORDERED_METRICS if m not in df.index]
    if missing:
        raise ValueError(
            f"comparison.csv missing expected metrics: {missing}. "
            f"Found: {list(df.index)}"
        )

    global_vals = [df.loc[m, 'global']        for m in ORDERED_METRICS]
    perq_vals   = [df.loc[m, 'per_query']     for m in ORDERED_METRICS]
    rel_deltas  = [df.loc[m, 'rel_delta_pct'] for m in ORDERED_METRICS]
    abs_deltas  = [df.loc[m, 'delta_abs']     for m in ORDERED_METRICS]

    bar_colors = [_rel_delta_color(m, d) for m, d in zip(ORDERED_METRICS, rel_deltas)]

    # ──────────────────────────────────────────────────────────
    # Build figure: 2 panels stacked
    # ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.5),
        gridspec_kw={'height_ratios': [1, 1]},
    )
    x = range(len(ORDERED_METRICS))
    bar_width = 0.38

    # ── Panel A: relative delta (%) ──
    bars1 = ax1.bar(
        x, rel_deltas,
        color=bar_colors, edgecolor='black', linewidth=0.5,
    )
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_ylabel('Relative Δ (%)\nper-query vs global', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([''] * len(ORDERED_METRICS))
    ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax1.set_title(
        'Per-query taxonomy lifts validity & actionability, '
        'eliminates wrong-direction violations',
        fontsize=12, fontweight='bold', pad=10,
    )

    # Y-limit with headroom for labels
    ymax = max(rel_deltas) if max(rel_deltas) > 0 else 0
    ymin = min(rel_deltas) if min(rel_deltas) < 0 else 0
    ax1.set_ylim(ymin * 1.30 - 8, ymax * 1.25 + 8)

    # Annotate each bar with rel_delta value + absolute delta
    for i, (bar, rel, absd) in enumerate(zip(bars1, rel_deltas, abs_deltas)):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 1.5 if height >= 0 else -1.5
        label = f'{rel:+.1f}%\n(Δ {absd:+.3f})'
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            label,
            ha='center', va=va, fontsize=8.5, fontweight='bold',
            color='black',
        )

    # Legend (color key)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=GREEN, edgecolor='black', label='Δ in good direction'),
        Patch(facecolor=RED,   edgecolor='black', label='Δ in bad direction'),
        Patch(facecolor=NEUTRAL_GREY, edgecolor='black', label='No change'),
    ]
    ax1.legend(handles=legend_handles, loc='lower right', fontsize=9, framealpha=0.95)

    # ── Panel B: absolute values, side-by-side, log-scale ──
    ax2.bar(
        [i - bar_width / 2 for i in x], global_vals, bar_width,
        label='Global (DiCE binary mutability)',
        color='#d32f2f', edgecolor='black', linewidth=0.5,
    )
    ax2.bar(
        [i + bar_width / 2 for i in x], perq_vals, bar_width,
        label='Per-query (P4 directional taxonomy)',
        color='#388e3c', edgecolor='black', linewidth=0.5,
    )

    ax2.set_ylabel('Value (log scale)', fontsize=11)
    ax2.set_yscale('symlog', linthresh=0.01)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [METRIC_LABELS[m] for m in ORDERED_METRICS],
        rotation=0, ha='center', fontsize=9,
    )
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(axis='y', which='both', linestyle='--', alpha=0.4, zorder=1)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.20)

    # Save outputs
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    return {'png': png_path, 'pdf': pdf_path}
