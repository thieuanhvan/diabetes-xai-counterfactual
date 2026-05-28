"""Generate Figure: Per-feature actionability comparison (global vs per-query).

Module-level implementation. Called by run_make_figures.py at repo root.

Reads outputs/run_*_per_feature.csv (output of main.py compare_modes mode)
and produces PNG + PDF figure showing:
- Top panel: violation rate per feature (wrong_dir + immutable / total changes)
- Bottom panel: count of CF changes per feature
Both panels compare global vs per-query mode side-by-side.

Figure conventions:
- All in-image text in English (axes, labels, legends)
- No 'Figure N' in title — caption in manuscript handles numbering
- Title is descriptive of content
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


# Feature order grouped by taxonomy class (matches Table 3.2 in manuscript)
ORDERED_FEATURES = [
    # Immutable (4)
    'Age', 'Sex', 'Stroke', 'HeartDiseaseorAttack',
    # Monotonic UP (7) — protective behaviors
    'PhysActivity', 'Fruits', 'Veggies', 'AnyHealthcare',
    'CholCheck', 'Education', 'Income',
    # Monotonic DOWN (8) — risk behaviors/states
    'Smoker', 'HvyAlcoholConsump', 'NoDocbcCost', 'HighBP',
    'HighChol', 'GenHlth', 'MentHlth', 'PhysHlth',
    # Bidirectional (1)
    'BMI',
    # Conditional (1)
    'DiffWalk',
]

# Background tint per taxonomy class (for visual grouping)
CLASS_COLOR = {
    'immutable': '#fce4e4',         # light red
    'monotonic_up': '#e8f5e9',      # light green
    'monotonic_down': '#fff3e0',    # light orange
    'bidirectional': '#e3f2fd',     # light blue
    'conditional': '#f3e5f5',       # light purple
}

# Display labels for class group headers (top of figure)
CLASS_DISPLAY = {
    'immutable': 'Immutable',
    'monotonic_up': 'Monotonic UP (protective)',
    'monotonic_down': 'Monotonic DOWN (risk)',
    'bidirectional': 'Bidir.',
    'conditional': 'Cond.',
}


def _compute_class_boundaries(features: list, feature_class: dict) -> list:
    """Return list of (class_name, start_idx, end_idx) tuples for x-axis grouping."""
    boundaries = []
    prev = None
    start = 0
    for i, f in enumerate(features):
        if feature_class[f] != prev:
            if prev is not None:
                boundaries.append((prev, start, i))
            start = i
            prev = feature_class[f]
    boundaries.append((prev, start, len(features)))
    return boundaries


def generate(
    per_feature_csv: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate per-feature actionability figure from a run's per_feature.csv.

    Args:
        per_feature_csv: Path to outputs/run_*_per_feature.csv.
        output_dir: Where to save PNG + PDF. Defaults to per_feature_csv.parent
                    (i.e. same outputs/ folder).

    Returns:
        {'png': Path, 'pdf': Path} — paths of saved figure files.
    """
    per_feature_csv = Path(per_feature_csv)
    if output_dir is None:
        output_dir = per_feature_csv.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output base name: fixed, no run_id prefix.
    # outputs/ reflects latest run only; manuscript references fig_*.png
    # unambiguously regardless of when the run happened.
    base = 'fig_per_feature'
    png_path = output_dir / f"{base}.png"
    pdf_path = output_dir / f"{base}.pdf"

    # Load data
    df = pd.read_csv(per_feature_csv)
    global_df = df[df['mode'] == 'global'].set_index('feature')
    perquery_df = df[df['mode'] == 'per_query'].set_index('feature')
    feature_class = {f: global_df.loc[f, 'taxonomy_class'] for f in ORDERED_FEATURES}

    # Compute class boundaries for background tint + group labels
    boundaries = _compute_class_boundaries(ORDERED_FEATURES, feature_class)

    # ──────────────────────────────────────────────────────────
    # Build figure: 2 panels stacked
    # ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.5),
        gridspec_kw={'height_ratios': [1, 1]},
    )
    x = range(len(ORDERED_FEATURES))
    bar_width = 0.38

    # Background tint by class (both panels)
    for ax in (ax1, ax2):
        for cls, s, e in boundaries:
            ax.axvspan(s - 0.5, e - 0.5, color=CLASS_COLOR[cls], alpha=0.5, zorder=0)

    # ── Panel A: violation rate ──
    global_viol = [global_df.loc[f, 'violation_rate'] for f in ORDERED_FEATURES]
    perq_viol = [perquery_df.loc[f, 'violation_rate'] for f in ORDERED_FEATURES]

    ax1.bar(
        [i - bar_width / 2 for i in x], global_viol, bar_width,
        label='Global (DiCE binary mutability)',
        color='#d32f2f', edgecolor='black', linewidth=0.5,
    )
    ax1.bar(
        [i + bar_width / 2 for i in x], perq_viol, bar_width,
        label='Per-query (P4 directional taxonomy)',
        color='#388e3c', edgecolor='black', linewidth=0.5,
    )

    ax1.set_ylabel('Violation rate\n(wrong-direction + immutable)', fontsize=11)
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(x)
    ax1.set_xticklabels([''] * len(ORDERED_FEATURES))
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax1.set_title(
        'Per-feature actionability: directional taxonomy eliminates '
        'wrong-direction violations',
        fontsize=12, fontweight='bold', pad=10,
    )

    # Annotate 100% violation features in global mode
    for i, f in enumerate(ORDERED_FEATURES):
        if global_viol[i] >= 0.99:
            ax1.text(
                i - bar_width / 2, global_viol[i] + 0.03, '100%',
                ha='center', fontsize=8, color='#b71c1c', fontweight='bold',
            )

    # Class group labels at top (stagger narrow classes to avoid overlap)
    for cls, s, e in boundaries:
        mid = (s + e - 1) / 2
        width = e - s
        # bidirectional + conditional are 1-feature wide → stagger vertically
        y_pos = 1.13 if width > 1 or cls == 'bidirectional' else 1.18
        ax1.text(
            mid, y_pos, CLASS_DISPLAY[cls],
            ha='center', fontsize=9, fontweight='bold', style='italic',
            transform=ax1.get_xaxis_transform(),
        )

    # ── Panel B: count of CF changes ──
    global_n = [global_df.loc[f, 'n_total_cf_changes'] for f in ORDERED_FEATURES]
    perq_n = [perquery_df.loc[f, 'n_total_cf_changes'] for f in ORDERED_FEATURES]

    ax2.bar(
        [i - bar_width / 2 for i in x], global_n, bar_width,
        label='Global', color='#d32f2f', edgecolor='black', linewidth=0.5,
    )
    ax2.bar(
        [i + bar_width / 2 for i in x], perq_n, bar_width,
        label='Per-query', color='#388e3c', edgecolor='black', linewidth=0.5,
    )

    ax2.set_ylabel('Number of CF changes', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ORDERED_FEATURES, rotation=45, ha='right', fontsize=9)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.15)

    # Save outputs
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    return {'png': png_path, 'pdf': pdf_path}
