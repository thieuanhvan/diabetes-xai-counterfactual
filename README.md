# Part C v6 — Tiny make_figures Column Fix (15/05/2026)

**1-line change** vs v5: rename comparison.csv column `delta_rel_pct` → `rel_delta_pct`
to match `analysis/figures/comparison_metrics.py:120` expectation.

```diff
-"delta_rel_pct": 100.0 * (agg_pq.values - agg_g.values) / agg_g.values,
+"rel_delta_pct": 100.0 * (agg_pq.values - agg_g.values) / agg_g.values,
```

## Apply when

This fix is **OPTIONAL for ablation runs** — `run_ablation_all.py` does NOT call
make_figures (only `run_main.py` does). So:

- **Launch ablation_all with v5 NOW** (already applied) — fine
- Apply v6 **anytime later** to fix make_figures step in standalone `run_main.py` runs
- Or apply v6 NOW (10 seconds) then launch ablation — also fine

Either path produces identical ablation results.

## Confirmed v5 → v6 results identical for ablation

The only thing that changes is the column name written to `comparison.csv`. Data values
identical. CSV reader scripts (aggregate.py, topk_violations.py) read different columns
and are unaffected.

## After v6, run_main produces clean figures

Smoke 16:33 with v5 finished but make_figures failed `KeyError: rel_delta_pct`.
After v6, all 4 figure outputs generate:
- `outputs/run_*_fig_per_feature.{png,pdf}`
- `outputs/run_*_fig_comparison_metrics.{png,pdf}`
