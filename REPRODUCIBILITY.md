# Reproducibility

This document specifies the exact environment, data, seeds and expected
numerical outputs needed to reproduce the results reported in
*"Knowledge-Guided Counterfactual Explanations for Diabetes Risk Decision
Support: A Directional Intervention Taxonomy"* (accepted, International Journal of Medical Informatics, 2026; DOI 10.1016/j.ijmedinf.2026.106555).

Companion file `README.md` is the high-level entry point; this file is the
rigorous step-by-step for reviewers and future maintainers.

## TL;DR

```bash
# Clone the public repository, then:
git clone https://github.com/thieuanhvan/diabetes-xai-counterfactual.git
cd diabetes-xai-counterfactual
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The two cohorts are committed under data/ (CC0-1.0) — no acquisition step needed.
python run_main.py
```

Expected wall-clock: ~8 minutes on the reference hardware (Intel Core i7,
8 logical cores, no GPU; full run, both CF modes — see §5.1 for the
per-stage breakdown). Outputs land in `outputs/scratch/`. Compare the
numerical values against the "Expected numerical outputs" tables below, or
diff directly against the frozen authoritative snapshot at
`outputs/archive/manuscript/`.

## 1. Environment

| Component | Pinned version | Source |
|---|---|---|
| Python | 3.12.10 | python.org |
| numpy | 1.26.4 | `requirements.txt` |
| pandas | 3.0.3 | `requirements.txt` |
| scipy | 1.17.1 | `requirements.txt` |
| scikit-learn | 1.8.0 | `requirements.txt` |
| xgboost | 3.2.0 | `requirements.txt` |
| dice-ml | 0.12 | `requirements.txt` |
| shap | 0.49.1 | `requirements.txt` |
| matplotlib | ≥3.8 | `requirements.txt` |
| pyyaml | 6.0.3 | `requirements.txt` |
| psutil | 7.2.2 | `requirements.txt` |

The full pinned list is in `requirements.txt`. Each
`outputs/scratch/run_*.json` sidecar records the actual versions present at
runtime, so a reviewer can verify environment fidelity by diffing the
sidecar against this table or against
`outputs/archive/manuscript/config.json`.

Operating system tested: Windows 10 (primary author environment). Linux
and macOS are expected to work; the only OS-dependent code is path
handling, which uses `pathlib` throughout.

## 2. Data

See `data/README.md`. The pipeline expects `data/cdc_brfss_diabetes_2021.csv`
with the 21-feature julnazz/Teboul schema and `Diabetes_binary` as target
column. Both cohorts are committed under data/ (CC0-1.0; the CDC source remains the authoritative upstream).

Expected dataset properties (sanity check):

```
X.shape == (236378, 21)
y.shape == (236378,)
y.mean() ≈ 0.1420   # class prevalence
```

The pipeline logs these values at startup as a guardrail.

## 3. Seeds and determinism

All random number generators are seeded from a single root seed = **42**:

| Component | Seeding mechanism |
|---|---|
| Python `random` | `random.seed(42)` |
| numpy | `np.random.seed(42)` |
| PYTHONHASHSEED | `os.environ["PYTHONHASHSEED"] = "42"` |
| scikit-learn `train_test_split` | `random_state=42` |
| XGBoost | `random_state=42`, `tree_method='hist'` |
| DiCE `generate_counterfactuals` | `random_seed=42` |

All seeding goes through `src/utils/seed.py::seed_everything(42)`, called
once at the top of `run_main.py`. Running the pipeline twice on the same
machine and OS produces byte-identical CSV outputs.

Multi-seed variance ablation uses `{42, 123, 2024, 7, 31337}` (Section
4.5.1 of the manuscript); CV across seeds is ≤0.7% on validity and ≤0.7%
on actionability.

## 4. Hardware tested

The authoritative reference run was carried out on:

| Spec | Value |
|---|---|
| CPU | Intel Core i7-1068NG7 (Ice Lake, 4 cores / 8 logical threads, 2.30 GHz base) |
| RAM | 32 GB |
| GPU | none used |
| Disk | local SSD |
| OS | Windows 10 (10.0.19045) |

Wall-clock figures below assume comparable hardware. The pipeline is
CPU-bound (XGBoost `tree_method='hist'` + DiCE `random` method); no GPU
acceleration is used or required. Each `outputs/scratch/run_*.json`
sidecar records the actual hardware at runtime.

## 5. Reproduction steps

### 5.1 Baseline

```bash
python run_main.py
```

`run_main.py` takes no CLI arguments; all configuration is read from
`configs/default.yaml`. The script runs the full pipeline end-to-end:

1. Load BRFSS 2021 CSV from `data/`.
2. Stratified 80/20 train/test split (no feature scaling — XGBoost is scale-invariant; CFs are generated in the original encoded space for interpretability).
3. Train XGBoost (`n_estimators=500`, `max_depth=6`, `learning_rate=0.05`,
   `subsample=0.9`, `colsample_bytree=0.9`, `tree_method='hist'`).
4. Select top-200 high-risk patients by predicted `P(diabetes=1)`.
5. Generate counterfactuals in both modes (global, per-query) via DiCE
   method='random', `n_CF=5` per query.
6. Compute CF metrics (validity, proximity, sparsity, plausibility,
   actionability) per mode.
7. Run analysis modules: `make_figures.py`, `topk_violations.py`,
   `per_feature_actionability.py`, `comparison_metrics.py`.

Expected wall-clock breakdown on the reference hardware (full BRFSS 2021,
`sample_n=null` in `configs/default.yaml`):

| Stage | Wall-clock |
|---|---|
| Data load + preprocessing | ≈1 s |
| XGBoost training | ≈5 s |
| Global CF generation (200 queries × 5 CFs) | ≈3.5 min |
| Per-query CF generation (200 queries × 5 CFs) | ≈3.5 min |
| Scoring + analysis | ≈30 s |
| **Total** | **≈8 min** |

Outputs land in `outputs/scratch/`:
`run_<YYYYMMDD_HHMM>.{log,json}` sidecar pair plus
`{comparison,global_cf_metrics,perquery_cf_metrics,per_feature}.csv` and
`fig_*.{png,pdf}`.

### 5.2 External validation (BRFSS 2015)

```bash
python analysis/run_external_validation_brfss2015.py
```

Trains XGBoost on BRFSS 2021 (identical seed and hyperparameters as the
baseline) and applies the trained classifier directly to BRFSS 2015 without
recalibration, then generates counterfactuals on the top-200 high-risk
BRFSS 2015 patients in both modes. Output: `outputs/external_2015/`. This is
a tracked directory, so verify reproduction with `git diff
outputs/external_2015/` — it should report no changes. Wall-clock ~9 min.
Expected headline numbers (manuscript Section 4.4, Table 5): external AUC
0.8269, per-query wrong-direction violation rate 0.0000.

### 5.3 Ablations

```bash
python run_ablation_all.py
```

Sequentially runs the five ablation grids (multi-seed, method,
n_counterfactuals, class-balance / risk threshold, taxonomy granularity).
Total wall-clock: ≈6 hours on the reference hardware. After the grids
complete, aggregate into comparison tables:

```bash
python run_ablation_aggregate.py
```

This wrapper invokes the five aggregations in cheap-first order; equivalent
manual invocation is `python -m ablation.aggregate <type>` per ablation
type, in any order. Outputs: `outputs/scratch/ablation_<type>_table.csv`
(or `outputs/ablation_<type>_table.csv` for repo-level summary tables).

Individual ablation runs land in
`outputs/_ablation_archive/<cell_name>/` (e.g.
`taxonomy_3class_conservative`, `seed_42`, `class_general`) as frozen
snapshots that can be diffed against the manuscript numbers without
rerunning.

## 6. Expected numerical outputs

The single frozen authoritative reference is at `outputs/archive/manuscript/`
(the main experimental configuration; full provenance — exact versions, seed,
hardware — recorded in its `config.json`). Values below are the
headline numbers reported in the manuscript. Reproduction should match to
all displayed decimal places (deterministic pipeline given identical seed
and pinned versions).

### 6.1 Classifier

| Quantity | Expected value | Source |
|---|---|---|
| Test AUC | **0.8233** | run log line, `outputs/archive/manuscript/config.json` |
| Train/test split sizes | 189,102 / 47,276 | log line |
| Class prevalence (full) | 0.1420 | log line |
| Test prevalence | 0.1420 | log line |

### 6.2 CF metrics — global mode

(from `outputs/archive/manuscript/comparison.csv`, column `global`)

| Quantity | Expected value |
|---|---|
| Validity | 0.7520 |
| Proximity ($L_1$) | 1.3966 |
| Sparsity | 0.0671 |
| Diversity | 15.1938 |
| Plausibility ($k$=50) | 4.2989 |
| **Actionability** | **0.6655** |
| Wrong-direction violation rate | 0.4350 |
| Immutable violation rate | 0.0000 |

### 6.3 CF metrics — per-query mode

(from `outputs/archive/manuscript/comparison.csv`, column `per_query`)

| Quantity | Expected value |
|---|---|
| Validity | 0.8080 |
| Proximity ($L_1$) | 1.4710 |
| Sparsity | 0.0723 |
| Diversity | 14.7095 |
| Plausibility ($k$=50) | 4.0297 |
| **Actionability** | **0.9880** |
| Wrong-direction violation rate | 0.0000 |
| Immutable violation rate | 0.0000 |

Total CF changes across all features: **1,411 (global) → 1,520 (per-query, +7.7%)**.
Verify via `outputs/archive/manuscript/per_feature.csv` by summing
`n_total_cf_changes` per mode.

### 6.4 Per-feature breakdown (Pattern (a) check)

In `outputs/archive/manuscript/per_feature.csv`, three features should
record 100% wrong-direction violations under global mode and 0 changes
under per-query mode (suppressed by taxonomy):

- `AnyHealthcare`
- `CholCheck`
- `HvyAlcoholConsump`

A further eleven features (Pattern (b) — redirected): `NoDocbcCost`,
`Education`, `Smoker`, `Fruits`, `MentHlth`, `Veggies`, `PhysActivity`,
`PhysHlth`, `Income`, `GenHlth`, `HighChol` — record CF changes in both
modes with zero wrong-direction violations under per-query mode.

## 7. Verifying reproduction

After running, compare your `outputs/scratch/` against the frozen
authoritative reference at `outputs/archive/manuscript/`:

```bash
diff outputs/scratch/comparison.csv outputs/archive/manuscript/comparison.csv
diff outputs/scratch/per_feature.csv outputs/archive/manuscript/per_feature.csv
```

A successful reproduction shows zero diff (deterministic pipeline). **On Windows a fresh run writes CRLF line endings while the committed reference is LF**, so a plain `diff` may report every line as changed even when the numbers are identical; compare ignoring line endings instead — e.g. `diff --strip-trailing-cr ...` (Git Bash) or `git diff --no-index --ignore-cr-at-eol ...`. The key headline values to verify:

- AUC == 0.8233 (exact match)
- Per-query actionability == 0.9880 (exact match)
- Per-query wrong-direction violation rate == 0.0000 (exact match)
- Three Pattern (a) features all show 100% / 0 split

If any value drifts beyond the multi-seed CV reported in §4.7.1 of the
manuscript (≤0.7% on validity and actionability), see Troubleshooting below.

## 8. Known sources of variation

By design, **none** under fixed seed and pinned versions. The pipeline is
fully deterministic. The following are *not* sources of variation:

- Run timestamp (only affects sidecar filename, not values)
- Run-to-run timing (CPU contention; affects wall-clock, not numbers)
- OS (Windows / Linux / macOS produce identical floating-point results
  with the same numpy + xgboost versions)

The following *would* introduce variation but are guarded against:

- Library version drift — pinned in `requirements.txt`, captured in
  sidecar JSON.
- Random seed leakage — single root seed propagated via
  `seed_everything(42)`.
- Data file drift — schema sanity check at load (logs class prevalence,
  shape, and column presence; aborts if any column missing).

## 9. Troubleshooting

**AUC differs from 0.8233.** Diff your run's JSON sidecar
(`outputs/scratch/run_*.json`) against
`outputs/archive/manuscript/config.json`. The most common cause is a
numpy or xgboost version mismatch from a stale virtual environment.

**`KeyError` on column load.** The BRFSS CSV must contain exactly the 22
columns listed in `data/README.md`. The cleaning steps documented in
`data/PROVENANCE.md` produce the correct schema from the public CDC source.

**`TypeError: Invalid value <x> for dtype 'float32'` during kdtree method
ablation.** Known DiCE 0.12 + pandas 3.x incompatibility; workaround is
applied in `src/pipelines/counterfactual/dice_runner.py` (continuous
columns recast to float64 in `DiCE Data` constructor). See README §Known
issues for the upstream issue references.

**Out-of-memory during ablations.** The ablation grids serialise to disk
between runs but still hold the test set + XGBoost model in memory. The
per-cell results are already provided under `outputs/_ablation_archive/`, so
re-running the full grid is not required to verify the ablation numbers. If
you do re-run on a 16 GB machine, run `run_ablation_all.py` alone (close
other applications), or re-run only the specific cells of interest via the
utility scripts `ablation_taxonomy_rerun.py` and `ablation_method_kdtree.py`.

**Wall-clock far exceeds the ≈8-minute estimate.** Confirm
`tree_method='hist'` is in effect (XGBoost defaults to `exact` on some
installations). Confirm matplotlib backend is not opening interactive
windows that block the run. Confirm `configs/default.yaml: data.sample_n`
is `null` (full dataset) rather than left at a dev value.

## 10. Provenance

The single authoritative reference is the frozen snapshot at
`outputs/archive/manuscript/`. Its accompanying `config.json` records the full
provenance:

- Exact library versions present at the time of authoritative-run execution
- Hardware capture (CPU model, core count, RAM, OS)
- Random seeds set
- Wall-clock per pipeline stage

Reviewers who wish to verify any single number reported in the manuscript
can locate the producing run via the sidecar and reproduce by checking out
the corresponding code state and rerunning. The author retains a local
archive of all run sidecars; on request through the editorial system, any
specific sidecar can be supplied as supplementary material.
