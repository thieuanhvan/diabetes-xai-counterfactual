# Reproducibility

This document specifies the exact environment, data, seeds and expected
numerical outputs needed to reproduce the results reported in
*"Per-Query Actionability Taxonomy for Counterfactual Explanations in
Diabetes Risk Prediction"* (target venue: International Journal of Medical
Informatics, target submission 01/06/2026).

Companion file `README.md` is the high-level entry point; this file is the
rigorous step-by-step for reviewers and future maintainers.

## TL;DR

```bash
git clone https://github.com/thieuanhvan/diabetes-xai-counterfactual.git
cd diabetes-xai-counterfactual
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Place brfss_2021.csv into data/ — see data/README.md
python run_main.py
```

Expected wall-clock: ≈10 minutes on Intel i7 Ice Lake 8-core. Outputs land in
`outputs/run_<YYYYMMDD_HHMM>.{log,json,csv}`. Compare the numerical values
against the "Expected numerical outputs" table below.

## 1. Environment

| Component | Pinned version | Source |
|---|---|---|
| Python | 3.11.x | python.org |
| numpy | 1.26.4 | `requirements.txt` |
| pandas | 2.2.2 | `requirements.txt` |
| scikit-learn | 1.5.1 | `requirements.txt` |
| xgboost | 2.0.3 | `requirements.txt` |
| dice-ml | 0.12 | `requirements.txt` |
| matplotlib | 3.9.1 | `requirements.txt` |

The full pinned list is in `requirements.txt`. Each `outputs/run_*.json`
sidecar records the actual versions present at runtime (per Generalrule v37
§11.4), so a reviewer can verify environment fidelity by diffing the sidecar
against this table.

Operating system tested: **Windows 11 (primary author environment)**.
Linux and macOS are expected to work; the only OS-dependent code is path
handling, which uses `pathlib` throughout.

## 2. Data

See `data/README.md`. The pipeline expects `data/brfss_2021.csv` with the
21-feature julnazz/Teboul schema and `Diabetes_binary` as target column.
Data files are not committed (CDC source remains authoritative; sister
Kaggle dataset `thieuanhvan/brfss-diabetes` goes public after Paper 2
acceptance).

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
| StandardScaler | deterministic (no RNG) |
| XGBoost | `random_state=42`, `tree_method='hist'` |
| DiCE `generate_counterfactuals` | `random_seed=42` |

All seeding goes through `src/utils/seed.py::seed_everything(42)`, called
once at the top of `run_main.py`. Running the pipeline twice on the same
machine and OS produces byte-identical CSV outputs.

Multi-seed variance ablation uses `{42, 123, 2024, 7, 31337}` and is the
empirical evidence for the determinism claim across seeds.

## 4. Hardware tested

The authoritative run was carried out on:

| Spec | Value |
|---|---|
| CPU | Intel Core i7 (Ice Lake, 8 cores) |
| RAM | 32 GB |
| GPU | none used |
| Disk | local SSD |
| OS | Windows 11 |

Wall-clock figures below assume comparable hardware. The pipeline is
CPU-bound (XGBoost `tree_method='hist'` + DiCE `random` method); no GPU
acceleration is used or required. Each `outputs/run_*.json` sidecar records
the actual hardware at runtime (per Generalrule v37 §11.6).

## 5. Reproduction steps

### 5.1 Baseline (recommended starting point)

```bash
python run_main.py
```

This runs the full pipeline end-to-end:

1. Load BRFSS 2021 CSV from `data/`.
2. Stratified 80/20 train/test split, StandardScaler fit on train.
3. Train XGBoost (`n_estimators=500`, `max_depth=6`, `learning_rate=0.05`,
   `subsample=0.9`, `colsample_bytree=0.9`, `tree_method='hist'`).
4. Select top-200 high-risk patients by predicted `P(diabetes=1)`.
5. Generate counterfactuals in both modes (global, per-query) via DiCE
   method='random', `n_CF=5` per query.
6. Compute CF metrics (validity, proximity, sparsity, plausibility,
   actionability) per mode.
7. Run analysis modules: `make_figures.py`, `topk_violations.py`,
   `per_feature_actionability.py`, `comparison_metrics.py`.

Expected wall-clock: **≈10 minutes** on the reference hardware
(≈4 min global + ≈5 min per-query + ≈1 min analysis).

### 5.2 Ablations

```bash
python run_ablation_all.py
```

Sequentially runs the five ablation grids (multi-seed, method,
n_counterfactuals, class-balance / risk threshold, taxonomy granularity).
Total wall-clock: **≈3 hours** on the reference hardware. After the grids
complete, aggregate into comparison tables:

```bash
python -m ablation.aggregate seed
python -m ablation.aggregate method
python -m ablation.aggregate n_cf
python -m ablation.aggregate class
python -m ablation.aggregate taxonomy
```

Outputs: `outputs/ablation_<type>_table.csv`.

### 5.3 Smoke test

```bash
python run_main.py --smoke
```

Runs the pipeline on a 1,000-row subsample with 20 queries instead of 200.
Wall-clock ≈30 seconds. Useful for verifying environment install before
committing to a 10-minute baseline run. Outputs land in
`outputs/run_scratch_<timestamp>/` and are auto-excluded from version
control via the `_scratch_` marker.

## 6. Expected numerical outputs

The authoritative reference run is `run_20260516_1030`. Values below are
the headline numbers reported in the manuscript. Reproduction should match
to all displayed decimal places (deterministic pipeline).

### 6.1 Classifier

| Quantity | Expected value | Source file |
|---|---|---|
| Test AUC | **0.8233** | `outputs/run_*/classifier_metrics.csv` |
| Test accuracy @ threshold 0.5 | 0.8595 | same |
| Train/test split sizes | 189,102 / 47,276 | log line |
| Class prevalence (full) | 0.1420 | log line |

### 6.2 CF metrics — global mode

| Quantity | Expected value |
|---|---|
| Validity | 0.752 |
| Proximity ($L_1$) | 1.397 |
| Sparsity | 0.067 |
| Diversity | 15.194 |
| Plausibility ($k$=50) | 4.299 |
| **Actionability** | **0.666** |
| Wrong-direction violation rate | 0.435 |
| Immutable violation rate | 0.000 |

### 6.3 CF metrics — per-query mode

| Quantity | Expected value |
|---|---|
| Validity | 0.808 |
| Proximity ($L_1$) | 1.471 |
| Sparsity | 0.072 |
| Diversity | 14.710 |
| Plausibility ($k$=50) | 4.030 |
| **Actionability** | **0.988** |
| Wrong-direction violation rate | 0.000 |
| Immutable violation rate | 0.000 |

Total CF changes across all features: **1,411 (global) → 1,520 (per-query, +7.7%)**.
Verify via `outputs/run_*/per_feature.csv` by summing `n_total_cf_changes` per mode.

### 6.4 Per-feature breakdown (Pattern (a) check)

In `outputs/run_*/per_feature.csv`, three features should record 100%
wrong-direction violations under global mode and 0 changes under per-query
mode (suppressed by taxonomy):

- `AnyHealthcare`
- `CholCheck`
- `HvyAlcoholConsump`

A further eleven features (Pattern (b) — redirected): `NoDocbcCost`,
`Education`, `Smoker`, `Fruits`, `MentHlth`, `Veggies`, `PhysActivity`,
`PhysHlth`, `Income`, `GenHlth`, `HighChol` — record CF changes in both
modes with zero wrong-direction violations under per-query mode.

### 6.5 Representative patient (Patient 168)

Documented in manuscript §4 as a worked example:

| Feature | Baseline | Global CF Δ | Per-query CF Δ |
|---|---|---|---|
| Age band | 45–49 | — (immutable) | — (immutable) |
| BMI | 36 | 25 | 27 |
| HighBP | 1 | 0 | 0 |
| PhysActivity | 0 | 1 | 1 |
| HvyAlcoholConsump | 0 | 1 (wrong-dir) | — (suppressed) |

The exact numerical values are in `outputs/run_*/patient_examples.json`.

## 7. Verifying reproduction

After running, compare your outputs against the manuscript values:

```bash
# Pick the latest run directory
RUN_DIR=$(ls -d outputs/run_2026* | tail -1)

# Quick check — classifier AUC
grep "test AUC" "$RUN_DIR".log

# Headline CF metrics
cat "$RUN_DIR/cf_metrics.csv"

# Per-feature pattern
head -22 "$RUN_DIR/per_feature.csv"
```

A successful reproduction shows:
- AUC == 0.8233 (exact match, deterministic)
- Per-query actionability == 0.988 (exact match)
- Per-query wrong-direction violation rate == 0.000 (exact match)
- Three Pattern (a) features all show 100% / 0 split

If any value drifts, see Troubleshooting below.

## 8. Known sources of variation

By design, **none**. The pipeline is fully deterministic given the seed,
the pinned versions, and the data file. The following are *not* sources
of variation:

- Run timestamp (only affects directory name, not values)
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

**AUC differs from 0.8233.** Check `outputs/run_*.json` sidecar against the
environment table in §1. The most common cause is a numpy or xgboost
version mismatch from a stale virtual environment.

**`KeyError` on column load.** The BRFSS CSV must contain exactly the 22
columns listed in `data/README.md`. The cleaning script in the sister
repository `thieuanhvan/brfss-diabetes` produces the correct schema.

**`TypeError: Invalid value <x> for dtype 'float32'` during kdtree method
ablation.** Known DiCE 0.12 + pandas 3.x incompatibility; workaround is
applied in `src/pipelines/counterfactual/dice_runner.py` (continuous columns
recast to float64 in `DiCE Data` constructor). See README §Known issues for
the upstream issue references.

**Out-of-memory during ablations.** The ablation grids serialise to disk
between runs but still hold the test set + XGBoost model in memory.
On a 16 GB machine, run the grids one at a time
(`python run_ablation_<type>.py`) instead of `run_ablation_all.py`.

**Wall-clock far exceeds estimates.** Confirm `tree_method='hist'` is in
effect (XGBoost defaults to `exact` on some installations). Confirm
matplotlib backend is not opening interactive windows that block the run.

## 10. Provenance

Authoritative reference run: `run_20260516_1030`. The associated
`outputs/run_20260516_1030.json` sidecar records:

- Exact library versions present at the time of authoritative-run execution
- Git commit hash of the code revision used
- Hardware capture (CPU model, core count, RAM, OS)
- Random seeds set
- Wall-clock per pipeline stage

Reviewers who wish to verify any single number reported in the manuscript
can locate the producing run via the sidecar's `git_commit` and reproduce
by checking out that commit and rerunning. The author also retains a local
archive of all run sidecars; on request through the editorial system, any
specific sidecar can be supplied as supplementary material.
