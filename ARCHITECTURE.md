# Architecture

Internal documentation for `diabetes-xai-counterfactual`. Audience: future
maintainer (likely the original author returning to this repo months later for
thesis defense or NCS-phase extension). Not customer-facing.

## Overview

The pipeline runs a single experiment end-to-end:

```
BRFSS 2021 CSV
    ->  data loader  (src/pipelines/data/)
    ->  preprocessing  (src/pipelines/preprocessing/)
    ->  XGBoost training  (src/pipelines/models/)
    ->  DiCE CF generation in two modes  (src/pipelines/counterfactual/)
            global: features_to_vary='all'
            per-query: taxonomy-derived features_to_vary + permitted_range
    ->  CF metric computation  (src/pipelines/evaluate/)
    ->  outputs/ CSVs + JSON sidecar + log
    ->  analysis/make_figures.py        (figures from CSVs)
    ->  analysis/topk_violations.py     (ranked CSV + markdown table)
```

The two-mode comparison is the experimental design that lets per-query
taxonomy be evaluated against a fixed global baseline on identical patients
and identical model. Both modes are deterministic given the seed; running
twice on the same machine produces byte-identical CSV outputs.

## Module responsibilities

`src/pipelines/data/loader.py`: read BRFSS CSV, return X (DataFrame) and y
(Series). Schema fixed: 21 features, Diabetes_binary target. No augmentation
or rebalancing.

`src/pipelines/preprocessing/pipeline.py`: stratified 80/20 split with
`random_state=42`, StandardScaler fitted on train only. No leakage.

`src/pipelines/models/xgb_train.py`: train XGBoost with hyperparameters
copied from Paper 2 run 12 (n_estimators=500, max_depth=6, learning_rate=0.05,
subsample=0.9, colsample_bytree=0.9, tree_method='hist'). Reuse is intentional:
matching baseline lets P4 results be compared against P2 audit findings without
confounding hyperparameter differences.

`src/pipelines/counterfactual/feature_taxonomy.py`: the 5-class taxonomy lives
here. Each feature has one `Mutability` enum value
(IMMUTABLE / MONOTONIC_UP / MONOTONIC_DOWN / BIDIRECTIONAL / CONDITIONAL).
BMI has a clinical `permitted_range` constant [18.5, 35.0]. To extend the
taxonomy: add the feature to `FEATURE_TAXONOMY` dict with appropriate class,
add unit test in `tests/test_feature_taxonomy.py`.

`src/pipelines/counterfactual/dice_runner.py`: wraps DiCE-ML. Two entry
points: `generate_global(x)` and `generate_per_query(x)`. The latter derives
`features_to_vary` and `permitted_range` from the patient baseline and the
taxonomy. Categorical features are configured as continuous in the DiCE Data
object to bypass a known dtype quirk in dice-ml 0.11; integer-typed features
are rounded post-hoc.

`src/pipelines/counterfactual/actionability.py`: computes the actionability
score = 1 - (wrong_direction + immutable + out_of_range) / total_changes.
Linear weighting of three violation types. Returns the aggregate score plus
per-violation counts for downstream debugging.

`src/pipelines/evaluate/cf_metrics.py`: standard CF metrics from the
literature (validity, proximity L1-normalized, sparsity, diversity,
plausibility via kNN k=50).

`src/pipelines/evaluate/per_feature.py`: per-feature breakdown of where
violations occur in each mode. Used by `analysis/topk_violations.py` to rank
features by violation reduction.

`src/utils/run_logger.py`: `setup_run()` opens a log file, captures stdout,
records start timestamp; `finalize_run()` writes the JSON config sidecar with
hardware/libraries/seeds/git-commit (Generalrule v37 sections 11.4 + 11.6).

`src/utils/seed.py`: `seed_everything(42)` for numpy, Python random,
PYTHONHASHSEED, xgboost. Called once at pipeline start.

`analysis/make_figures.py`: reads the most recent `outputs/run_*` files,
emits two PNG+PDF pairs (per-feature breakdown bar chart, comparison metrics
bar chart). Title text inside figures is English by convention; caption with
"Hinh N" or "Figure N" lives in the manuscript, never the image.

`analysis/topk_violations.py`: ranks 21 features by per-mode violation rate
delta, classifies each into one of five patterns (suppressed_entirely,
redirected, no_global_violations, no_changes, etc.), writes both CSV and
markdown table.

## Decision records

### ADR-1: dataset = BRFSS 2021

Considered alternatives: BRFSS 2015 (more years available), NHIS 2019-2024
(in-person interview, different selection process), NHANES 2017-2023
(includes lab-measured BMI and HbA1c).

Decision: BRFSS 2021.

Rationale: Paper 2 used BRFSS 2015/2021/2023; reusing 2021 lets P4 train on
the same data as P2 and report XGBoost AUC matching P2 run 12 baseline
(0.8228 +- float noise). Cross-paper baseline consistency was the deciding
factor over modality coverage (NHANES) or longitudinal breadth (multi-year
BRFSS). NHIS and NHANES are kept available for Paper 6 cross-dataset
extension after P2 acceptance.

Trade-off accepted: self-reported BMI in BRFSS limits clinical fidelity
versus NHANES measured BMI. Documented in main_vi section 5.3 (external
validity).

### ADR-2: classifier = single XGBoost

Considered alternatives: ensemble of LR + RF + XGBoost (matches Paper 2),
single LR (interpretable baseline), single neural net.

Decision: single XGBoost, hyperparameters frozen from Paper 2 run 12.

Rationale: P4 ethics requires no overlap with Paper 2 in framing. P2 uses
three classifiers to audit cross-model agreement. P4 audits cross-mode
actionability on one fixed classifier; the methodological contribution is
the taxonomy and per-query application, not the classifier. Single model
also reduces compute (one CF generation pass per mode versus three).

Trade-off accepted: results are model-specific. Generalization to LR or RF
is left to future work; method comparison (random / genetic / kdtree) was
ablated in §4.5.2 main_vi v8 but cross-classifier generalization to LR / RF
remains open.

### ADR-3: CF method = random

Considered alternatives: DiCE method='genetic' (high validity + zero
violations but ~10x compute cost and sparsity inflated by ~2x) and DiCE
method='kdtree' (retrieval-based, requires training set in memory).

random was chosen as default for three reasons: (i) lowest computational cost
- practical wall-clock ~5 minutes for 200 patients in compare-modes setup;
(ii) no dependency on training set at CF generation time; (iii) yields
acceptable validity (0.808) and high actionability (0.988) when paired with
per-query taxonomy constraints.

Method ablation (§4.5.2 main_vi v8) confirmed the trade-off empirically.
Genetic improves validity to 0.993 and actionability to 1.000 but doubles
sparsity (0.139 vs 0.072) and runs at ~51 minutes; it remains a high-quality
alternative when compute budget is not constrained. kdtree failed
structurally - 0/200 counterfactuals were found in per-query mode because
nearest training neighbours of high-risk queries violate the taxonomy
permitted_range on >=8/21 features simultaneously. A dtype incompatibility
between DiCE 0.12 and pandas 3.x (float32 downcast in
PublicData._set_feature_dtypes vs float64 scalar assignment in
posthoc_sparsity) compounded the issue; the 3-line workaround in
dice_runner.py unblocks the run but does not address the structural failure.

Trade-off accepted: random does not guarantee monotone convergence to a
diverse Pareto front. Genetic is documented as the production alternative.

### ADR-4: taxonomy = 5 classes (not binary)

Considered alternatives: binary mutable / immutable (Wachter, DiCE default),
three kinds (Ustun et al. 2019: immutable, conditionally immutable, mutable),
arbitrary per-feature constraint specification.

Decision: five classes (IMMUTABLE, MONOTONIC_UP, MONOTONIC_DOWN,
BIDIRECTIONAL, CONDITIONAL).

Rationale: binary mutability cannot distinguish 'reduce blood pressure'
(clinically correct) from 'increase blood pressure' (clinically wrong).
Ustun's three kinds is closer but BMI does not fit any of them. BMI requires
two-sided clinical bounds, not a fixed direction. Adding BIDIRECTIONAL gives
BMI a home; adding CONDITIONAL gives DiffWalk a placeholder for future
context-dependent expansion. Five is the minimum that maps cleanly onto
BRFSS 2021's 21 features without forcing any feature into a wrong category.

Trade-off accepted: more classes need more maintenance when adding features.
Mitigated by `tests/test_feature_taxonomy.py` which enumerates every feature
and asserts class membership.

### ADR-5: application = per-query (not global)

Considered alternatives: global features_to_vary applied identically to all
queries, taxonomy-derived but applied with same range for all queries.

Decision: per-query application. For each patient x_i, derive features_to_vary and
permitted_range from x_i baseline values.

Rationale: monotonic constraints depend on the patient's current value.
Example: PhysActivity is MONOTONIC_UP, so the allowed range is
[x_i[PhysActivity], max]. If x_i[PhysActivity] = 1 the range collapses to
[1, 1] and the feature cannot be changed. A global constraint (e.g. always
allow [0, 1]) would let DiCE suggest reducing PhysActivity for the second
patient even though it cannot.

Trade-off accepted: implementation is more complex than a single constraint
applied uniformly. Empirically the per-query mode does not hurt validity
(0.752 -> 0.808 in run 20260513_1935) and eliminates wrong-direction
violations entirely.

### ADR-6: query selection = top-200 by predicted probability

Considered alternatives: full test set (n=47,276), random sample of 1,000,
stratified sample across deciles.

Decision: top-200 patients by predicted P(diabetes=1).

Rationale: counterfactual generation is most clinically relevant for
high-risk patients, who are the ones a clinician would actually be giving
recourse to. Lower-risk patients with predicted probability below the
decision threshold do not need diabetes-prevention counterfactuals; their
model output is already "no diabetes". Computational cost is the second
factor: full-test CF generation at 1.1 seconds per query is roughly 14 hours;
the cohort selection brings total wall-clock to under 8 minutes.

Trade-off accepted: results may not generalize to the full risk distribution.
Class-balance ablation (§4.5.4 main_vi v8) tested thr in {0.0, 0.5, 0.7} and
confirmed taxonomy operates independently of cohort risk profile;
wrong_direction_violations remained 0 across all three thresholds.

### ADR-7: repo layout = src/pipelines + src/utils + analysis

Considered alternatives: flat `src/` with all modules at one level (the
original layout up to 14/05/2026), nested per-paper folders, monorepo with
P2 and P4 sharing code.

Decision: src/pipelines/ for pipeline steps, src/utils/ for shared
utilities, analysis/ for post-pipeline scripts.

Rationale: pipeline steps and analysis scripts have different lifecycles.
Pipeline steps are core code with unit tests and are run by run_main.py once
per experiment. Analysis scripts are derived computations on pipeline outputs;
they can be re-run any time on an existing run without re-training and they
write back to outputs/. Separating them prevents accidentally treating
analysis as a pipeline step. src/utils/ stays at src level (not under
pipelines/) because logger and seed are shared by both pipelines and analysis.

Trade-off accepted: one more directory level for pipeline modules. Mitigated
by Python's package mechanism. Imports read naturally as
`from src.pipelines.evaluate.cf_metrics import ...`.

### ADR-8: reproducibility = log + JSON sidecar + hardware

Considered alternatives: log file only, requirements.txt pinning only, MLflow
tracking, no reproducibility artifacts.

Decision: per-run log + per-run JSON sidecar following Generalrule v37
sections 11.4 (config schema with 8 fields) and 11.6 (hardware sub-dict).
Both files live in outputs/.

Rationale: a reviewer reproducing the experiment needs four things to
diagnose any discrepancy with the paper's reported numbers: (1) library
versions, since numpy 1.26 versus 2.0 changes some edge cases; (2) seeds,
to verify determinism setup; (3) hardware, to know whether a 30-second
timing difference is code drift or a slower CPU; (4) git commit, to locate
the exact code revision. The JSON sidecar captures all four in a parseable
format that diffs cleanly when comparing two runs.

Trade-off accepted: extra ~5 KB per run for the JSON. Negligible. The 8-field
schema is the same one used by Paper 2; reusing it lets cross-paper
reproducibility be checked with the same tooling.
