# diabetes-xai-counterfactual

Counterfactual explanations with directional actionability constraints for type-2 diabetes risk prediction. Reference implementation for *"A Knowledge-Guided Constraint Compiler for Actionable Counterfactual Explanations in Diabetes Risk Prediction"* (manuscript under peer review).

## What this does

Generates counterfactual explanations (CFs) for high-risk patients in BRFSS 2021, using a 5-class intervention-direction taxonomy that encodes clinical actionability semantics without requiring a full structural causal model. Compares two CF generation modes:

- **Global** mode — DiCE with a single `features_to_vary` list applied to every query. The list already excludes four immutable features (Age, Sex, prior Stroke, prior HeartDiseaseorAttack), so the comparison isolates the directional refinement from the standard immutable exclusion present in current CF tooling.
- **Per-query** mode — taxonomy-derived `features_to_vary` + `permitted_range` per patient, layered on the same immutable exclusion.

Per-query mode improves actionability score from 0.666 → 0.988 (+48.5% relative) with zero wrong-direction and zero immutable-feature violations on BRFSS 2021.

## Quick start

```bash
git clone https://github.com/thieuanhvan/diabetes-xai-counterfactual.git
cd diabetes-xai-counterfactual

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

pip install -r requirements.txt
```

Place the BRFSS 2021 CSV at `data/cdc_brfss_diabetes_2021.csv` (schema: 21 features + `Diabetes_binary` target, n=236,378). See `data/README.md` for acquisition details. The cleaning toolkit lives in the separate sister repository `thieuanhvan/brfss-diabetes`.

## Authoritative results

The metrics reported in the manuscript (Tables 4 and 5, Sections 4.3 and 4.4) reproduce from the frozen reference at:

```
outputs/archive/manuscript/
├── comparison.csv          ← Table 4 (headline comparison)
├── global_cf_metrics.csv
├── perquery_cf_metrics.csv
├── per_feature.csv          ← Table 5 (per-feature breakdown)
├── config.json              ← full runtime/library/seed audit
└── manifest.json
```

Expected key metrics from the authoritative reference (`outputs/archive/manuscript/comparison.csv`, sourced from the `class_general` cell of the ablation grid, run_id `run_20260517_1716`):

| Metric | Global | Per-query |
|---|---|---|
| Classifier AUC (test) | 0.8233 | 0.8233 |
| Validity | 0.752 | 0.808 |
| Actionability score | 0.6655 | 0.9880 |
| Wrong-direction violations | 0.435 | 0.000 |
| Immutable violations | 0.000 | 0.000 |

## Reproduce the baseline run

```bash
python run_main.py
```

Outputs (scratch space — overwritten on each run):

- `outputs/scratch/run_<YYYYMMDD_HHMM>.{log,json}` — sidecar log + config audit trail.
- `outputs/scratch/{comparison,global_cf_metrics,perquery_cf_metrics,per_feature}.csv` — current run results.
- `outputs/scratch/fig_*.{png,pdf}` and `topk_violations.{csv,md}` — analysis outputs.

Expected wall-clock: ~30 minutes on Intel i7 Ice Lake 8-core, 32 GB RAM, no GPU.

To verify reproduction, compare `outputs/scratch/comparison.csv` against `outputs/archive/manuscript/comparison.csv`. The two should match to within the multi-seed variance reported in Section 4.5.1 of the manuscript (CV ≤ 0.7% on validity, ≤ 0.7% on actionability across 5 seeds).

See `REPRODUCIBILITY.md` for the full expected-output table and verification commands.

## Reproduce the ablation studies

```bash
python run_ablation_all.py
```

Runs all 5 ablation grids sequentially in cheap-first order (~6 hours total compute on reference hardware):

1. **Taxonomy granularity** — 5-class vs 4-class collapsed
2. **Class-balance / risk threshold cohort** — `≥ 0.0` / `0.5` / `0.7`
3. **n_counterfactuals sensitivity** — `n ∈ {1, 3, 5, 10}`
4. **Multi-seed variance** — 5 seeds `{42, 123, 2024, 7, 31337}`
5. **Method comparison** — `random` / `genetic` / `kdtree`

A sixth ablation (random-sample cohort, manuscript §4.5.4) is invoked separately because it draws an independent 100-patient cohort:

```bash
python ablation/run_random_sample.py
```

Output: `outputs/random_sample/{comparison, per_feature, global_cf_metrics, perquery_cf_metrics}.csv`. Wall-clock ~5 min. The cohort selection uses a dedicated `COHORT_SEED = 42` declared at the top of the script so the random draw is reproducible regardless of upstream RNG state.

Aggregate the resulting per-cell runs into comparison tables:

```bash
python run_ablation_aggregate.py
```

This wrapper invokes the 5 aggregations in cheap-first order and writes `outputs/scratch/ablation_<type>_table.csv` per ablation. The aggregator filters runs via the `ablation=<type>` marker in `notes_suffix` with a fallback dedup-by-latest path for runs predating the marker convention.

(Equivalent manual invocation: `python -m ablation.aggregate <type>` per ablation type, in any order.)

## Repository layout

```
diabetes-xai-counterfactual/
├── run_main.py                    # baseline + analysis chained
├── run_ablation_all.py            # master ablation wrapper, 5 grids
├── run_ablation_aggregate.py      # aggregate 5 ablation tables
├── requirements.txt               # pinned versions for reproducibility
├── configs/default.yaml           # pipeline config (deterministic, seed=42)
├── data/                          # BRFSS CSV location (gitignored; see data/README.md)
├── src/
│   ├── pipelines/
│   │   ├── data/loader.py         # BRFSS 21-feature schema
│   │   ├── preprocessing/         # stratified 80/20 split (no scaling)
│   │   ├── models/xgb_train.py    # XGBoost
│   │   ├── counterfactual/
│   │   │   ├── feature_taxonomy.py  # 5-class taxonomy
│   │   │   ├── actionability.py     # actionability score formula
│   │   │   └── dice_runner.py       # DiCE wrapper + kdtree dtype workaround
│   │   ├── evaluate/              # CF metrics + per-feature breakdown
│   │   └── main.py                # orchestrator
│   └── utils/                     # logger, seed, hardware capture
├── analysis/
│   ├── make_figures.py            # figures from comparison.csv
│   ├── comparison_metrics.py      # global vs per-query Δ
│   ├── per_feature_actionability.py
│   └── topk_violations.py
├── ablation/
│   ├── core.py                    # grid runner
│   └── aggregate.py               # builds ablation_*_table.csv
├── tests/                         # taxonomy + per-feature unit tests
└── outputs/
    ├── archive/
    │   └── manuscript/         # frozen reference — manuscript Tables 4 and 5
    ├── _ablation_archive/          # 18 ablation cells (all snapshots from §4.5)
    ├── ablation_*_table.csv        # aggregated ablation summary tables
    └── scratch/                    # working dir — overwritten on each run
```

## Known issues

**DiCE 0.12 + pandas 3.x dtype incompatibility on `kdtree` method.** `dice_ml.data.public_data_interface.PublicData._set_feature_dtypes` casts continuous features to `float32`. `posthoc_sparsity_enhancement` then assigns a `float64` scalar via `.at[]` indexing, which pandas 3.x rejects with strict-cast `TypeError: Invalid value <x> for dtype 'float32'`. The `random` and `genetic` methods are unaffected because they build candidate counterfactuals from `query_instance.values` (auto-`float64`).

Workaround applied in `src/pipelines/counterfactual/dice_runner.py` (`DiCERunner.__init__`):

```python
for col in self.dice_data.continuous_feature_names:
    self.dice_data.data_df[col] = self.dice_data.data_df[col].astype("float64")
```

Upstream issue: `dice-ml` GitHub #423 / #445 (not yet patched as of v0.12).

## Dataset

BRFSS 2021 (n = 236,378 records, 21 features, 14.20% diabetes prevalence). The cleaned slice used here mirrors the julnazz/Teboul Kaggle convention with `Diabetes_binary` as target. Cleaning toolkit lives in the separate repository `thieuanhvan/brfss-diabetes` (private during peer review; will be released alongside this paper's acceptance). For reviewer access during review, the cleaned CSV can be supplied as a journal supplement on request through the submission portal.

See `data/README.md` for full acquisition instructions and `REPRODUCIBILITY.md` for the runtime environment specification.

## Citation

```bibtex
@unpublished{thieu2026p4,
  title  = {A Knowledge-Guided Constraint Compiler for Actionable Counterfactual Explanations in Diabetes Risk Prediction},
  author = {Van Thieu},
  year   = {2026},
  note   = {Manuscript under peer review}
}
```

## License

Source code: MIT (planned, finalized at publication).

BRFSS 2021 microdata are publicly released by the U.S. Centers for Disease Control and Prevention under U.S. federal public-domain terms. The cleaned processed cohort used here is redistributed with full attribution to CDC and remains subject to the CDC public-use terms of the original release; no additional license is asserted over the underlying data.

## Contact

Van Thieu — thieuanhvan@gmail.com — ORCID: 0009-0003-9637-0195
