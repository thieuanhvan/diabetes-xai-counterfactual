# diabetes-xai-counterfactual

Counterfactual explanations with directional actionability constraints for type-2 diabetes risk prediction. Reference implementation for *"Knowledge-Guided Counterfactual Explanations for Diabetes Risk Decision Support: A Directional Intervention Taxonomy"* (manuscript under peer review).

## What this does

Generates counterfactual explanations (CFs) for high-risk patients in BRFSS 2021, using a 5-class intervention-direction taxonomy that encodes clinical actionability semantics without requiring a full structural causal model. Compares two CF generation modes:

- **Global** mode — DiCE with a single `features_to_vary` list applied to every query. The list already excludes four immutable features (Age, Sex, prior Stroke, prior HeartDiseaseorAttack), so the comparison isolates the directional refinement from the standard immutable exclusion present in current CF tooling.
- **Per-query** mode — taxonomy-derived `features_to_vary` + `permitted_range` per patient, layered on the same immutable exclusion.

Per-query mode improves actionability score from 0.666 → 0.988 (+48.5% relative) with zero wrong-direction and zero immutable-feature violations on BRFSS 2021.

## Quick start

This repository is provided as a zip archive through the journal submission portal. After extracting the archive:

```bash
cd diabetes-xai-counterfactual

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

pip install -r requirements.txt
```

The BRFSS 2021 and BRFSS 2015 cleaned cohorts are bundled in the same archive under `data/` (`data/cdc_brfss_diabetes_2021.csv` and `data/cdc_brfss_diabetes_2015.csv`). See `data/README.md` for the schema and `data/PROVENANCE.md` for the full cleaning trail from the original CDC source files.

## Reproduction entry points (run in this order)

1. `python run_main.py`        — START HERE. Reproduces the headline results
                                  (Tables 6 and 7) plus figures. ~10 min.
2. `python analysis/external_validation_brfss2015.py`
                                — External validation (Table 5). ~9 min.
3. `python run_ablation_all.py` — All 5 ablation grids (Section 4.7 +
                                  Appendix D). ~6 hours. Optional.
4. `python run_ablation_aggregate.py`
                                — Build ablation summary tables (after step 3).

Utility scripts `ablation_taxonomy_rerun.py` and `ablation_method_kdtree.py`
re-run individual ablation cells already covered by step 3; reviewers do not
need to run them. To verify the paper's central claim, step 1 alone is
sufficient.

## Authoritative results

The metrics reported in the manuscript (Tables 6 and 7, Sections 4.5 and 4.6) reproduce from the frozen reference at:

```
outputs/archive/manuscript/
├── comparison.csv          ← Table 6 (headline comparison)
├── global_cf_metrics.csv
├── perquery_cf_metrics.csv
├── per_feature.csv          ← Table 7 (per-feature breakdown)
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

To verify reproduction, compare `outputs/scratch/comparison.csv` against `outputs/archive/manuscript/comparison.csv`. The two should match to within the multi-seed variance reported in Section 4.7.1 of the manuscript (CV ≤ 0.7% on validity, ≤ 0.7% on actionability across 5 seeds).

See `REPRODUCIBILITY.md` for the full expected-output table and verification commands.

## Reproduce the external validation (BRFSS 2015)

```bash
python analysis/external_validation_brfss2015.py
```

Trains on BRFSS 2021 (identical seed and hyperparameters as the baseline run), applies the trained classifier directly to BRFSS 2015 without recalibration, and generates counterfactuals on the top-200 high-risk BRFSS 2015 patients under both modes. Output: `outputs/external_2015/`. Wall-clock ~5–9 min. Headline numbers (external AUC 0.827, per-query wrong-direction violations 0.000) match manuscript Section 4.4, Table 5.

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

A sixth ablation (random-sample cohort, manuscript Section 4.7.4) is invoked separately because it draws an independent 100-patient cohort:

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
├── data/                          # BRFSS 2021 + 2015 cohorts (bundled; see data/README.md)
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
│   ├── external_validation_brfss2015.py  # cross-year external validation (§4.4)
│   └── topk_violations.py
├── ablation/
│   ├── core.py                    # grid runner
│   └── aggregate.py               # builds ablation_*_table.csv
├── tests/                         # taxonomy + per-feature unit tests
└── outputs/
    ├── archive/
    │   └── manuscript/         # frozen reference — manuscript Tables 6 and 7
    ├── _ablation_archive/          # 18 ablation cells (all snapshots from §4.7)
    ├── ablation_*_table.csv        # aggregated ablation summary tables
    └── scratch/                    # working dir — overwritten on each run
```

## Known issues

**DiCE 0.12 + pandas 3.x dtype incompatibility on `kdtree` method.** `dice_ml.data.public_data_interface.PublicData._set_feature_dtypes` casts continuous features to `float32`. `posthoc_sparsity_enhancement` then assigns a `float64` scalar via `.at[]` indexing, which pandas 3.x rejects with strict-cast `TypeError: Invalid value <x> for dtype 'float32'`. The `random` and `genetic` methods are unaffected because they build candidate counterfactuals from `query_instance.values` (auto-`float64`).

Workaround applied in `src/pipelines/counterfactual/dice_runner.py` (`DiCERunner.__init__`):

```text
# src/pipelines/counterfactual/dice_runner.py, inside DiCERunner.__init__:
for col in self.dice_data.continuous_feature_names:
    self.dice_data.data_df[col] = self.dice_data.data_df[col].astype("float64")
```

Upstream issue: `dice-ml` GitHub #423 / #445 (not yet patched as of v0.12).

## Dataset

BRFSS 2021 (n = 236,378 records, 21 features, 14.20% diabetes prevalence) for training and internal evaluation; BRFSS 2015 (n = 253,680) for cross-year external validation. The cleaned slices follow the julnazz/Teboul Kaggle convention with `Diabetes_binary` as target. Both cohorts are bundled in this archive under `data/`. The full cleaning trail from the original CDC `LLCP*.XPT` source files is documented in `data/PROVENANCE.md`.

See `data/README.md` for the schema and acquisition routes, and `REPRODUCIBILITY.md` for the runtime environment specification.

## Citation

```bibtex
@unpublished{thieu2026p4,
  title  = {Knowledge-Guided Counterfactual Explanations for Diabetes Risk Decision Support: A Directional Intervention Taxonomy},
  author = {Van Thieu},
  year   = {2026},
  note   = {Manuscript under peer review}
}
```

## License

Source code: MIT (planned, finalized at publication).

BRFSS 2021 and 2015 microdata are publicly released by the U.S. Centers for Disease Control and Prevention under U.S. federal public-domain terms. The cleaned processed cohorts used here are redistributed with full attribution to CDC and remain subject to the CDC public-use terms of the original release; no additional license is asserted over the underlying data.

## Contact

Van Thieu — vantv.20@grad.uit.edu.vn — ORCID: 0009-0003-9637-0195
