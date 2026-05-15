# diabetes-xai-counterfactual

Counterfactual explanations with actionability constraints for type-2 diabetes risk prediction. Reference implementation for *"Per-Query Actionability Taxonomy for Counterfactual Explanations in Diabetes Risk Prediction"* (in preparation, target venue: International Journal of Medical Informatics, target submission 01/06/2026).

## What this does

Generates counterfactual explanations (CFs) for high-risk patients in BRFSS 2021, using a 5-class intervention-direction taxonomy that encodes clinical actionability semantics without requiring a full structural causal model. Compares two CF generation modes:

- **Global** mode — DiCE with `features_to_vary='all'` (baseline).
- **Per-query** mode — taxonomy-derived `features_to_vary` + `permitted_range` per patient.

Per-query mode improves actionability score from 0.666 → 0.988 (+48.5% relative) with zero counter-clinical and zero immutable-feature violations.

## Quick start

```bash
git clone https://github.com/thieuanhvan/diabetes-xai-counterfactual.git
cd diabetes-xai-counterfactual

python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows

pip install -r requirements.txt
```

Place the BRFSS 2021 CSV at `data/brfss_2021.csv` (schema: 21 features + `Diabetes_binary` target, n=236,378). The cleaning script is in `thieuanhvan/brfss-diabetes` (separate sister repo).

## Reproduce the baseline run

```bash
python run_main.py
```

Generates `outputs/run_<YYYYMMDD_HHMM>.{log,json}` plus `_cf_metrics.csv`. After the pipeline, `make_figures.py` and `topk_violations.py` run automatically (best-effort; pipeline outputs are valid even if these fail).

Expected wall-clock: ~10 minutes on Intel i7 Ice Lake 8-core, 32 GB RAM, no GPU.

Expected key metrics (deterministic with `seed=42`):

| Metric | Value |
|---|---|
| Classifier AUC (test) | 0.8233 |
| Validity (per-query) | 0.808 |
| Actionability score (per-query) | 0.988 |
| Wrong-direction violations | 0.000 |
| Immutable violations | 0.000 |

## Reproduce the ablation studies

```bash
python run_ablation_all.py
```

Runs all 5 ablation grids sequentially (~3 hours total compute):

1. **Multi-seed variance** — 5 seeds `{42, 123, 2024, 7, 31337}`
2. **Method comparison** — `random` / `genetic` / `kdtree`
3. **n_counterfactuals sensitivity** — `n ∈ {1, 3, 5, 10}`
4. **Class-balance / risk threshold cohort** — `≥ 0.0` / `0.5` / `0.7`
5. **Taxonomy granularity** — 5-class vs 4-class collapsed

Aggregate results into comparison tables:

```bash
python -m ablation.aggregate seed
python -m ablation.aggregate method
python -m ablation.aggregate n_cf
python -m ablation.aggregate class
python -m ablation.aggregate taxonomy
```

Each command writes `outputs/ablation_<type>_table.csv` summarizing the corresponding ablation. The aggregator filters runs via the `ablation=<type>` marker in `notes_suffix` (set by `run_ablation_all.py`) with a fallback dedup-by-latest path for runs predating the marker convention.

## Repository layout

```
diabetes-xai-counterfactual/
├── run_main.py                    # baseline + analysis wrapper (§11.5)
├── run_ablation_all.py            # master ablation wrapper (§11.5)
├── run_method_kdtree.py           # one-shot kdtree rerun (post dtype fix)
├── requirements.txt               # pinned versions for reproducibility
├── configs/default.yaml           # pipeline config (deterministic, seed=42)
├── data/                          # BRFSS CSV (gitignored, see brfss-diabetes repo)
├── src/
│   ├── pipelines/
│   │   ├── data/loader.py         # BRFSS 21-feature schema
│   │   ├── preprocessing/         # stratified 80/20 + StandardScaler
│   │   ├── models/xgb_train.py    # XGBoost matching P2 baseline
│   │   ├── counterfactual/
│   │   │   ├── feature_taxonomy.py  # 5-class taxonomy
│   │   │   ├── actionability.py     # actionability score formula
│   │   │   └── dice_runner.py       # DiCE wrapper + kdtree dtype workaround
│   │   ├── evaluate/              # CF metrics + per-feature breakdown
│   │   └── main.py                # orchestrator
│   └── utils/                     # logger (§11.4), seed (§11.5), hardware (§11.6)
├── analysis/
│   ├── make_figures.py            # figures from comparison.csv
│   ├── comparison_metrics.py      # global vs per-query Δ
│   ├── per_feature_actionability.py
│   └── topk_violations.py
├── ablation/
│   ├── core.py                    # grid runner
│   └── aggregate.py               # builds ablation_*_table.csv
├── tests/                         # taxonomy + per-feature unit tests
└── outputs/                       # run artifacts (§11.4 sidecar JSON + CSV + log)
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

BRFSS 2021 (n = 236,378 records, 21 features, 14.20% diabetes prevalence). Cleaning toolkit lives in the separate repository `thieuanhvan/brfss-diabetes` (private during P2 review window, expected public 2026 Q3 after P2 acceptance). Schema mirrors the julnazz/Teboul Kaggle convention with `Diabetes_binary` as target.

## Citation

```bibtex
@article{thieu2026p4,
  title  = {Per-Query Actionability Taxonomy for Counterfactual Explanations in Diabetes Risk Prediction},
  author = {Thieu, Anh Van},
  journal= {International Journal of Medical Informatics},
  year   = {2026},
  note   = {Manuscript in preparation, target submission 01/06/2026}
}
```

## License

Code: MIT (planned, finalized at publication). Data redistribution governed by CDC BRFSS terms of use.

## Contact

Thiều Anh Vân — thieuanhvan@gmail.com — ORCID: 0009-0003-9637-0195
