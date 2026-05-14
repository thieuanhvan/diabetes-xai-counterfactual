# diabetes-xai-counterfactual

Counterfactual explanations and actionability evaluation for type-2 diabetes
risk prediction on BRFSS 2021.

## Setup

Requires Python 3.12+.

```bash
pip install -r requirements.txt
```

## Configure

Edit `configs/default.yaml` and set `paths.data_csv` to your BRFSS 2021 CSV.

## Run

```bash
python run_main.py
```

This trains XGBoost, generates counterfactuals in two modes (global baseline +
per-query taxonomy), writes metrics to `outputs/`, then chains the analysis
scripts (`analysis/make_figures.py`, `analysis/topk_violations.py`) which add
figures and a ranked violation table.

To run analysis on an existing run without re-training, call the analysis
scripts directly:

```bash
python analysis/make_figures.py
python analysis/topk_violations.py
```

## Layout

```
run_main.py             pipeline wrapper at repo root
configs/                YAML configs (paths, hyperparameters, DiCE settings)
data/                   place BRFSS 2021 CSV here (gitignored)
src/
  pipelines/
    data/               BRFSS loading
    preprocessing/      train/test split, scaling
    models/             XGBoost training
    counterfactual/     DiCE wrapper + feature taxonomy + actionability score
    evaluate/           CF metrics: validity, proximity, sparsity, diversity, plausibility
    main.py             pipeline entry called by run_main.py
  utils/                shared utilities (run_logger, seed) used across pipelines
analysis/               post-pipeline scripts (figures, top-k violations)
tests/                  unit tests for taxonomy + per-feature breakdown
outputs/                run artifacts (gitignored)
```

## Reproducibility

Each authoritative run emits four artifacts to `outputs/`:

- `run_YYYYMMDD_HHMM.log`: timing milestones and stdout
- `run_YYYYMMDD_HHMM.json`: runtime, library versions, seeds, dataset shape,
  hyperparameters, git commit, hardware (cpu/ram/gpu), timestamps
- `run_YYYYMMDD_HHMM_*_cf_metrics.csv`: CF metrics per query
- `run_YYYYMMDD_HHMM_comparison.csv`: aggregated metrics, global vs per-query

To verify reproducibility on a different machine: clone, install requirements
matching `outputs/run_*.json` library versions, set `random_state=42` (default),
and run. Expect numerical metrics within float-precision tolerance (AUC drift
under 0.001 from thread scheduling).

## Tests

```bash
pytest tests/
```

## License

Private repository. Will be made public after paper acceptance.
