# diabetes-xai-counterfactual

Counterfactual explanations and actionability evaluation for type-2 diabetes
risk prediction on BRFSS 2021.

## Setup

```bash
pip install -r requirements.txt
```

## Configure

Edit `configs/default.yaml` and set `paths.data_csv` to your BRFSS 2021 CSV.

## Run

```bash
python -m src.main --config configs/default.yaml
```

## Layout

```
configs/        YAML configs (paths, hyperparams, DiCE settings)
src/data/       BRFSS loading
src/preprocessing/   train/test split
src/models/     XGBoost training
src/counterfactual/  DiCE wrapper + feature taxonomy + actionability
src/evaluate/   CF metrics (validity, proximity, sparsity, diversity, plausibility)
src/utils/      seeding
src/main.py     pipeline entry
tests/          unit tests
outputs/        run artifacts (gitignored)
```
