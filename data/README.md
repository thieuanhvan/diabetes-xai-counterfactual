# data/

This directory is intentionally empty in the public repository. BRFSS 2021
data files are excluded by `.gitignore` and must be acquired separately
before running the pipeline.

## Required file

```
data/brfss_2021.csv
```

Schema: 21 features + `Diabetes_binary` target, n = 236,378 records, 14.20%
diabetes prevalence. Features follow the julnazz/Teboul Kaggle convention.

Expected column order (`Diabetes_binary` last):

```
HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income,
Diabetes_binary
```

## How to obtain

There are two supported routes:

### Route A — sister Kaggle dataset (recommended after Paper 2 acceptance)

```
https://www.kaggle.com/datasets/thieuanhvan/brfss-diabetes
```

Status: private during Paper 2 review window; expected public 2026 Q3 after
Paper 2 acceptance. The Kaggle dataset bundles the cleaned 2015 / 2021 / 2023
slices with a `provenance.json` describing every cleaning step. Download
`brfss_2021.csv` directly into `data/`.

### Route B — original CDC source + cleaning script

If the sister Kaggle dataset is not yet public:

1. Download the BRFSS 2021 raw file from CDC:
   `https://www.cdc.gov/brfss/annual_data/annual_2021.html`
   The relevant artefact is `LLCP2021.XPT` (~71 MB) inside the annual ZIP.

2. Run the cleaning toolkit from the sister code repository:
   `https://github.com/thieuanhvan/brfss-diabetes`
   The toolkit converts XPT to CSV, selects the 21 julnazz-style features,
   harmonises the income encoding and produces `brfss_2021.csv` matching
   the schema above.

The cleaned CSV is byte-identical regardless of route.

## Peer-review access

For Paper 4 peer review (target submission 01/06/2026, target venue
International Journal of Medical Informatics), the cleaned CSV can be
provided as a journal supplement file on reviewer request through the
submission portal. The author is also happy to share Kaggle dataset access
with the assigned reviewer email on request through the editorial system.

## CDC BRFSS Terms of Use

BRFSS is a CDC public-release dataset distributed under the CDC public-domain
notice. Redistribution is permitted but the upstream CDC source is treated
as the authoritative version. The cleaning script in the sister repository
documents every transformation between the upstream XPT and the cleaned CSV.
