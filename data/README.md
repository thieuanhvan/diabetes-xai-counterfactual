# data/

This directory holds the cleaned BRFSS cohorts used by the pipeline. In the
public GitHub repository the CSV files are excluded by `.gitignore` (the CDC
source remains the authoritative version); in the reviewer archive distributed
through the journal submission portal, both cohorts are bundled here directly.

## Required files

```
data/cdc_brfss_diabetes_2021.csv   # training + internal evaluation
data/cdc_brfss_diabetes_2015.csv   # cross-year external validation (§4.4)
```

Schema (identical for both cohorts): 21 features + `Diabetes_binary` target.
- BRFSS 2021: n = 236,378 records, 14.20% diabetes prevalence
- BRFSS 2015: n = 253,680 records, 13.93% diabetes prevalence

Features follow the julnazz/Teboul Kaggle convention. Expected column order
(`Diabetes_binary` last):

```
HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income,
Diabetes_binary
```

One schema caveat between waves: the `Income` feature is encoded on a 1–11
scale in BRFSS 2021 and a 1–8 scale in BRFSS 2015 (CDC survey instrument
change between waves). Tree-based classifiers handle out-of-range predictor
values without extrapolation, so the encoding shift does not break inference;
it is documented for transparency and discussed in manuscript Section 4.4.

## How to obtain

Both cohorts are bundled in the reviewer archive under this `data/` directory,
so no acquisition step is needed to reproduce the results. The routes below
document how the cohorts were derived, for full transparency and for anyone
working from the public GitHub repository where the CSVs are gitignored.

### Route A — original CDC source + cleaning toolkit (authoritative)

**BRFSS 2021:**
1. Download the BRFSS 2021 raw file from CDC:
   `https://www.cdc.gov/brfss/annual_data/annual_2021.html`
   The relevant artefact is `LLCP2021.XPT` (~71 MB) inside the annual ZIP.
2. Apply the cleaning steps documented in `data/PROVENANCE.md`, which converts
   the XPT to CSV, selects the 21 julnazz-style features, harmonises the
   income encoding and produces `cdc_brfss_diabetes_2021.csv` matching the
   schema above.

**BRFSS 2015:**
1. Download the BRFSS 2015 raw file from CDC:
   `https://www.cdc.gov/brfss/annual_data/annual_2015.html`
   The relevant artefact is `LLCP2015.XPT` inside the annual ZIP.
2. Apply the same cleaning steps (`data/PROVENANCE.md`) to produce
   `cdc_brfss_diabetes_2015.csv`. The only wave-specific difference is the
   `Income` recode (1–8 in 2015 vs 1–11 in 2021), handled by the toolkit.

The cleaned CSVs are byte-identical regardless of who runs the cleaning, given
the same CDC source files.

### Route B — companion Kaggle dataset (after publication)

A companion Kaggle dataset bundling the cleaned 2015 / 2021 / 2023 slices with
a `provenance.json` describing every cleaning step is planned for public
release alongside this paper's acceptance. The dataset is not yet public during
peer review; the bundled CSVs in this archive are the canonical reviewer copy.

## Peer-review access

For this manuscript's peer review (International Journal of Medical
Informatics), both cleaned cohorts are bundled in the reviewer archive
distributed through the submission portal. If a reviewer prefers the original
CDC source plus the cleaning trail instead of the bundled CSVs, the full
derivation is in `data/PROVENANCE.md` and reproduces byte-identical cohorts
from the public CDC `LLCP*.XPT` files.

## CDC BRFSS Terms of Use

BRFSS is a CDC public-release dataset distributed under the CDC public-domain
notice. Redistribution is permitted but the upstream CDC source is treated as
the authoritative version. `data/PROVENANCE.md` documents every transformation
between the upstream XPT and the cleaned CSV.
