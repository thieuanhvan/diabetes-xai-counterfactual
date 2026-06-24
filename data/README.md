# data/

This directory holds the cleaned BRFSS cohorts used by the pipeline. Both
cohorts are committed here directly under a CC0-1.0 public-domain dedication
(see `data/LICENSE`), consistent with the public-domain status of the CDC
source; the upstream CDC release remains the authoritative version.

## Files

```
data/cdc_brfss_diabetes_2021.csv   # training + internal evaluation
data/cdc_brfss_diabetes_2015.csv   # cross-year external validation (§4.4)
```

Schema (identical for both cohorts): 21 features + `Diabetes_binary` target.
- BRFSS 2021: n = 236,378 records, 14.20% diabetes prevalence
- BRFSS 2015: n = 253,680 records, 13.93% diabetes prevalence

Features follow the julnazz/Teboul Kaggle convention. Actual column order
(`Diabetes_binary` first):

```
Diabetes_binary, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age,
Education, Income
```

The loader selects columns by name (`TARGET_COL = "Diabetes_binary"`), so the
physical column order is not load-bearing for the pipeline; the order above
documents the committed CSVs.

One schema caveat between waves: the `Income` feature is encoded on a 1–11
scale in BRFSS 2021 and a 1–8 scale in BRFSS 2015 (CDC survey instrument
change between waves). Tree-based classifiers handle out-of-range predictor
values without extrapolation, so the encoding shift does not break inference;
it is documented for transparency and discussed in manuscript Section 4.4.

## Reproducing the cohorts from the CDC source

The cleaned CSVs above are committed directly, so no acquisition step is needed
to run the pipeline. The route below documents how they were derived from the
original CDC files, for full transparency and independent reproduction.

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

## License and CDC BRFSS terms

The processed cohorts in this directory are dedicated to the public domain
under CC0-1.0 (`data/LICENSE`). BRFSS is a CDC public-release dataset under the
CDC public-domain notice; redistribution is permitted, and the upstream CDC
source is treated as the authoritative version. `data/PROVENANCE.md` documents
every transformation between the upstream `LLCP*.XPT` files and the cleaned
CSVs. Attribution to CDC's BRFSS as the original source is retained.
