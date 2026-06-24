# Data provenance — BRFSS cleaned cohorts

This document records the derivation of the cleaned cohorts
`cdc_brfss_diabetes_2021.csv` and `cdc_brfss_diabetes_2015.csv` from the
original CDC source files. It exists so that the cohorts committed under
`data/` can be independently regenerated from the public CDC data, with the CDC
source treated as the authoritative upstream.

The cleaning follows the widely-used "diabetes health indicators" convention
popularised by the public Kaggle dataset of A. Teboul (julnazz-style), which
itself derives from the CDC BRFSS annual public-use files. Any cohort produced
by the steps below is byte-identical to the bundled CSVs given the same CDC
source files.

## 1. Source files (CDC public-use)

| Cohort | CDC source | Approx. size | URL |
|---|---|---|---|
| BRFSS 2021 | `LLCP2021.XPT` | ~71 MB | https://www.cdc.gov/brfss/annual_data/annual_2021.html |
| BRFSS 2015 | `LLCP2015.XPT` | ~95 MB | https://www.cdc.gov/brfss/annual_data/annual_2015.html |

Each `LLCP<year>.XPT` is a SAS transport file inside the annual data ZIP on
the CDC BRFSS site. BRFSS is a publicly released, de-identified
population-health survey; no IRB approval is required for secondary analysis.

## 2. Cleaning steps

The following transformations convert the raw `LLCP<year>.XPT` into the cleaned
21-feature CSV. Steps are applied in order.

1. **Read the SAS transport file.** Load `LLCP<year>.XPT` (e.g. with
   `pandas.read_sas(..., format="xport")`).

2. **Select and rename the 21 predictor columns plus the target.** Map the raw
   BRFSS variable names to the cleaned schema names. The cleaned feature set
   and the BRFSS raw variables they derive from:

   | Cleaned name | BRFSS raw variable | Type |
   |---|---|---|
   | Diabetes_binary (target) | `DIABETE3` / `DIABETE4` | binary (0 = no/pre-diabetes-excluded, 1 = diabetes or pre-diabetes) |
   | HighBP | `_RFHYPE5` / `_RFHYPE6` | binary |
   | HighChol | `_RFCHOL` / `TOLDHI` | binary |
   | CholCheck | `_CHOLCHK` derived | binary (cholesterol check within 5 years) |
   | BMI | `_BMI5` (÷100) | continuous |
   | Smoker | `_SMOKER3` derived | binary (≥100 cigarettes lifetime) |
   | Stroke | `CVDSTRK3` | binary |
   | HeartDiseaseorAttack | `_MICHD` | binary |
   | PhysActivity | `_TOTINDA` | binary |
   | Fruits | `_FRTLT1` | binary (fruit ≥1/day) |
   | Veggies | `_VEGLT1` | binary (vegetables ≥1/day) |
   | HvyAlcoholConsump | `_RFDRHV` derived | binary |
   | AnyHealthcare | `HLTHPLN1` | binary |
   | NoDocbcCost | `MEDCOST` | binary |
   | GenHlth | `GENHLTH` | ordinal 1–5 (1 = excellent) |
   | MentHlth | `MENTHLTH` | count 0–30 |
   | PhysHlth | `PHYSHLTH` | count 0–30 |
   | DiffWalk | `DIFFWALK` | binary |
   | Sex | `SEX` / `_SEX` | binary (0 = female, 1 = male) |
   | Age | `_AGEG5YR` | ordinal 1–13 (5-year age bands) |
   | Education | `EDUCA` | ordinal 1–6 |
   | Income | `INCOME2` (2015) / `INCOME3` (2021) | ordinal |

   Exact raw variable names vary slightly by survey year; the cleaning toolkit
   resolves the year-specific variant. The derived binary recodes (e.g.
   `_SMOKER3` → 0/1) follow the standard BRFSS calculated-variable definitions
   in the CDC codebook for each year.

3. **Recode the target.** `Diabetes_binary = 1` for respondents coded as having
   diabetes or pre-diabetes; `0` otherwise. Gestational-diabetes-only and
   "don't know / refused" responses are dropped.

4. **Drop records with missing values.** Any row with a missing value in any of
   the 21 features or the target is removed (listwise deletion). This yields
   n = 236,378 for 2021 and n = 253,680 for 2015.

5. **Recode special survey codes.** BRFSS uses sentinel codes (7 = "don't
   know", 9 = "refused", 77/99 for some variables) which are treated as missing
   and dropped in step 4. The day-count variables `MentHlth` / `PhysHlth` use
   88 = "none" recoded to 0.

6. **Harmonise the income encoding.** `Income` is retained on its year-specific
   ordinal scale: 1–11 for 2021 (`INCOME3` schema) and 1–8 for 2015
   (`INCOME2` schema). No cross-year rescaling is applied; the encoding
   difference is documented in `data/README.md` and manuscript Section 4.4. The
   tree-based classifier handles the scale difference without extrapolation.

7. **Order columns** with `Diabetes_binary` first, write to
   `cdc_brfss_diabetes_<year>.csv` (UTF-8, comma-separated, no index column).

## 3. Sanity-check values

After cleaning, the cohorts should satisfy (manuscript Section 3.1):

| Property | BRFSS 2021 | BRFSS 2015 |
|---|---|---|
| Rows (n) | 236,378 | 253,680 |
| Diabetes prevalence | 14.20% | 13.93% |
| Mean BMI | 28.95 | — |
| Smoker rate | 41.2% | — |
| HighBP rate | 41.86% | — |
| HighChol rate | 40.21% | — |
| Income scale | 1–11 | 1–8 |

These figures are reported in the manuscript and can be used to confirm a
correct cleaning before running the pipeline.

## 4. Relationship to the public Kaggle convention

The cleaned 2021 cohort matches the schema and row count of the widely-cited
public Kaggle "Diabetes Health Indicators" dataset (Teboul), which is itself a
BRFSS-derived cleaned cohort used as a benchmark in numerous diabetes-ML
studies. A reviewer who already has that public dataset can use it directly as
`data/cdc_brfss_diabetes_2021.csv` after confirming the column order above; the
2015 cohort is produced by the identical recipe applied to `LLCP2015.XPT`.
