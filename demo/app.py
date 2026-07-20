"""
Counterfactual Recommendation Demo — Streamlit App
=======================================

Interactive exploration of counterfactual recommendations on diabetes
risk predictions (BRFSS 2021), using DiCE-ML on XGBoost.

This file (audit-then-act view):
    - Preset patient archetypes (sidebar dropdown)
    - Method selector (random / kdtree / genetic) — affects main panel
    - On Generate, ALL 3 methods run and cache; main panel shows the
      selected method's full result (gauges + narrative + waterfall);
      a new side-by-side section compares the 3 best CFs compactly
      to make method-choice sensitivity visible.

Run:
    streamlit run demo/app.py

Pre-requisite (one-time):
    python demo/prepare_demo_artifacts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────────
# Make the project src/ + demo/ importable
# ─────────────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
for _p in (REPO_ROOT, DEMO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.pipelines.counterfactual.dice_runner import (  # noqa: E402
    DiCEConfig,
    DiCERunner,
)
from src.pipelines.counterfactual.feature_taxonomy import (  # noqa: E402
    FEATURE_TAXONOMY,
    Mutability,
    get_actionable_features,
    get_discrete_features,
    get_features_to_vary_for_query,
)
from src.pipelines.data.loader import TARGET_COL  # noqa: E402

# Local modules (sibling files in demo/)
from narrative import cf_to_narrative  # noqa: E402
from presets import PRESETS  # noqa: E402
from visualizations import (  # noqa: E402
    feature_delta_bar,
    risk_gauge,
    risk_waterfall,
)


# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
MODELS_DIR = DEMO_DIR / "models"
MODEL_PATH = MODELS_DIR / "xgb_brfss2021.joblib"
XTRAIN_PATH = MODELS_DIR / "X_train_sample.parquet"
YTRAIN_PATH = MODELS_DIR / "y_train_sample.parquet"
XTEST_PATH = MODELS_DIR / "X_test.parquet"
META_PATH = MODELS_DIR / "metadata.json"
PROBA_PATH = MODELS_DIR / "proba_test.parquet"

# ─────────────────────────────────────────────────────────────────────
# Cohort context (top-200 high-risk reference) — added for presentation
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_proba_test():
    """All test-set predicted probabilities (for cohort context). None if absent."""
    if not PROBA_PATH.exists():
        return None
    import numpy as np
    return np.asarray(pd.read_parquet(PROBA_PATH).iloc[:, 0].to_numpy()).ravel()


@st.cache_data(show_spinner=False)
def load_cohort_stats(n_eval: int = 200):
    """Summary of the top-n_eval high-risk cohort used by the Action phase."""
    import numpy as np
    p = load_proba_test()
    if p is None:
        return None
    top = np.sort(p)[-n_eval:]
    return {
        "n_test": int(len(p)),
        "base_rate": float(p.mean()),
        "n_eval": int(n_eval),
        "cutoff": float(top.min()),
        "mean": float(top.mean()),
        "max": float(top.max()),
    }


@st.cache_data(show_spinner=False)
def load_top200_table(n_eval: int = 200):
    """Full top-n_eval high-risk cohort with feature values (for display)."""
    import numpy as np
    p = load_proba_test()
    if p is None or not XTEST_PATH.exists():
        return None
    X = pd.read_parquet(XTEST_PATH)
    idx = np.argsort(p)[-n_eval:][::-1]
    tbl = X.iloc[idx].reset_index(drop=True)
    tbl.insert(0, "P(Diabetes=1)", np.round(p[idx], 4))
    tbl.insert(0, "x_test_row", idx.astype(int))
    tbl.insert(0, "rank", range(1, len(idx) + 1))
    return tbl



# ─────────────────────────────────────────────────────────────────────
# DiCE settings
# ─────────────────────────────────────────────────────────────────────
DICE_METHODS = ["random", "kdtree", "genetic"]
DEFAULT_METHOD = "random"
N_COUNTERFACTUALS = 5
DESIRED_CLASS = 0   # 0 = non-diabetic outcome
DEFAULT_SEED = 42   # seeds numpy immediately before each DiCE call

# ─────────────────────────────────────────────────────────────────────
# Build identity
# ─────────────────────────────────────────────────────────────────────
APP_VERSION = "v0.15.0"
PAPER_DOI_URL = "https://doi.org/10.1016/j.ijmedinf.2026.106555"
REPO_URL = "https://github.com/thieuanhvan/diabetes-xai-counterfactual"
GLUCO2_URL = "https://gluco2.com"
GLUCO2_ALIAS = "demo.gluco2.com"


@st.cache_data(show_spinner=False)
def get_build_stamp() -> str:
    """Identify the deployed source, in UTC+7.

    Preferred: the git commit that produced this checkout, which is the only
    timestamp that is stable across redeploys of the same code. Streamlit
    Community Cloud clones the repo, so `git log` usually works there.

    Fallback: the mtime of this file. That is the moment the file landed on
    disk during deploy, NOT the commit time, so it is labelled differently
    to avoid claiming more precision than we have.
    """
    from datetime import datetime, timedelta, timezone
    import subprocess

    vn = timezone(timedelta(hours=7))
    try:
        sha = subprocess.run(
            ["git", "log", "-1", "--format=%h"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        iso = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        if sha.returncode == 0 and iso.returncode == 0 and iso.stdout.strip():
            when = datetime.fromisoformat(iso.stdout.strip()).astimezone(vn)
            return f"commit {sha.stdout.strip()} · {when:%Y-%m-%d %H:%M} (UTC+7)"
    except Exception:
        pass

    when = datetime.fromtimestamp(Path(__file__).stat().st_mtime, tz=vn)
    return f"build {when:%Y-%m-%d %H:%M} (UTC+7)"


# ─────────────────────────────────────────────────────────────────────
# Feature spec (mirrors src/.../feature_taxonomy.py)
# ─────────────────────────────────────────────────────────────────────
MODEL_FEATURE_ORDER = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

# Sidebar input spec. The `default` values are NOT invented: they are the
# rank-32 patient of the top-200 high-risk cohort, lifted verbatim from
# demo/models/X_test.parquet, so the app opens on a real member of the
# population the Action phase actually targets (P(Diabetes=1) = 0.786).
# The same row is also available from the preset dropdown.
FEATURE_SPEC = {
    # `choices` maps the stored BRFSS code to its codebook meaning. When present,
    # the sidebar renders a dropdown showing "code - meaning" while still storing
    # the numeric code, so presets, the model, and every table stay unchanged.
    # Sources: BRFSS 2021 codebook (_AGEG5YR, SEX, EDUCA, INCOME3, GENHLTH).
    "Age": {
        # A bare "6" reads as six years old. The code_prefix makes the dropdown
        # render "Group 6 (45-49 years old)" while still storing the integer 6.
        "label": "Age - BRFSS age group", "min": 1, "max": 13, "default": 6, "step": 1, "type": "int",
        "fmt": "Group {code} ({label})",
        "choices": {
            1: "18-24 years old", 2: "25-29 years old", 3: "30-34 years old",
            4: "35-39 years old", 5: "40-44 years old", 6: "45-49 years old",
            7: "50-54 years old", 8: "55-59 years old", 9: "60-64 years old",
            10: "65-69 years old", 11: "70-74 years old", 12: "75-79 years old",
            13: "80 years old or older",
        },
    },
    "Sex": {
        "label": "Sex", "min": 0, "max": 1, "default": 1, "step": 1, "type": "int",
        "choices": {0: "Female", 1: "Male"},
    },
    "Education": {
        "label": "Education - highest level completed", "min": 1, "max": 6, "default": 5, "step": 1, "type": "int",
        "choices": {
            1: "Never attended school or only kindergarten",
            2: "Grades 1-8 (elementary)",
            3: "Grades 9-11 (some high school)",
            4: "Grade 12 or GED (high school graduate)",
            5: "College 1-3 years (some college or technical school)",
            6: "College 4 years or more (college graduate)",
        },
    },
    "Income": {
        "label": "Income - annual household income (2021 brackets)", "min": 1, "max": 11, "default": 5, "step": 1, "type": "int",
        "choices": {
            1: "Less than $10,000", 2: "$10,000 to less than $15,000",
            3: "$15,000 to less than $20,000", 4: "$20,000 to less than $25,000",
            5: "$25,000 to less than $35,000", 6: "$35,000 to less than $50,000",
            7: "$50,000 to less than $75,000", 8: "$75,000 to less than $100,000",
            9: "$100,000 to less than $150,000", 10: "$150,000 to less than $200,000",
            11: "$200,000 or more",
        },
    },
    "BMI": {"label": "BMI - body mass index (kg/m\u00b2)", "min": 12.0, "max": 60.0, "default": 45.0, "step": 0.1, "type": "float"},
    "HighBP": {
        "label": "HighBP - told they have high blood pressure", "min": 0, "max": 1, "default": 1, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "HighChol": {
        "label": "HighChol - told they have high cholesterol", "min": 0, "max": 1, "default": 1, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "Stroke": {
        "label": "Stroke - ever had a stroke (immutable)", "min": 0, "max": 1, "default": 1, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "HeartDiseaseorAttack": {
        "label": "HeartDiseaseorAttack - CHD or MI history (immutable)", "min": 0, "max": 1, "default": 0, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "DiffWalk": {
        "label": "DiffWalk - serious difficulty walking or climbing stairs", "min": 0, "max": 1, "default": 0, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "Smoker": {
        "label": "Smoker - smoked at least 100 cigarettes in lifetime", "min": 0, "max": 1, "default": 0, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "PhysActivity": {
        "label": "PhysActivity - physical activity in the past 30 days", "min": 0, "max": 1, "default": 0, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "Fruits": {
        "label": "Fruits - eats fruit at least once a day", "min": 0, "max": 1, "default": 0, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "Veggies": {
        "label": "Veggies - eats vegetables at least once a day", "min": 0, "max": 1, "default": 0, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "HvyAlcoholConsump": {
        "label": "HvyAlcoholConsump - heavy alcohol consumption", "min": 0, "max": 1, "default": 0, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "AnyHealthcare": {
        "label": "AnyHealthcare - has any health care coverage", "min": 0, "max": 1, "default": 1, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "NoDocbcCost": {
        "label": "NoDocbcCost - could not see a doctor because of cost", "min": 0, "max": 1, "default": 0, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "CholCheck": {
        "label": "CholCheck - cholesterol checked in the past 5 years", "min": 0, "max": 1, "default": 1, "step": 1, "type": "int",
        "choices": {0: "No", 1: "Yes"},
    },
    "GenHlth": {
        "label": "GenHlth - self-rated general health (1 = best, 5 = worst)", "min": 1, "max": 5, "default": 5, "step": 1, "type": "int",
        "choices": {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"},
    },
    # The raw BRFSS column names carry no "poor", but both count days health was
    # NOT good. Spelling that out in the label prevents the exact inversion a
    # reader falls into otherwise: more days is worse, not better.
    "MentHlth": {
        "label": "MentHlth - days mental health was NOT good, in the past 30",
        "min": 0, "max": 30, "default": 0, "step": 1, "type": "int",
        "fmt": "{label}",
        "choices": {d: ("0 days" if d == 0 else f"{d} day" if d == 1 else f"{d} days")
                    for d in range(0, 31)},
    },
    "PhysHlth": {
        "label": "PhysHlth - days physical health was NOT good, in the past 30",
        "min": 0, "max": 30, "default": 0, "step": 1, "type": "int",
        "fmt": "{label}",
        "choices": {d: ("0 days" if d == 0 else f"{d} day" if d == 1 else f"{d} days")
                    for d in range(0, 31)},
    },
}

FEATURE_GROUPS = [
    ("Demographic & socioeconomic", ["Age", "Sex", "Education", "Income"]),
    ("Biometric",                   ["BMI"]),
    ("Comorbidities",               ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk"]),
    ("Behavioral lifestyle",        ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"]),
    ("Healthcare access",           ["AnyHealthcare", "NoDocbcCost", "CholCheck"]),
    ("Self-rated health",           ["GenHlth", "MentHlth", "PhysHlth"]),
]


# ─────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes CF Demo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────
# Artifact loaders
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    import joblib
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_train_sample():
    if not (XTRAIN_PATH.exists() and YTRAIN_PATH.exists()):
        return None, None
    X = pd.read_parquet(XTRAIN_PATH)
    y = pd.read_parquet(YTRAIN_PATH).iloc[:, 0]
    return X, y


@st.cache_data
def load_metadata():
    if not META_PATH.exists():
        return None
    import json
    return json.loads(META_PATH.read_text())


@st.cache_resource(show_spinner="Initializing DiCE runner…")
def get_runner(
    method: str, per_query: bool, _model, _X_train: pd.DataFrame, _y_train: pd.Series,
):
    """Build a cached DiCERunner per (method, constraint-mode). Underscore-prefixed
    args skip Streamlit hashing (mutable DataFrames).

    `per_query` MUST stay a positional cache-key argument: it selects a different
    constraint regime inside DiCERunner.generate, so a shared cache entry across
    the two modes would silently return the wrong runner.
    """
    cfg = DiCEConfig(
        method=method,
        n_counterfactuals=N_COUNTERFACTUALS,
        desired_class=DESIRED_CLASS,
        per_query=per_query,
    )
    return DiCERunner(
        model=_model,
        X_train=_X_train,
        y_train=_y_train,
        target_col=TARGET_COL,
        config=cfg,
    )


# ─────────────────────────────────────────────────────────────────────
# Core inference helpers
# ─────────────────────────────────────────────────────────────────────
def patient_to_query_df(patient: dict) -> pd.DataFrame:
    """Single-row query DataFrame in MODEL_FEATURE_ORDER."""
    row = {f: patient[f] for f in MODEL_FEATURE_ORDER}
    return pd.DataFrame([row])


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """P(Diabetes=1) for each row."""
    return model.predict_proba(X.values)[:, 1]


def round_discrete(cfs_df: pd.DataFrame) -> pd.DataFrame:
    """Round discrete features post-DiCE. Mirrors src/pipelines/main.py:114."""
    out = cfs_df.copy()
    for c in get_discrete_features():
        if c in out.columns:
            out[c] = out[c].round().astype(int)
    return out


def compute_feature_delta(query: pd.Series, cf: pd.Series) -> pd.DataFrame:
    """Return only the features that changed, side-by-side with the delta."""
    rows = []
    for f in MODEL_FEATURE_ORDER:
        if f not in cf.index:
            continue
        v0, v1 = query[f], cf[f]
        if pd.notna(v0) and pd.notna(v1) and v0 != v1:
            spec = FEATURE_TAXONOMY.get(f)
            rows.append({
                "feature": f,
                "current": v0,
                "counterfactual": v1,
                "delta": v1 - v0,
                "class": spec.mutability.value if spec else "-",
                "direction check": direction_check(f, v0, v1),
            })
    return pd.DataFrame(rows)


DIR_OK = "ok"
DIR_VIOLATION = "direction violation"
DIR_IMMUTABLE = "immutable violation"
DIR_CONDITIONAL = "conditional violation"


def direction_check(feature: str, v0: float, v1: float) -> str:
    """Label one feature change against the directional intervention taxonomy.

    Mirrors the violation accounting in
    `src/pipelines/evaluate/cf_metrics.py` (Eq. 1 numerator terms). Returned
    for display only — it does not alter CF generation.
    """
    spec = FEATURE_TAXONOMY.get(feature)
    if spec is None:
        return DIR_OK
    a, b = float(v0), float(v1)
    if a == b:
        return DIR_OK
    if spec.mutability == Mutability.IMMUTABLE:
        return DIR_IMMUTABLE
    if spec.mutability == Mutability.CONDITIONAL:
        return DIR_CONDITIONAL
    if spec.mutability == Mutability.MONOTONIC_UP and b < a:
        return DIR_VIOLATION
    if spec.mutability == Mutability.MONOTONIC_DOWN and b > a:
        return DIR_VIOLATION
    return DIR_OK


def get_method_data(result: dict | None, method: str) -> dict | None:
    """Return the per-method result block if available and ok, else None."""
    if not result or not result.get("ok"):
        return None
    method_block = result["by_method"].get(method)
    if not method_block or not method_block.get("ok"):
        return None
    return method_block


def run_one_method(
    method: str, query_df: pd.DataFrame, model, X_train, y_train,
    per_query: bool = True, seed: int = DEFAULT_SEED,
) -> dict:
    """Run one DiCE method on a single query; return a per-method result block.

    `seed` is applied to numpy immediately before generation. This makes
    DiCE-`random` reproducible across sessions. It does NOT make DiCE-`genetic`
    reproducible — dice-ml 0.12 exposes no seed for the genetic search.
    """
    try:
        runner = get_runner(method, per_query, model, X_train, y_train)
        with st.spinner(f"DiCE-{method}: generating {N_COUNTERFACTUALS} CFs…"):
            np.random.seed(int(seed))
            cf_examples = runner.generate(query_df)

        if not cf_examples or cf_examples[0] is None:
            return {"ok": False, "reason": "No CFs returned."}

        cfs_raw = cf_examples[0].final_cfs_df
        if cfs_raw is None or len(cfs_raw) == 0:
            return {"ok": False, "reason": "Empty CF set."}

        cfs_df = cfs_raw.drop(columns=[TARGET_COL]) if TARGET_COL in cfs_raw.columns else cfs_raw
        cfs_df = round_discrete(cfs_df)[MODEL_FEATURE_ORDER]
        cf_probas = predict_proba(model, cfs_df)
        best_idx = int(np.argmin(cf_probas))
        return {
            "ok": True,
            "cfs_df": cfs_df.reset_index(drop=True),
            "cf_probas": cf_probas,
            "best_idx": best_idx,
        }
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


# ─────────────────────────────────────────────────────────────────────
# Top-level artifact loads
# ─────────────────────────────────────────────────────────────────────
model = load_model()
X_train, y_train = load_train_sample()
meta = load_metadata()
artifacts_ready = (model is not None) and (X_train is not None)


# ─────────────────────────────────────────────────────────────────────
# Preset callback — populates session_state input keys + clears stale CF
# ─────────────────────────────────────────────────────────────────────
def apply_preset():
    # Read defensively. Streamlit discards the session-state entry for any
    # widget that was not rendered in the previous run, so if a run aborts
    # before the sidebar is built, this callback can fire while
    # `preset_choice` is absent and attribute access would raise
    # AttributeError. Returning quietly is correct: no preset was chosen.
    name = st.session_state.get("preset_choice")
    if name is None:
        return
    preset_values = PRESETS.get(name)
    if preset_values is not None:
        for feature, value in preset_values.items():
            spec = FEATURE_SPEC.get(feature)
            if spec is not None:
                # A selectbox rejects a value that is not identical to one of
                # its options, so 1.0 would break where 1 works. Coerce to the
                # spec's declared type before writing to session state.
                value = float(value) if spec["type"] == "float" else int(round(float(value)))
            st.session_state[f"input_{feature}"] = value
            if feature == "BMI":
                # Re-derive weight from the preset's BMI at the current height,
                # so the calculator does not keep showing the previous patient.
                _h = float(st.session_state.get("input_height_cm", DEFAULT_HEIGHT_CM))
                st.session_state["input_weight_kg"] = weight_for(value, _h)
    # Always clear stale CF result on preset change — even Custom — because
    # the user is signalling "I want a fresh look".
    st.session_state.cf_result = None


def clear_cf_result():
    """Invalidate the cached CF whenever a generation setting changes.

    Without this, flipping the constraint toggle would leave the previous
    mode's counterfactual on screen until the user clicks Generate — the
    single most misleading state this app can be in.
    """
    st.session_state.cf_result = None


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
st.sidebar.header("Patient profile")

st.sidebar.selectbox(
    "Preset patient",
    options=list(PRESETS.keys()),
    key="preset_choice",
    on_change=apply_preset,
    help=(
        "Pick a clinical archetype to populate all 21 fields below. "
        "After loading, you can still adjust individual values — the "
        "preset is just a starting point, not a lock."
    ),
)
st.sidebar.caption(
    "BRFSS 2021 · 21 features · "
    "Pick a preset to load values, then adjust freely."
)
st.sidebar.divider()

patient = {}

# Seed every input from FEATURE_SPEC before the widgets are created. Selectbox
# has no `value` parameter, so without this it would open on its first option
# (Age=1) instead of the default profile. Seeding also lets the number_inputs
# drop their `value=` argument, which silences Streamlit's warning about
# setting session state for a widget that also declares a default.
for _f, _s in FEATURE_SPEC.items():
    _k = f"input_{_f}"
    if _k not in st.session_state:
        st.session_state[_k] = (
            float(_s["default"]) if _s["type"] == "float" else int(_s["default"])
        )

# Weight and height are UI-only helpers, not model features. Height is seeded at
# a fixed value and weight is back-computed so the pair is consistent with the
# current BMI the moment the calculator is first opened.
# BRFSS records BMI. It does NOT record weight and height, so no true pair
# exists for any patient in this dataset. The calculator therefore shows ONE
# pair that reproduces the current BMI, using a reference height for the
# profile's sex. Only BMI reaches the model.
#
# Bounds are adult ranges because BRFSS surveys adults aged 18 and over.
DEFAULT_HEIGHT_CM = 170.0
HEIGHT_MIN_CM, HEIGHT_MAX_CM = 130.0, 220.0
WEIGHT_MIN_KG, WEIGHT_MAX_KG = 30.0, 250.0
REFERENCE_HEIGHT_CM = {1: 175.0, 0: 162.0}   # approximate US adult means, male / female


def reference_height_for(sex_code) -> float:
    """Reference height used only to seed the calculator, never a data value."""
    try:
        return REFERENCE_HEIGHT_CM.get(int(sex_code), DEFAULT_HEIGHT_CM)
    except (TypeError, ValueError):
        return DEFAULT_HEIGHT_CM


def weight_for(bmi: float, height_cm: float) -> float:
    """Weight reproducing `bmi` at `height_cm`, clamped to the widget bounds.

    Clamping matters: Streamlit rejects a session-state value outside a
    number_input's declared range, so an unclamped write here would crash the
    app at extreme BMI and height combinations.
    """
    raw = float(bmi) * (float(height_cm) / 100.0) ** 2
    return round(min(max(raw, WEIGHT_MIN_KG), WEIGHT_MAX_KG), 1)


if "seed_custom" not in st.session_state:
    st.session_state["seed_custom"] = DEFAULT_SEED
if "input_height_cm" not in st.session_state:
    st.session_state["input_height_cm"] = reference_height_for(
        st.session_state.get("input_Sex", 1)
    )
if "input_weight_kg" not in st.session_state:
    st.session_state["input_weight_kg"] = weight_for(
        st.session_state["input_BMI"], st.session_state["input_height_cm"]
    )

for group_title, feature_names in FEATURE_GROUPS:
    st.sidebar.markdown(f"**{group_title}**")
    for fname in feature_names:
        spec = FEATURE_SPEC[fname]

        if fname == "BMI":
            # BMI is what the model consumes and what BRFSS stores, so it stays
            # the single source of truth. Most people do not know their BMI, so
            # this offers an equivalent way in: enter weight and height, and BMI
            # is derived. Switching modes never loses the current value because
            # both branches write back to `input_BMI`.
            mode = st.sidebar.radio(
                "BMI input",
                options=["BMI directly", "Weight and height"],
                key="bmi_mode",
                horizontal=True,
                help=(
                    "BRFSS records BMI, not weight and height, so BMI is the "
                    "value actually stored and fed to the model. The second "
                    "mode simply computes it as weight / height squared."
                ),
            )
            if st.session_state.get("_prev_bmi_mode") != mode:
                # On every mode switch, re-derive weight from the BMI currently
                # in effect. Without this, a weight left over from an aborted
                # run (or reset to the widget minimum) would drive BMI to a
                # value the user never entered the moment the calculator opens.
                _h_sync = float(st.session_state.get("input_height_cm", DEFAULT_HEIGHT_CM))
                st.session_state["input_weight_kg"] = weight_for(
                    st.session_state["input_BMI"], _h_sync
                )
                st.session_state["_prev_bmi_mode"] = mode

            if mode == "BMI directly":
                patient[fname] = st.sidebar.number_input(
                    spec["label"],
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    step=float(spec["step"]),
                    key="input_BMI",
                )
                # Keep the calculator consistent even while it is hidden, so
                # switching to it never shows a weight that contradicts the BMI
                # currently in effect.
                _h_now = float(st.session_state.get("input_height_cm", DEFAULT_HEIGHT_CM))
                st.session_state["input_weight_kg"] = weight_for(
                    patient[fname], _h_now
                )
            else:
                _w = st.sidebar.number_input(
                    "Weight (kg)", min_value=WEIGHT_MIN_KG, max_value=WEIGHT_MAX_KG,
                    step=0.5, key="input_weight_kg",
                )
                _h = st.sidebar.number_input(
                    "Height (cm)", min_value=HEIGHT_MIN_CM, max_value=HEIGHT_MAX_CM,
                    step=0.5, key="input_height_cm",
                )
                _raw = float(_w) / ((float(_h) / 100.0) ** 2)
                _bmi = round(_raw, 1)
                _lo, _hi = float(spec["min"]), float(spec["max"])
                _clamped = min(max(_bmi, _lo), _hi)
                # Write back so the value survives a switch to "BMI directly".
                st.session_state["input_BMI"] = _clamped
                patient[fname] = _clamped
                if _clamped != _bmi:
                    st.sidebar.warning(
                        f"Computed BMI {_bmi} is outside the model's supported "
                        f"range [{_lo:g}, {_hi:g}] and was clamped to {_clamped:g}."
                    )
                else:
                    st.sidebar.caption(
                        f"**BMI = {_bmi:g} kg/m²** (computed, this is the value "
                        "sent to the model)"
                    )
                st.sidebar.caption(
                    "BRFSS records BMI only, so no real weight or height exists "
                    "for these patients. The pair shown is one combination that "
                    "reproduces the BMI, seeded at a reference height for the "
                    "selected sex. Change either field freely."
                )
            continue

        choices = spec.get("choices")
        if choices is not None:
            # Dropdown over the BRFSS codes. The stored value stays the numeric
            # code, so presets, the model input, and every downstream table are
            # untouched; only the on-screen text changes.
            _fmt = spec.get("fmt", "{code} ({label})")
            patient[fname] = st.sidebar.selectbox(
                spec["label"],
                options=list(choices.keys()),
                format_func=lambda v, _c=choices, _f=_fmt: _f.format(code=v, label=_c[v]),
                key=f"input_{fname}",
            )
        elif spec["type"] == "float":
            patient[fname] = st.sidebar.number_input(
                spec["label"],
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                step=float(spec["step"]),
                key=f"input_{fname}",
            )
        else:
            patient[fname] = st.sidebar.number_input(
                spec["label"],
                min_value=int(spec["min"]),
                max_value=int(spec["max"]),
                step=int(spec["step"]),
                key=f"input_{fname}",
            )

st.sidebar.divider()

st.sidebar.markdown("**Generation settings**")

enforce_constraints = st.sidebar.toggle(
    "Directional constraints (taxonomy)",
    value=True,
    key="enforce_constraints",
    on_change=clear_cf_result,
    help=(
        "ON  = per-query mode. Features already at a monotonic extreme are "
        "dropped, and permitted_range is clipped to the taxonomy-correct "
        "direction — a CF can only push a feature the way an intervention "
        "could plausibly push it.\n\n"
        "OFF = global mode. Only the 4 immutable features stay locked; "
        "every other feature may move in either direction over its full "
        "range. This is the unconstrained baseline reported in the paper "
        "(Table: global vs per-query)."
    ),
)

# The seed list is a CONTIGUOUS block 0 to 20, not a hand-picked set. That
# matters: a curated list of "good" seeds would quietly bake outcome selection
# into the UI. A contiguous block cannot have been chosen for its outcome, and
# "Custom" keeps every other integer reachable.
SEED_CHOICES: list = list(range(0, 21)) + [42, 198]
SEED_CUSTOM = "Custom…"
SEED_LABELS = {42: "42 (app default)", 198: "198 (used in the reported run)"}

seed_pick = st.sidebar.selectbox(
    "Random seed",
    options=SEED_CHOICES + [SEED_CUSTOM],
    index=SEED_CHOICES.index(DEFAULT_SEED),
    key="seed_choice",
    on_change=clear_cf_result,
    format_func=lambda v: v if isinstance(v, str) else SEED_LABELS.get(v, str(v)),
    help=(
        "Applied to numpy immediately before each DiCE call. Makes "
        "DiCE-`random` reproducible run-to-run. DiCE-`genetic` has no seed "
        "hook in dice-ml 0.12 and will still vary."
    ),
)
if seed_pick == SEED_CUSTOM:
    seed_value = st.sidebar.number_input(
        "Custom seed",
        min_value=0,
        max_value=10_000,
        step=1,
        key="seed_custom",
        on_change=clear_cf_result,
    )
else:
    seed_value = seed_pick

if enforce_constraints:
    st.sidebar.success("Constraints ON — per-query (taxonomy-enforced)")
else:
    st.sidebar.warning("Constraints OFF — global (immutable-only baseline)")

st.sidebar.caption(
    "The list is a contiguous block 0 to 20 plus two documented seeds; pick "
    "**Custom** for any other integer. The seed changes only how DiCE-`random` "
    "searches; the "
    "model itself is fully deterministic. On the profile this app opens with, "
    "sweeping seeds 0 to 200 with constraints **OFF** produced at least one "
    "direction violation in **88.1%** of runs (177 of 201), and the violated "
    "feature was `CholCheck` in 172 of them. The phenomenon is not "
    "seed-specific; try your own seed."
)

generate_clicked = st.sidebar.button(
    "Generate counterfactual",
    type="primary",
    use_container_width=True,
    disabled=not artifacts_ready,
    help=(
        f"Run ALL {len(DICE_METHODS)} DiCE methods (random / kdtree / genetic) "
        f"on the current patient in the selected constraint mode — "
        f"{N_COUNTERFACTUALS} CFs each. The method selector below picks which "
        "method's full narrative + waterfall to display in the main panel; the "
        "side-by-side section compares all three best CFs."
    )
    if artifacts_ready
    else "Artifacts missing — run prepare_demo_artifacts.py first.",
)

# Method selector for the main panel. (All 3 methods run regardless of
# this selection; this only filters which one's full result is shown.)
selected_method = st.sidebar.radio(
    "Main-panel method",
    options=DICE_METHODS,
    key="method_choice",
    index=DICE_METHODS.index(DEFAULT_METHOD),
    horizontal=True,
    format_func=lambda m: f"DiCE-{m}",
    help=(
        "Which method's full result (narrative + waterfall + raw CFs) "
        "appears in the main panel. Switching does NOT re-run DiCE — "
        "all three methods are computed once on Generate and cached."
    ),
)
st.sidebar.caption(
    "These three are the names of DiCE-ML's counterfactual **search "
    "algorithms**, not quality settings. `DiCE-random` searches by sampling "
    "perturbations — it is seeded above, so its output is reproducible, not "
    "arbitrary. `DiCE-kdtree` searches for the nearest real training patient. "
    "`DiCE-genetic` runs an evolutionary search."
)
st.sidebar.caption(
    f"Methods compared: **{' · '.join('DiCE-' + m for m in DICE_METHODS)}** · "
    f"CFs per method: **{N_COUNTERFACTUALS}** · "
    f"Target class: **{DESIRED_CLASS}** (non-diabetic)"
)


# ─────────────────────────────────────────────────────────────────────
# Main panel — header + educational expander
# ─────────────────────────────────────────────────────────────────────
st.title("Diabetes Risk — Counterfactual Recommendations")
st.caption(
    "Knowledge-Guided Counterfactual Explanations · Directional Intervention Taxonomy · BRFSS 2021"
)

st.warning(
    "**Research tool, not medical advice.** This app is the companion artifact "
    "of a peer-reviewed *methods* paper (*Int. J. Med. Inform.* 2026, 106555, "
    f"[doi:10.1016/j.ijmedinf.2026.106555]({PAPER_DOI_URL})), which "
    "studies how counterfactual explanations behave with and without a "
    "directional intervention taxonomy. It is not a clinical validation study. "
    "The app does not diagnose, screen, or recommend treatment, and its outputs "
    "are not causal claims. Do not use it to make decisions about your own "
    "health; talk to a clinician."
)

with st.expander(
    "ℹ️ About this demo — what is a counterfactual, what is locked, what is 'best of N', why three methods?",
    expanded=False,
):
    st.markdown(
        """
**What is a counterfactual (CF)?**
A CF is an alternative profile for the same patient that the model would
predict differently. If the model predicts diabetes risk = 47% for the
current profile, a CF answers: *"what minimal, plausible changes would
bring that risk below the decision threshold?"*. It is not a forecast —
it is a recommendation surface generated from the model's learned
decision boundary.

**Why are some features locked (immutable / at-extreme)?**
The recommendation should be **ethical and actionable**. Four BRFSS
features are *immutable* — `Age`, `Sex`, `Stroke`, `HeartDiseaseorAttack`
— because they cannot be changed (demographic) or represent irreversible
history. They are excluded from CF generation regardless of the patient.
Additionally, features already at a *monotonic extreme* are excluded
per-patient: if `Smoker = 0` already, the CF won't suggest "non-smoking"
(redundant); if `PhysActivity = 1` already, the CF won't suggest "be
active" (redundant). This is the **per-query constraint** in
`src/pipelines/counterfactual/feature_taxonomy.py`.

**Why 'best of N'?**
DiCE generates *N = 5* candidate CFs per method, varying which features
it perturbs. They differ in **how many** features they change and **by
how much**. The "best" CF shown is simply the one with the lowest
predicted risk among the N candidates. Other candidates appear in the
*All N counterfactuals* expander below.

**Why three methods (random / kdtree / genetic)?**
DiCE-ML implements multiple CF-search strategies. `random` samples
perturbations stochastically — seeded here via the sidebar, so it is
reproducible run-to-run. `kdtree` finds the nearest training-set
neighbour (deterministic, tied to real patients, but it returns nothing
when no neighbour satisfies the constraints). `genetic` runs an
evolutionary search optimising proximity + diversity explicitly;
**dice-ml 0.12 exposes no seed for it, so its output varies between
runs even at a fixed sidebar seed.** They typically produce different
"best" CFs for the same patient. This is the central audit-then-act
observation: a single method's recommendation is only one possible
operational answer.

**Careful: "monotonic up" and "monotonic down" describe the CODED VALUE, not health.**
A feature's class says which way an intervention may move its number, not
whether the person gets better. For `GenHlth` (1 = excellent, 5 = poor),
`MentHlth` and `PhysHlth` (days health was not good), and `HighBP` /
`HighChol` / `Smoker` (1 = yes), the healthier direction is DOWNWARD, so
those are `monotonic_down`. For `PhysActivity`, `Fruits`, `CholCheck` and
`AnyHealthcare` the healthier direction is upward. Reading "monotonic down"
as "health declines" inverts the meaning of the entire taxonomy.

**What does the 'Directional constraints' toggle do?**
It selects between the two regimes compared in the paper. **ON**
(per-query) drops features already at a monotonic extreme and clips
`permitted_range` to the taxonomy-correct direction. **OFF** (global)
locks only the 4 immutable features and lets every other feature move
either way across its full range. Turning it OFF is how you see what
the taxonomy is actually buying: the model will happily reach for
changes like *stop getting cholesterol checks* — a real risk reduction
on the model's decision surface, and clinically indefensible advice.
"""
    )


# ─────────────────────────────────────────────────────────────────────
# Artifact status row
# ─────────────────────────────────────────────────────────────────────
status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    if model is None:
        st.error("Model: not found")
    else:
        st.success("Model: loaded")
with status_col2:
    if X_train is None:
        st.error("Train sample: not found")
    else:
        st.success(f"Train sample: {len(X_train):,} rows")
with status_col3:
    if meta is None:
        st.warning("Metadata: not found")
    else:
        st.success(f"Test AUC: {meta.get('test_auc', '?')}")

if not artifacts_ready:
    st.info(
        "ℹ️ Artifacts missing. Run **`python demo/prepare_demo_artifacts.py`** "
        "from the repo root to materialize the XGBoost model and training "
        "sample. One-time setup (~30-60 s)."
    )


# ─────────────────────────────────────────────────────────────────────
# Primary action, repeated at the top of the main panel
# ─────────────────────────────────────────────────────────────────────
# The sidebar button sits below 21 input widgets, so on a laptop viewport it
# is off-screen on first load. This duplicate fires the identical action; a
# distinct key prevents a DuplicateWidgetID collision with the sidebar button.
# Held to a quarter-width column so it reads as an action, not a banner.
_btn_col = st.columns([1, 3])[0]
with _btn_col:
    top_generate_clicked = st.button(
        "Generate counterfactual",
        type="primary",
        use_container_width=True,
        disabled=not artifacts_ready,
        key="generate_top",
        help="Identical to the Generate button at the bottom of the sidebar.",
    )
generate_clicked = bool(generate_clicked) or bool(top_generate_clicked)


# ─────────────────────────────────────────────────────────────────────
# CF generation — runs ALL 3 methods on click
# ─────────────────────────────────────────────────────────────────────
if "cf_result" not in st.session_state:
    st.session_state.cf_result = None


if generate_clicked and artifacts_ready:
    query_df = patient_to_query_df(patient)
    baseline = float(predict_proba(model, query_df)[0])
    # Mirror the two branches inside DiCERunner.generate so the on-screen
    # "N of 21 varied" caption matches what DiCE was actually given.
    if enforce_constraints:
        ftv = get_features_to_vary_for_query(query_df.iloc[0])
    else:
        ftv = get_actionable_features()

    if not ftv:
        st.session_state.cf_result = {
            "ok": False,
            "reason": (
                "Patient is already at all monotonic extremes — no actionable "
                "feature to vary. Try a less-healthy preset or adjust the "
                "sidebar (e.g. higher BMI, PhysActivity=0)."
            ),
            "baseline": baseline,
            "constrained": bool(enforce_constraints),
            "seed": int(seed_value),
        }
    else:
        by_method = {
            method: run_one_method(
                method, query_df, model, X_train, y_train,
                per_query=enforce_constraints, seed=int(seed_value),
            )
            for method in DICE_METHODS
        }
        st.session_state.cf_result = {
            "ok": True,
            "baseline": baseline,
            "by_method": by_method,
            "query": query_df.iloc[0],
            "n_features_varied": len(ftv),
            "constrained": bool(enforce_constraints),
            "seed": int(seed_value),
        }


# ─────────────────────────────────────────────────────────────────────
# Main panel — two-column (Baseline | CF for selected method)
# ─────────────────────────────────────────────────────────────────────
result = st.session_state.cf_result
method_data = get_method_data(result, selected_method)

if result is not None:
    if result.get("constrained", True):
        st.info(
            f"Displayed result: **directional constraints ON** (per-query) · "
            f"seed {result.get('seed', DEFAULT_SEED)}"
        )
    else:
        st.warning(
            f"Displayed result: **directional constraints OFF** (global baseline) · "
            f"seed {result.get('seed', DEFAULT_SEED)}"
        )

def risk_context_note(p_risk: float, n_eval: int = 200) -> None:
    """Place one score in the cohort, in plain language first.

    Two audiences read this line. Someone arriving from the paper wants rank
    and cohort membership; someone who just typed in their own height and
    weight wants to know what the number means. So the headline is a
    percentile against the dataset, and the research-facing detail sits underneath
    in a caption.

    Everything here describes the position of a MODEL SCORE within a dataset.
    It deliberately avoids saying the person is or is not at risk: the label
    this model was trained on is "has been diagnosed by a doctor", so a high
    score is not a diagnosis and a low score is not a clearance.

    Keyed to RANK, not to 0.5. At a test-set prevalence of 0.142 the 0.5
    decision threshold has very low sensitivity, so the paper selects the
    Action-phase population by rank instead.
    """
    cohort = load_cohort_stats(n_eval)
    p_all = load_proba_test()
    if cohort is None or p_all is None:
        return
    n_test = cohort["n_test"]
    rank = int((p_all > p_risk).sum()) + 1
    pct_below = float((p_all < p_risk).mean()) * 100.0
    base = cohort["base_rate"]
    ratio = p_risk / base if base else float("nan")
    ratio_txt = f"{ratio:.2f}x" if ratio < 1 else f"{ratio:.1f}x"
    cutoff = cohort["cutoff"]

    if p_risk >= cutoff:
        st.error(
            f"**This profile scores higher than {pct_below:.1f}% of the "
            f"{n_test:,} people in the study data.** That is a score from a "
            "research model, not a diagnosis."
        )
    elif p_risk >= 2 * base:
        st.warning(
            f"**This profile scores higher than {pct_below:.0f}% of the "
            f"{n_test:,} people in the study data.**"
        )
    else:
        st.info(
            f"**This profile scores lower than most of the {n_test:,} people "
            "in the study data.**"
        )

    in_cohort = ("inside" if p_risk >= cutoff else "outside")
    st.caption(
        f"For reference: rank {rank:,} of {n_test:,} · {ratio_txt} the dataset "
        f"average of {base:.3f} · {in_cohort} the top-{n_eval} group this study "
        f"analyses (score {cutoff:.3f} or above)."
    )


col_baseline, col_cf = st.columns(2, gap="large")

with col_baseline:
    st.subheader("Baseline risk")
    if result is None:
        if artifacts_ready:
            live_baseline = float(predict_proba(model, patient_to_query_df(patient))[0])
            st.metric(
                label="Predicted P(Diabetes=1)",
                value=f"{live_baseline:.3f}",
                help=(
                    "XGBoost-predicted probability that the patient has "
                    "diabetes, given the current sidebar inputs. Computed on "
                    "every rerun (live). Decision threshold = 0.5; "
                    "population base rate in BRFSS 2021 test set ≈ 0.142."
                ),
            )
            st.plotly_chart(
                risk_gauge(live_baseline, title="P(Diabetes=1) — live"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            risk_context_note(live_baseline)
            st.caption("Updates live with sidebar inputs. Click **Generate counterfactual** for the CF.")
        else:
            st.metric(label="Predicted P(Diabetes=1)", value="—")
    else:
        st.metric(
            label="Predicted P(Diabetes=1)",
            value=f"{result['baseline']:.3f}",
            help=(
                "XGBoost-predicted probability that the patient has diabetes, "
                "frozen at the moment 'Generate counterfactual' was clicked."
            ),
        )
        st.plotly_chart(
            risk_gauge(result["baseline"], title="Baseline P(Diabetes=1)"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        risk_context_note(result["baseline"])

with col_cf:
    st.subheader(f"Counterfactual recommendation (DiCE-{selected_method})")
    if result is None:
        st.info("Click **Generate counterfactual** above (or at the bottom of the sidebar) to find an actionable profile change.")
        st.metric(label="CF risk", value="—")
    elif not result["ok"]:
        st.warning(result["reason"])
        st.metric(label="CF risk", value="—")
    elif method_data is None:
        # The currently-selected method failed for this patient, but others
        # may have succeeded — surface this clearly.
        method_block = result["by_method"].get(selected_method, {})
        st.warning(
            f"DiCE-{selected_method} could not generate CFs for this patient. "
            f"Reason: {method_block.get('reason', 'unknown')}. "
            "Check the Compare methods section below — another method may "
            "have succeeded."
        )
        st.metric(label="CF risk", value="—")
    else:
        best_proba = float(method_data["cf_probas"][method_data["best_idx"]])
        delta_proba = best_proba - result["baseline"]
        st.metric(
            label=f"CF risk (best of {len(method_data['cf_probas'])})",
            value=f"{best_proba:.3f}",
            delta=f"{delta_proba:+.3f}",
            delta_color="inverse",
            help=(
                f"Lowest predicted P(Diabetes=1) among the "
                f"{len(method_data['cf_probas'])} candidates DiCE-{selected_method} "
                "returned. Green delta = risk reduced (good)."
            ),
        )
        st.plotly_chart(
            risk_gauge(
                best_proba,
                title=f"Best-CF P(Diabetes=1) ({selected_method})",
                baseline=result["baseline"],
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        if result.get("constrained", True):
            st.caption(
                f"DiCE varied **{result['n_features_varied']}** of 21 features — "
                "**constraints ON** (immutable + at-extreme excluded, direction "
                "clipped per `feature_taxonomy.py`)."
            )
        else:
            st.caption(
                f"DiCE varied **{result['n_features_varied']}** of 21 features — "
                "**constraints OFF** (only the 4 immutable features locked; "
                "direction unrestricted)."
            )


# ─────────────────────────────────────────────────────────────────────
# Feature delta table + bar chart + narrative + waterfall
# ─────────────────────────────────────────────────────────────────────
if method_data is not None:
    st.divider()
    st.subheader(f"What changed (DiCE-{selected_method} best CF)")

    best_cf = method_data["cfs_df"].iloc[method_data["best_idx"]]
    delta_df = compute_feature_delta(result["query"], best_cf)

    if delta_df.empty:
        st.info("No feature changes detected. The model already predicts the desired class for this profile after rounding.")
    else:
        violations = delta_df[delta_df["direction check"] != DIR_OK]
        if len(violations):
            offenders = ", ".join(
                f"`{r['feature']}` {float(r['current']):g}→"
                f"{float(r['counterfactual']):g} ({r['direction check']})"
                for _, r in violations.iterrows()
            )
            st.error(
                f"**{len(violations)} of {len(delta_df)} changes violate the "
                f"intervention taxonomy:** {offenders}"
            )
        else:
            st.success(
                f"All {len(delta_df)} changes are taxonomy-consistent — every "
                "feature moves in a direction an intervention could produce."
            )

        def _flag_row(row):
            bad = row["direction check"] != DIR_OK
            return ["background-color: #ffe0e0" if bad else "" for _ in row]

        st.dataframe(
            delta_df.style
            .apply(_flag_row, axis=1)
            .format({
                "current": "{:.3g}",
                "counterfactual": "{:.3g}",
                "delta": "{:+.3g}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Showing only the {len(delta_df)} features that differ between the patient and the best CF. "
            "Discrete features rounded to int (mirrors `src/pipelines/main.py:114`). "
            "`direction check` compares each change against the feature's taxonomy "
            "class — it is a post-hoc label, not a filter."
        )

        bar_fig = feature_delta_bar(delta_df)
        if bar_fig is not None:
            st.plotly_chart(
                bar_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # Narrative
    st.divider()
    st.subheader("Recommendation narrative")
    best_proba_for_narr = float(method_data["cf_probas"][method_data["best_idx"]])
    narrative_md = cf_to_narrative(
        query=result["query"],
        cf=best_cf,
        baseline_risk=result["baseline"],
        cf_risk=best_proba_for_narr,
    )
    st.markdown(narrative_md)

    # Waterfall (expander)
    with st.expander(
        "📊 Cumulative risk reduction (per-feature attribution)",
        expanded=False,
    ):
        def _predict_one(row_df: pd.DataFrame) -> float:
            return float(predict_proba(model, row_df.astype(float))[0])

        wf_fig = risk_waterfall(
            predict_fn=_predict_one,
            query=result["query"],
            best_cf=best_cf,
            feature_order=MODEL_FEATURE_ORDER,
        )
        if wf_fig is not None:
            st.plotly_chart(
                wf_fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "Each bar shows the marginal change in P(Diabetes=1) from applying one "
                "feature change on top of the previous step. Order is `MODEL_FEATURE_ORDER` "
                "(deterministic). The **total** reduction is order-invariant; the per-step "
                "deltas depend on order because XGBoost is non-additive — so read these "
                "as *one valid decomposition*, not the unique attribution."
            )


# ─────────────────────────────────────────────────────────────────────
# Side-by-side: Compare methods (headline section)
# ─────────────────────────────────────────────────────────────────────
if result is not None and result.get("ok"):
    st.divider()
    st.subheader("Compare methods — same patient, three DiCE strategies")
    st.caption(
        "Side-by-side best CF per method. The **audit-then-act** observation: "
        "three search strategies on the same patient can land on different "
        "recommendations. This mirrors a cross-method audit of "
        "explanation-method agreement."
    )

    cols = st.columns(len(DICE_METHODS))
    for col, method in zip(cols, DICE_METHODS):
        with col:
            block = result["by_method"].get(method, {})
            st.markdown(f"##### DiCE-`{method}`")
            if not block.get("ok"):
                st.warning(block.get("reason", "Not available"))
                continue

            best_p = float(block["cf_probas"][block["best_idx"]])
            delta_p = best_p - result["baseline"]
            best_cf_row = block["cfs_df"].iloc[block["best_idx"]]
            mini_delta = compute_feature_delta(result["query"], best_cf_row)

            st.metric(
                label="Best CF risk",
                value=f"{best_p:.3f}",
                delta=f"{delta_p:+.3f}",
                delta_color="inverse",
            )
            st.plotly_chart(
                risk_gauge(
                    best_p,
                    title=f"P(Diabetes=1)",
                    baseline=result["baseline"],
                    height=180,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            n_changed = len(mini_delta)
            if n_changed == 0:
                st.caption("No features changed (rare — model already at desired class).")
            else:
                feature_list = ", ".join(mini_delta["feature"].tolist())
                st.caption(
                    f"**{n_changed} feature{'s' if n_changed != 1 else ''} changed:** {feature_list}"
                )


# ─────────────────────────────────────────────────────────────────────
# Raw CF expanders (for the active method)
# ─────────────────────────────────────────────────────────────────────
if method_data is not None:
    st.divider()
    with st.expander(
        f"All {len(method_data['cfs_df'])} counterfactuals from DiCE-{selected_method} (raw)",
        expanded=False,
    ):
        cfs_display = method_data["cfs_df"].copy()
        cfs_display.insert(0, "P(Diabetes=1)", method_data["cf_probas"].round(3))
        st.dataframe(cfs_display, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Export run report — plain text, reproducible, paste-friendly
# ─────────────────────────────────────────────────────────────────────
def _env_versions() -> dict:
    """Package versions that actually affect CF output."""
    import platform
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    for pkg in ("streamlit", "dice-ml", "xgboost", "scikit-learn",
                "pandas", "numpy"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def build_run_report(result: dict, meta: dict | None) -> str:
    """Render the whole run as plain text: inputs, settings, every method.

    Everything reported is read back from `result`, i.e. the state frozen when
    Generate was clicked — not from the live sidebar. If the sidebar was edited
    after generating, the report still describes the run that produced the
    numbers on screen.
    """
    from datetime import datetime, timedelta, timezone

    vn_now = datetime.now(timezone(timedelta(hours=7)))
    L: list[str] = []
    L.append("# Counterfactual Run Report")
    L.append("")
    L.append(f"- Generated: {vn_now:%Y-%m-%d %H:%M} (UTC+7)")
    L.append(f"- App: diabetes-xai-counterfactual demo {APP_VERSION} · {get_build_stamp()}")
    L.append("- Paper: Int. J. Med. Inform. (2026), doi:10.1016/j.ijmedinf.2026.106555")
    if meta:
        L.append(
            f"- Model: XGBoost · test AUC {meta.get('test_auc', '?')} · "
            f"n_train {meta.get('n_train', '?'):,} · n_test {meta.get('n_test', '?'):,} · "
            f"test prevalence {meta.get('prevalence_test', '?')}"
        )
    L.append("")

    L.append("## Generation Settings")
    L.append("")
    mode = "ON (per-query, taxonomy-enforced)" if result.get("constrained", True) \
        else "OFF (global, immutable-only baseline)"
    L.append(f"- Directional constraints: **{mode}**")
    L.append(f"- Random seed: {result.get('seed', DEFAULT_SEED)}")
    L.append(f"- CFs requested per method: {N_COUNTERFACTUALS}")
    L.append(f"- Desired class: {DESIRED_CLASS} (non-diabetic)")
    if "n_features_varied" in result:
        L.append(f"- Features DiCE was allowed to vary: {result['n_features_varied']} of 21")
    L.append("")
    L.append("Reminder: DiCE-`random` is reproducible at a fixed seed. "
             "DiCE-`genetic` is not — dice-ml 0.12 exposes no seed for it.")
    L.append("")

    L.append("## Patient Profile (frozen at Generate)")
    L.append("")
    q = result["query"]
    L.append("| feature | value | taxonomy class |")
    L.append("|---|---|---|")
    for f in MODEL_FEATURE_ORDER:
        spec = FEATURE_TAXONOMY.get(f)
        cls = spec.mutability.value if spec else "-"
        L.append(f"| {f} | {float(q[f]):g} | {cls} |")
    L.append("")
    L.append(f"- Baseline P(Diabetes=1): **{result['baseline']:.4f}**")
    _p_all = load_proba_test()
    if _p_all is not None:
        _rank = int((_p_all > result["baseline"]).sum()) + 1
        L.append(f"- Rank in test set: {_rank:,} of {len(_p_all):,}")
    L.append("")

    L.append("## Results By Method")
    L.append("")
    for method in DICE_METHODS:
        block = result["by_method"].get(method, {})
        L.append(f"### DiCE-{method}")
        L.append("")
        if not block.get("ok"):
            L.append(f"FAILED: {block.get('reason', 'unknown')}")
            L.append("")
            continue
        probas = block["cf_probas"]
        bi = block["best_idx"]
        L.append(f"- All {len(probas)} CF risks: "
                 + ", ".join(f"{float(x):.4f}" for x in probas))
        L.append(f"- Best CF risk: **{float(probas[bi]):.4f}** "
                 f"(delta {float(probas[bi]) - result['baseline']:+.4f})")
        d = compute_feature_delta(q, block["cfs_df"].iloc[bi])
        if d.empty:
            L.append("- Changes: none")
            L.append("")
            continue
        n_bad = int((d["direction check"] != DIR_OK).sum())
        L.append(f"- Changes: {len(d)} · taxonomy violations: {n_bad}")
        L.append("")
        L.append("| feature | current | counterfactual | delta | class | direction check |")
        L.append("|---|---|---|---|---|---|")
        for _, r in d.iterrows():
            L.append(
                f"| {r['feature']} | {float(r['current']):g} | "
                f"{float(r['counterfactual']):g} | {float(r['delta']):+g} | "
                f"{r['class']} | {r['direction check']} |"
            )
        L.append("")

    L.append("## Environment")
    L.append("")
    for k, v in _env_versions().items():
        L.append(f"- {k}: {v}")
    L.append("")
    return "\n".join(L)


if result is not None and result.get("ok"):
    st.divider()
    st.subheader("Export run report")
    st.caption(
        "Plain-text summary of this exact run: inputs, constraint mode, seed, "
        "and every method's counterfactuals with taxonomy checks. Copy it with "
        "the button in the top-right of the box, or download it as a file."
    )
    report_text = build_run_report(result, meta)
    st.code(report_text, language="markdown")

    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    _stamp = _dt.now(_tz(_td(hours=7))).strftime("%Y%m%d_%H%M")
    _mode_tag = "constrained" if result.get("constrained", True) else "unconstrained"
    st.download_button(
        "Download report (.md)",
        data=report_text,
        file_name=f"cf_run_report_{_mode_tag}_seed{result.get('seed', DEFAULT_SEED)}_{_stamp}.md",
        mime="text/markdown",
    )


# ─────────────────────────────────────────────────────────────────────
# Patient input echo
# ─────────────────────────────────────────────────────────────────────
with st.expander("📊 Cohort context — top-200 high-risk (reference for presentation)", expanded=False):
    _cohort = load_cohort_stats()
    _proba_all = load_proba_test()
    if _cohort is None or _proba_all is None:
        st.info("proba_test.parquet not found — run `python demo/prepare_demo_artifacts.py`.")
    else:
        _r1 = st.columns(3)
        _r1[0].metric("Test patients (N)", f"{_cohort['n_test']:,}")
        _r1[1].metric("Base rate", f"{_cohort['base_rate']:.3f}")
        _r1[2].metric(f"Top-{_cohort['n_eval']} cutoff", f"{_cohort['cutoff']:.3f}")
        _r2 = st.columns(3)
        _r2[0].metric(f"Top-{_cohort['n_eval']} mean risk", f"{_cohort['mean']:.3f}")
        _r2[1].metric("Top-1 (highest) risk", f"{_cohort['max']:.3f}")
        if artifacts_ready:
            _cur = float(predict_proba(model, patient_to_query_df(patient))[0])
            _rank = int((_proba_all > _cur).sum()) + 1
            _r2[2].metric("This patient rank", f"{_rank:,} / {_cohort['n_test']:,}")
            if _cur >= _cohort["cutoff"]:
                st.success(
                    f"Current profile (risk {_cur:.3f}) IS in the top-{_cohort['n_eval']} "
                    "cohort — the population the Action phase generates CFs for."
                )
            else:
                st.warning(
                    f"Current profile (risk {_cur:.3f}) is NOT in the top-{_cohort['n_eval']} "
                    f"(cutoff {_cohort['cutoff']:.3f}). In the study, CFs are generated only "
                    f"for the top-{_cohort['n_eval']}; the demo allows any profile for illustration."
                )
        st.caption(
            f"The Action phase generates CFs for the top-{_cohort['n_eval']} highest-risk "
            "patients (rank-based selection, not a 0.5 cutoff). n=200 is a pragmatic "
            "compute choice, stable across 5 seeds (CV 0.65%) — not an XGBoost convention. "
            "Numbers computed live from proba_test.parquet."
        )


with st.expander("📋 Top-200 high-risk cohort — full list (the Action-phase CF population)", expanded=False):
    _tbl = load_top200_table()
    if _tbl is None:
        st.info("Artifacts missing — run `python demo/prepare_demo_artifacts.py`.")
    else:
        # Locate the sidebar's default profile inside the cohort by matching all
        # 21 features, rather than hard-coding a rank. If the defaults are ever
        # edited, the marker follows them instead of silently pointing at the
        # wrong patient.
        _tbl = _tbl.copy()
        _match = np.ones(len(_tbl), dtype=bool)
        for _f in MODEL_FEATURE_ORDER:
            _match &= np.isclose(
                _tbl[_f].to_numpy(dtype=float),
                float(FEATURE_SPEC[_f]["default"]),
            )
        _tbl.insert(3, "note", np.where(_match, "◀ default profile", ""))

        def _hl_default(row):
            hit = bool(row["note"])
            return ["background-color: #ffe0e0; font-weight: 600" if hit else ""
                    for _ in row]

        st.caption(
            "The 200 highest-risk patients in the BRFSS 2021 test set, ranked by predicted "
            "probability. This is the exact cohort the Action phase generates counterfactuals "
            "for (top-200 by rank, not a 0.5 cutoff)."
        )
        if _match.any():
            _rk = int(_tbl.loc[_match, "rank"].iloc[0])
            st.caption(
                f"The row highlighted in red (**rank {_rk}**) is the profile the "
                "sidebar loads by default, so the app opens on a real member of "
                "this cohort rather than an invented one."
            )
        else:
            st.caption(
                "The sidebar's current default profile is not a member of this "
                "cohort, so no row is highlighted."
            )
        st.dataframe(
            _tbl.style.apply(_hl_default, axis=1),
            use_container_width=True,
            height=430,
            hide_index=True,
        )


with st.expander("Patient input (raw, in model feature order)", expanded=False):
    ordered = {f: patient[f] for f in MODEL_FEATURE_ORDER}
    st.dataframe(
        pd.DataFrame([ordered]).T.rename(columns={0: "value"}),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Diabetes XAI Counterfactual Demo · {APP_VERSION} · {get_build_stamp()} · "
    f"Companion to [Int. J. Med. Inform. (2026), "
    f"doi:10.1016/j.ijmedinf.2026.106555]({PAPER_DOI_URL}) · "
    f"[github.com/thieuanhvan/diabetes-xai-counterfactual]({REPO_URL})"
)
st.caption(
    f"Short link: **{GLUCO2_ALIAS}** · part of "
    f"[Gluco2]({GLUCO2_URL}), a diabetes data and research initiative"
)