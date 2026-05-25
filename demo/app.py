"""
P4 Counterfactual Demo — Streamlit App
=======================================

Interactive exploration of counterfactual recommendations on diabetes
risk predictions (BRFSS 2021), using DiCE-ML on XGBoost.

Phase 5 — Audit-then-act (this file):
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
# Make P4's src/ + demo/ importable
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


# ─────────────────────────────────────────────────────────────────────
# DiCE settings
# ─────────────────────────────────────────────────────────────────────
DICE_METHODS = ["random", "kdtree", "genetic"]
DEFAULT_METHOD = "random"
N_COUNTERFACTUALS = 5
DESIRED_CLASS = 0   # 0 = non-diabetic outcome


# ─────────────────────────────────────────────────────────────────────
# Feature spec (mirrors src/.../feature_taxonomy.py)
# ─────────────────────────────────────────────────────────────────────
MODEL_FEATURE_ORDER = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

FEATURE_SPEC = {
    "Age":             {"label": "Age category (1=18-24 … 13=80+)",         "min": 1,    "max": 13,   "default": 10,   "step": 1,   "type": "int"},
    "Sex":             {"label": "Sex (0=female, 1=male)",                  "min": 0,    "max": 1,    "default": 0,    "step": 1,   "type": "int"},
    "Education":       {"label": "Education (1=none … 6=college grad)",     "min": 1,    "max": 6,    "default": 4,    "step": 1,   "type": "int"},
    "Income":          {"label": "Income 2021 (1=<\\$10K … 11=>=\\$200K)",   "min": 1,    "max": 11,   "default": 6,    "step": 1,   "type": "int"},
    "BMI":             {"label": "BMI (kg/m²)",                             "min": 12.0, "max": 60.0, "default": 30.0, "step": 0.1, "type": "float"},
    "HighBP":          {"label": "High blood pressure (0/1)",               "min": 0,    "max": 1,    "default": 1,    "step": 1,   "type": "int"},
    "HighChol":        {"label": "High cholesterol (0/1)",                  "min": 0,    "max": 1,    "default": 1,    "step": 1,   "type": "int"},
    "Stroke":          {"label": "Ever had stroke (0/1, immutable)",        "min": 0,    "max": 1,    "default": 0,    "step": 1,   "type": "int"},
    "HeartDiseaseorAttack": {"label": "CHD or MI history (0/1, immutable)", "min": 0,    "max": 1,    "default": 0,    "step": 1,   "type": "int"},
    "DiffWalk":        {"label": "Serious difficulty walking (0/1)",        "min": 0,    "max": 1,    "default": 0,    "step": 1,   "type": "int"},
    "Smoker":          {"label": "Smoked ≥100 cigarettes lifetime (0/1)",   "min": 0,    "max": 1,    "default": 0,    "step": 1,   "type": "int"},
    "PhysActivity":    {"label": "Physical activity past 30 days (0/1)",    "min": 0,    "max": 1,    "default": 0,    "step": 1,   "type": "int"},
    "Fruits":          {"label": "Fruit ≥1/day (0/1)",                      "min": 0,    "max": 1,    "default": 0,    "step": 1,   "type": "int"},
    "Veggies":         {"label": "Vegetables ≥1/day (0/1)",                 "min": 0,    "max": 1,    "default": 1,    "step": 1,   "type": "int"},
    "HvyAlcoholConsump": {"label": "Heavy alcohol consumption (0/1)",       "min": 0,    "max": 1,    "default": 0,    "step": 1,   "type": "int"},
    "AnyHealthcare":   {"label": "Has healthcare coverage (0/1)",           "min": 0,    "max": 1,    "default": 1,    "step": 1,   "type": "int"},
    "NoDocbcCost":     {"label": "Couldn't see doctor due to cost (0/1)",   "min": 0,    "max": 1,    "default": 0,    "step": 1,   "type": "int"},
    "CholCheck":       {"label": "Cholesterol check past 5 years (0/1)",    "min": 0,    "max": 1,    "default": 1,    "step": 1,   "type": "int"},
    "GenHlth":         {"label": "General health (1=Excellent … 5=Poor)",   "min": 1,    "max": 5,    "default": 4,    "step": 1,   "type": "int"},
    "MentHlth":        {"label": "Bad mental health days past 30",          "min": 0,    "max": 30,   "default": 2,    "step": 1,   "type": "int"},
    "PhysHlth":        {"label": "Bad physical health days past 30",        "min": 0,    "max": 30,   "default": 5,    "step": 1,   "type": "int"},
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
def get_runner(method: str, _model, _X_train: pd.DataFrame, _y_train: pd.Series):
    """Build a cached DiCERunner per method. Underscore-prefixed args
    skip Streamlit hashing (mutable DataFrames)."""
    cfg = DiCEConfig(
        method=method,
        n_counterfactuals=N_COUNTERFACTUALS,
        desired_class=DESIRED_CLASS,
        per_query=True,
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
            rows.append({
                "feature": f,
                "current": v0,
                "counterfactual": v1,
                "delta": v1 - v0,
            })
    return pd.DataFrame(rows)


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
) -> dict:
    """Run one DiCE method on a single query; return a per-method result block."""
    try:
        runner = get_runner(method, model, X_train, y_train)
        with st.spinner(f"DiCE-{method}: generating {N_COUNTERFACTUALS} CFs…"):
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
    name = st.session_state.preset_choice
    preset_values = PRESETS.get(name)
    if preset_values is not None:
        for feature, value in preset_values.items():
            st.session_state[f"input_{feature}"] = value
    # Always clear stale CF result on preset change — even Custom — because
    # the user is signalling "I want a fresh look".
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
for group_title, feature_names in FEATURE_GROUPS:
    st.sidebar.markdown(f"**{group_title}**")
    for fname in feature_names:
        spec = FEATURE_SPEC[fname]
        if spec["type"] == "float":
            patient[fname] = st.sidebar.number_input(
                spec["label"],
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(spec["default"]),
                step=float(spec["step"]),
                key=f"input_{fname}",
            )
        else:
            patient[fname] = st.sidebar.number_input(
                spec["label"],
                min_value=int(spec["min"]),
                max_value=int(spec["max"]),
                value=int(spec["default"]),
                step=int(spec["step"]),
                key=f"input_{fname}",
            )

st.sidebar.divider()

generate_clicked = st.sidebar.button(
    "Generate counterfactual",
    type="primary",
    use_container_width=True,
    disabled=not artifacts_ready,
    help=(
        f"Run ALL {len(DICE_METHODS)} DiCE methods (random / kdtree / genetic) "
        f"in per-query mode on the current patient — {N_COUNTERFACTUALS} CFs "
        "each. The method selector below picks which method's full "
        "narrative + waterfall to display in the main panel; the "
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
    help=(
        "Which method's full result (narrative + waterfall + raw CFs) "
        "appears in the main panel. Switching does NOT re-run DiCE — "
        "all three methods are computed once on Generate and cached."
    ),
)
st.sidebar.caption(
    f"Methods compared: **{' · '.join(DICE_METHODS)}** · "
    f"CFs per method: **{N_COUNTERFACTUALS}** · "
    f"Target class: **{DESIRED_CLASS}** (non-diabetic)"
)


# ─────────────────────────────────────────────────────────────────────
# Main panel — header + educational expander
# ─────────────────────────────────────────────────────────────────────
st.title("Diabetes Risk — Counterfactual Recommendations")
st.caption(
    "P4 · Knowledge-Guided Constraint Compiler for Actionable CFs · BRFSS 2021"
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
perturbations stochastically (fast, diverse but non-deterministic).
`kdtree` finds the nearest training-set neighbor (deterministic, tied
to real patients). `genetic` runs an evolutionary search over the
feature space (deterministic with seed, optimizes proximity + diversity
explicitly). **They typically produce different "best" CFs for the same
patient.** This is the central audit-then-act observation: a single
method's recommendation is only one possible operational answer.
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
# CF generation — runs ALL 3 methods on click
# ─────────────────────────────────────────────────────────────────────
if "cf_result" not in st.session_state:
    st.session_state.cf_result = None


if generate_clicked and artifacts_ready:
    query_df = patient_to_query_df(patient)
    baseline = float(predict_proba(model, query_df)[0])
    ftv = get_features_to_vary_for_query(query_df.iloc[0])

    if not ftv:
        st.session_state.cf_result = {
            "ok": False,
            "reason": (
                "Patient is already at all monotonic extremes — no actionable "
                "feature to vary. Try a less-healthy preset or adjust the "
                "sidebar (e.g. higher BMI, PhysActivity=0)."
            ),
            "baseline": baseline,
        }
    else:
        by_method = {
            method: run_one_method(method, query_df, model, X_train, y_train)
            for method in DICE_METHODS
        }
        st.session_state.cf_result = {
            "ok": True,
            "baseline": baseline,
            "by_method": by_method,
            "query": query_df.iloc[0],
            "n_features_varied": len(ftv),
        }


# ─────────────────────────────────────────────────────────────────────
# Main panel — two-column (Baseline | CF for selected method)
# ─────────────────────────────────────────────────────────────────────
col_baseline, col_cf = st.columns(2, gap="large")
result = st.session_state.cf_result
method_data = get_method_data(result, selected_method)

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
        prevalence = meta.get("prevalence_test") if meta else None
        if prevalence is not None:
            ratio = result["baseline"] / prevalence
            st.caption(
                f"Test-set base rate = {prevalence:.3f} → patient is **{ratio:.1f}×** the base rate."
            )

with col_cf:
    st.subheader(f"Counterfactual recommendation ({selected_method})")
    if result is None:
        st.info("Click **Generate counterfactual** in the sidebar to find an actionable profile change.")
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
        st.caption(
            f"DiCE varied **{result['n_features_varied']}** of 21 features "
            f"(immutable + at-extreme excluded per `feature_taxonomy.py`)."
        )


# ─────────────────────────────────────────────────────────────────────
# Feature delta table + bar chart + narrative + waterfall
# ─────────────────────────────────────────────────────────────────────
if method_data is not None:
    st.divider()
    st.subheader(f"What changed ({selected_method} best CF)")

    best_cf = method_data["cfs_df"].iloc[method_data["best_idx"]]
    delta_df = compute_feature_delta(result["query"], best_cf)

    if delta_df.empty:
        st.info("No feature changes detected. The model already predicts the desired class for this profile after rounding.")
    else:
        st.dataframe(
            delta_df.style.format({
                "current": "{:.3g}",
                "counterfactual": "{:.3g}",
                "delta": "{:+.3g}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Showing only the {len(delta_df)} features that differ between the patient and the best CF. "
            "Discrete features rounded to int (mirrors `src/pipelines/main.py:114`)."
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
# Side-by-side: Compare methods (the Phase 5 headline section)
# ─────────────────────────────────────────────────────────────────────
if result is not None and result.get("ok"):
    st.divider()
    st.subheader("Compare methods — same patient, three DiCE strategies")
    st.caption(
        "Side-by-side best CF per method. The **audit-then-act** observation: "
        "three search strategies on the same patient can land on different "
        "recommendations. This is the CF-family analogue of P2's "
        "explanation-method agreement audit."
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
# Patient input echo
# ─────────────────────────────────────────────────────────────────────
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
    "Diabetes XAI Counterfactual Demo · Phase 5 — audit-then-act · v0.6.0"
)
