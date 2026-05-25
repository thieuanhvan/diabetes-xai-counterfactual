"""
Narrative generation for counterfactual recommendations.

Template-based, deterministic — runs offline (no LLM dependency) so the
demo works at MAPR oral / thesis defense even without internet.

Strategy:
    1. Compute feature deltas (only features whose CF value differs from
       the patient's current value).
    2. Group deltas by clinical category (biometric / comorbidity /
       lifestyle / self-rated / healthcare-access / demographic).
    3. For each category, emit a per-category mechanism explainer +
       per-feature phrasing of the change.

Per-feature phrasings use clinical verbs that match the direction of the
change ("Control high cholesterol" rather than the raw 1 → 0). Ordinal
features (GenHlth, Education, Income) get labelled values.

Public API:
    cf_to_narrative(query, cf, baseline_risk, cf_risk) -> str
        Returns a multi-line markdown string suitable for st.markdown().

    FEATURE_CATEGORY: Dict[str, str]
        Stable mapping used in app.py for grouping the delta table.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Feature → clinical category
# ─────────────────────────────────────────────────────────────────────
FEATURE_CATEGORY: Mapping[str, str] = {
    # Biometric (1)
    "BMI":                  "biometric",
    # Comorbidities (5)
    "HighBP":               "comorbidity",
    "HighChol":             "comorbidity",
    "Stroke":               "comorbidity",
    "HeartDiseaseorAttack": "comorbidity",
    "DiffWalk":             "comorbidity",
    # Behavioral lifestyle (5)
    "Smoker":               "lifestyle",
    "PhysActivity":         "lifestyle",
    "Fruits":               "lifestyle",
    "Veggies":              "lifestyle",
    "HvyAlcoholConsump":    "lifestyle",
    # Healthcare access (3)
    "AnyHealthcare":        "healthcare-access",
    "NoDocbcCost":          "healthcare-access",
    "CholCheck":            "healthcare-access",
    # Self-rated health (3)
    "GenHlth":              "self-rated",
    "MentHlth":             "self-rated",
    "PhysHlth":             "self-rated",
    # Demographic & socioeconomic (4)
    "Age":                  "demographic",
    "Sex":                  "demographic",
    "Education":            "demographic",
    "Income":               "demographic",
}


# Display order: biometric & comorbidity first (highest-leverage clinical
# levers), then lifestyle (patient-initiated), then access + self-rated,
# then demographic last (context only — immutable per taxonomy).
CATEGORY_ORDER = [
    "biometric",
    "comorbidity",
    "lifestyle",
    "healthcare-access",
    "self-rated",
    "demographic",
]


CATEGORY_HEADERS: Mapping[str, str] = {
    "biometric":         "Biometric targets",
    "comorbidity":       "Comorbidity management",
    "lifestyle":         "Behavioral & lifestyle adjustments",
    "healthcare-access": "Healthcare access",
    "self-rated":        "Self-rated health",
    "demographic":       "Demographic & socioeconomic (context only)",
}


CATEGORY_MECHANISM: Mapping[str, str] = {
    "biometric": (
        "Body composition directly influences insulin resistance and glycemic "
        "control. Reduction strategies typically combine dietary change with "
        "physical activity."
    ),
    "comorbidity": (
        "Comorbidities co-occur with and amplify diabetes risk. Management is "
        "typically clinical (medication + monitoring) rather than purely "
        "behavioral."
    ),
    "lifestyle": (
        "Behavioral changes the patient can initiate directly. Effect is "
        "mediated through metabolic and cardiovascular pathways; reinforced "
        "by trusted-source guidance."
    ),
    "healthcare-access": (
        "Access enables earlier detection and consistent management. Often "
        "constrained by insurance status and cost — not purely a patient "
        "choice."
    ),
    "self-rated": (
        "The patient's own appraisal of health status. Improvement typically "
        "reflects underlying biometric or comorbidity gains rather than a "
        "directly actionable lever."
    ),
    "demographic": (
        "These features are not actionable but explain residual risk. Shown "
        "for context only. CFs should not change these — if they do, the "
        "taxonomy constraint has been violated."
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Ordinal value labels (where helpful)
# ─────────────────────────────────────────────────────────────────────
GENHLTH_LABELS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EDUCATION_LABELS = {
    1: "no schooling",
    2: "elementary",
    3: "some high school",
    4: "high school grad",
    5: "some college",
    6: "college graduate",
}
INCOME_LABELS = {
    1: "<\\$10K",  2: "\\$10–15K", 3: "\\$15–20K", 4: "\\$20–25K",
    5: "\\$25–35K", 6: "\\$35–50K", 7: "\\$50–75K", 8: "\\$75–100K",
    9: "\\$100–150K", 10: "\\$150–200K", 11: "≥\\$200K",
}
AGE_LABELS = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
    11: "70–74", 12: "75–79", 13: "80+",
}


# ─────────────────────────────────────────────────────────────────────
# Per-feature phrasing helpers
# ─────────────────────────────────────────────────────────────────────
def _phrase_bmi(c: float, n: float) -> str:
    delta = n - c
    verb = "Reduce" if delta < 0 else "Increase"
    return (
        f"**{verb} BMI** from {c:.1f} to {n:.1f} kg/m² "
        f"(Δ = {delta:+.1f} kg/m²)."
    )


def _phrase_binary_down(label: str, c: int, n: int) -> str:
    """For MONOTONIC_DOWN binary features. Verb depends on the meaning."""
    if c == 1 and n == 0:
        return f"**{label}** ({c} → {n})."
    # CF in wrong direction — should be filtered out by taxonomy
    return f"⚠️ {label} moved in unhealthy direction ({c} → {n})."


def _phrase_binary_up(label: str, c: int, n: int) -> str:
    """For MONOTONIC_UP binary features."""
    if c == 0 and n == 1:
        return f"**{label}** ({c} → {n})."
    return f"⚠️ {label} moved in unhealthy direction ({c} → {n})."


def _phrase_ordinal(
    display_name: str, c: int, n: int,
    labels: Mapping[int, str] | None = None,
    unit: str = "",
) -> str:
    if labels:
        c_lab, n_lab = labels.get(int(c), str(c)), labels.get(int(n), str(n))
        return (
            f"**{display_name}**: {int(c)} ({c_lab}) → {int(n)} ({n_lab})."
        )
    suffix = f" {unit}" if unit else ""
    return f"**{display_name}**: {int(c)}{suffix} → {int(n)}{suffix}."


def phrase_change(feature: str, current, counterfactual) -> str:
    """Return a single-line markdown phrasing for one feature change."""
    c, n = current, counterfactual

    # Biometric
    if feature == "BMI":
        return _phrase_bmi(float(c), float(n))

    # Comorbidity (binary, monotonic down)
    if feature == "HighBP":
        return _phrase_binary_down("Control high blood pressure", int(c), int(n))
    if feature == "HighChol":
        return _phrase_binary_down("Control high cholesterol", int(c), int(n))
    if feature == "DiffWalk":
        return _phrase_binary_down(
            "Address walking difficulty (depends on co-morbidity)", int(c), int(n),
        )
    if feature in ("Stroke", "HeartDiseaseorAttack"):
        # Immutable — should not appear in valid CFs
        return f"⚠️ {feature} flagged as changed — immutable per taxonomy."

    # Lifestyle (binary)
    if feature == "Smoker":
        return _phrase_binary_down("Quit smoking", int(c), int(n))
    if feature == "HvyAlcoholConsump":
        return _phrase_binary_down("Reduce heavy alcohol consumption", int(c), int(n))
    if feature == "PhysActivity":
        return _phrase_binary_up("Begin regular physical activity (past 30 days)", int(c), int(n))
    if feature == "Fruits":
        return _phrase_binary_up("Eat fruit at least once per day", int(c), int(n))
    if feature == "Veggies":
        return _phrase_binary_up("Eat vegetables at least once per day", int(c), int(n))

    # Healthcare access
    if feature == "AnyHealthcare":
        return _phrase_binary_up("Obtain healthcare coverage", int(c), int(n))
    if feature == "NoDocbcCost":
        return _phrase_binary_down("Address cost barrier to seeing a doctor", int(c), int(n))
    if feature == "CholCheck":
        return _phrase_binary_up("Get a cholesterol check (past 5 years)", int(c), int(n))

    # Self-rated health (ordinal, monotonic down)
    if feature == "GenHlth":
        return _phrase_ordinal("General health", int(c), int(n), labels=GENHLTH_LABELS)
    if feature == "MentHlth":
        return _phrase_ordinal("Bad mental-health days/month", int(c), int(n), unit="days")
    if feature == "PhysHlth":
        return _phrase_ordinal("Bad physical-health days/month", int(c), int(n), unit="days")

    # Demographic & socioeconomic — should not appear in valid CFs except
    # for Income/Education in the default 5-class taxonomy (MONOTONIC_UP).
    if feature == "Age":
        return f"⚠️ Age flagged as changed — immutable per taxonomy."
    if feature == "Sex":
        return f"⚠️ Sex flagged as changed — immutable per taxonomy."
    if feature == "Education":
        return _phrase_ordinal(
            "Education level", int(c), int(n), labels=EDUCATION_LABELS,
        )
    if feature == "Income":
        return _phrase_ordinal(
            "Income bracket", int(c), int(n), labels=INCOME_LABELS,
        )

    # Fallback for unknown features
    return f"**{feature}**: {c} → {n}."


# ─────────────────────────────────────────────────────────────────────
# Top-level renderer
# ─────────────────────────────────────────────────────────────────────
def cf_to_narrative(
    query: pd.Series,
    cf: pd.Series,
    baseline_risk: float,
    cf_risk: float,
) -> str:
    """
    Render a counterfactual as a clinical-style recommendation narrative.

    Args:
        query:         Patient's current feature values.
        cf:            Counterfactual feature values (post-rounding for discretes).
        baseline_risk: P(Diabetes=1) on `query` from the model.
        cf_risk:       P(Diabetes=1) on `cf` from the model.

    Returns:
        Multi-line markdown string. Pass directly to st.markdown().
    """
    # Group changes by category
    by_category: dict[str, list[tuple[str, object, object]]] = defaultdict(list)
    for feat in FEATURE_CATEGORY:
        if feat not in cf.index or feat not in query.index:
            continue
        c, n = query[feat], cf[feat]
        if pd.isna(c) or pd.isna(n) or c == n:
            continue
        by_category[FEATURE_CATEGORY[feat]].append((feat, c, n))

    if not by_category:
        return (
            "**No actionable changes identified.** The counterfactual matches "
            "the patient's profile after rounding."
        )

    # Header line
    delta_pp = (cf_risk - baseline_risk) * 100
    n_changed = sum(len(v) for v in by_category.values())
    plural = "feature" if n_changed == 1 else "features"
    lines: list[str] = [
        f"**Predicted risk: {baseline_risk:.1%} → {cf_risk:.1%}** "
        f"({delta_pp:+.1f} percentage points). "
        f"Achieved by changing **{n_changed} {plural}**.",
        "",
    ]

    # Render each category in canonical order
    for cat in CATEGORY_ORDER:
        if cat not in by_category:
            continue
        lines.append(f"##### {CATEGORY_HEADERS[cat]}")
        lines.append(f"*{CATEGORY_MECHANISM[cat]}*")
        lines.append("")
        for feat, c, n in by_category[cat]:
            lines.append(f"- {phrase_change(feat, c, n)}")
        lines.append("")

    # Limitations footer
    lines.extend([
        "---",
        (
            "**Limitations.** This is an algorithmic recommendation from a "
            "model trained on BRFSS 2021 self-reports — not medical advice. "
            "Direction of effect is correlation-implied, not causally "
            "validated. Comorbidity-management items require clinical "
            "supervision."
        ),
    ])

    return "\n".join(lines)
