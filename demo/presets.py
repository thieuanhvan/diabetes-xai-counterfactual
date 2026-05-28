"""
Archetypal patient presets for the counterfactual demo.

Hand-crafted clinical profiles spanning the diabetes-risk spectrum, used
to give the audience a one-click way to see different
CF behaviors without dialing 21 sidebar inputs.

Profiles are CURATED rather than mined from the BRFSS test set — this
trades real-patient authenticity for cross-machine reproducibility and
clinical coherence (each archetype tells one clear story).

The first entry ("Custom") is sentinel: when selected, the sidebar
inputs are NOT overwritten. All other entries are full 21-feature dicts
suitable for direct assignment to st.session_state["input_<feature>"].

Public API:
    PRESETS: Dict[str, Optional[Dict[str, float]]]
        Ordered dict (Python 3.7+) keyed by display name. Values are
        either None (Custom sentinel) or a full feature dict.
"""
from __future__ import annotations

from typing import Dict, Optional


PRESETS: Dict[str, Optional[Dict[str, float]]] = {

    # ────────────────────────────────────────────────────────────────
    # Custom — sentinel; the sidebar keeps whatever values are currently set.
    # ────────────────────────────────────────────────────────────────
    "Custom — use current sidebar values": None,

    # ────────────────────────────────────────────────────────────────
    # Very high risk — obese, multi-comorbid, sedentary smoker
    # Expected P(Diabetes=1) ≈ 0.75–0.90.
    # CF story: many actionable levers; expect 2-4 feature CFs.
    # ────────────────────────────────────────────────────────────────
    "Very high risk — obese, comorbid, sedentary": {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 35.0,
        "Smoker": 1, "Stroke": 0, "HeartDiseaseorAttack": 0,
        "PhysActivity": 0, "Fruits": 0, "Veggies": 0, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 1, "GenHlth": 4,
        "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 1,
        "Sex": 0, "Age": 10, "Education": 4, "Income": 4,
    },

    # ────────────────────────────────────────────────────────────────
    # High risk — mid-50s, controlled comorbidities, active
    # Expected P ≈ 0.30–0.45.
    # CF story: comorbidity control or lifestyle nudges.
    # ────────────────────────────────────────────────────────────────
    "High risk — mid-50s, comorbid but active": {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 29.0,
        "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
        "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3,
        "MentHlth": 2, "PhysHlth": 3, "DiffWalk": 0,
        "Sex": 0, "Age": 8, "Education": 5, "Income": 7,
    },

    # ────────────────────────────────────────────────────────────────
    # Moderate risk — young overweight, otherwise healthy
    # Expected P ≈ 0.10–0.20.
    # CF story: BMI is the dominant lever.
    # ────────────────────────────────────────────────────────────────
    "Moderate risk — young, overweight, otherwise healthy": {
        "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 28.0,
        "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
        "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3,
        "MentHlth": 2, "PhysHlth": 2, "DiffWalk": 0,
        "Sex": 0, "Age": 4, "Education": 5, "Income": 7,
    },

    # ────────────────────────────────────────────────────────────────
    # Low risk — active, normal weight, no comorbidities
    # Expected P < 0.05.
    # CF story: little to vary; useful for showing taxonomy filtering.
    # ────────────────────────────────────────────────────────────────
    "Low risk — active, normal weight, healthy lifestyle": {
        "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 22.0,
        "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
        "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 2,
        "MentHlth": 1, "PhysHlth": 1, "DiffWalk": 0,
        "Sex": 0, "Age": 6, "Education": 6, "Income": 8,
    },

    # ────────────────────────────────────────────────────────────────
    # Borderline — mixed signals, predicted probability near 0.5
    # Expected P ≈ 0.40–0.55.
    # CF story: small perturbations can flip the prediction either way —
    # method choice (random/kdtree/genetic) is most visible here.
    # ────────────────────────────────────────────────────────────────
    "Borderline — mixed signals, near decision boundary": {
        "HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 30.0,
        "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
        "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 3,
        "MentHlth": 3, "PhysHlth": 4, "DiffWalk": 0,
        "Sex": 0, "Age": 9, "Education": 4, "Income": 6,
    },
}


def load_presets() -> Dict[str, Optional[Dict[str, float]]]:
    """Return the preset dict (kept as a function for forward compatibility
    if presets ever move to a JSON artifact)."""
    return PRESETS
