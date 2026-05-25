# P4 Counterfactual Demo

Streamlit app for interactive exploration of counterfactual recommendations on diabetes risk predictions (BRFSS 2021).

Companion artifact to **P4 — Knowledge-Guided Constraint Compiler for Actionable CFs in Diabetes Risk Prediction**. Designed for live demo at MAPR oral (if accepted) and thesis defense.

## Status

**Phase 5 — Audit-then-act.** Sidebar preset selector (6 archetypes from very-high risk to low risk + boundary), main-panel method selector (random / kdtree / genetic). On Generate, all 3 DiCE methods run; main panel shows the selected method's full result (gauges + narrative + waterfall); a new **Compare methods** section displays all 3 best CFs side-by-side. Smoke-tested: 3 methods on the same patient produced 3 distinct recommendations (1, 4, 10 features changed respectively) — the headline observation for the thesis defense narrative.

## Setup

```bash
# From repo root (diabetes-xai-counterfactual)
python -m venv .venv-demo
source .venv-demo/bin/activate    # Windows: .venv-demo\Scripts\activate
pip install -r demo/requirements.txt
```

Then materialize the model and training-sample artifacts (one-time, ~30-60 s):

```bash
python demo/prepare_demo_artifacts.py
```

This reuses P4's own pipeline (loader → split → xgb_train) — no logic duplicated. Outputs land in `demo/models/`. The app warns gracefully if artifacts are absent.

## Run

```bash
streamlit run demo/app.py
```

App opens at <http://localhost:8501>.

## Structure

```
demo/
├── app.py                       # Streamlit entry point
├── prepare_demo_artifacts.py    # One-time artifact prep (reuses P4 pipeline)
├── narrative.py                 # Template-based CF → text (Phase 3)
├── visualizations.py            # Plotly gauges + bar chart + waterfall (Phase 4)
├── presets.py                   # Phase 5 — archetypal patient cache
├── models/                      # Generated artifacts (gitignored except .gitkeep)
│   ├── xgb_brfss2021.joblib
│   ├── X_train_sample.parquet
│   ├── y_train_sample.parquet
│   ├── X_test.parquet
│   ├── proba_test.parquet
│   └── metadata.json
├── requirements.txt
└── README.md
```

## Roadmap

| Phase | Scope                                                | Status   | Est. |
|-------|------------------------------------------------------|----------|------|
| 1     | Scaffold (layout, 10-feature stub)                   | **Done** |  2 h |
| 1.5   | Full 21-feature form + artifact loaders + prep script| **Done** |  1 h |
| 2     | DiCE CF generation + baseline risk                   | **Done** |  3 h |
| 3     | Template-based narrative (offline, no LLM)           | **Done** |  2 h |
| 4     | Risk gauge + feature delta charts (Plotly)           | **Done** |  2 h |
| 5     | Preset patients + 3 DiCE methods side-by-side        | **Done** |  3 h |
| 6     | Polish, README, backup demo video                    | Pending  |  1 h |

## Notes

- Demo runs **fully offline** — no LLM API calls. Intentional for oral presentation reliability.
- Phase 5 compares **DiCE-random vs DiCE-kdtree vs DiCE-genetic** on the same patient. This mirrors P4's `configs/ablation_method_*.yaml` framework and bridges to P2's cross-method audit narrative (method-choice sensitivity within the CF family ↔ method-choice sensitivity within the explanation family).
- Repo stays **private** until P4 acceptance.
