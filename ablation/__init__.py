"""Ablation studies harness for P4 §4.5.

Each `run_ablation_<name>.py` at repo root invokes ablation/core.py helpers
to (i) clone configs/default.yaml, (ii) override 1-2 fields, (iii) call
src.pipelines.main with the overridden config. Results land in outputs/ as
standard run_<timestamp>.json (with v4 classifier_metrics field).

ablation/aggregate.py reads run JSONs after the ablation finishes and writes
comparison CSVs for Bảng 4.5.1-4.5.5 in main_vi v5.

Architecture per ablation_vi v2 §8 + Generalrule v37 §11.4/§11.5/§11.6.
"""
