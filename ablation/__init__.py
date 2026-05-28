"""Ablation-studies harness.

Each `run_ablation_<name>.py` at the repo root invokes ablation/core.py
helpers to (i) clone configs/default.yaml, (ii) override 1-2 fields, and
(iii) call src.pipelines.main with the overridden config. Results land in
outputs/ as standard run_<timestamp>.json (with the classifier_metrics field).

ablation/aggregate.py reads the run JSONs after the ablation finishes and
writes the comparison CSVs for the ablation summary tables.
"""
