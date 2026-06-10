"""
Standalone 95% bootstrap CI for validity and actionability.
Read-only over frozen run artifacts; does NOT touch the CF pipeline or overwrite any run.
Right-click run in PyCharm (no argparse, no module flags).
"""
import os
import numpy as np
import pandas as pd

ART = "outputs"                 # authoritative submit artifacts
MAIN = os.path.join(ART, "_ablation_archive", "taxonomy_5class")  # main 5-class run
EXT = os.path.join(ART, "external_2015")
N_BOOT = 1000
SEED = 42

def trunc4(x):
    if np.isnan(x):
        return float("nan")
    return int(x * 10000) / 10000  # truncation, not rounding (4 dp)

def boot_ci(values, n_boot=N_BOOT, seed=SEED):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = v[idx].mean()
    lo, hi = np.nanpercentile(means, [2.5, 97.5])
    return v.mean(), lo, hi, n

def report(tag, csv_path):
    df = pd.read_csv(csv_path)
    for metric in ["validity", "actionability"]:
        pt, lo, hi, n = boot_ci(df[metric].values)
        print(f"{tag:22s} {metric:14s} n={n:3d}  point={trunc4(pt):.4f}  "
              f"95% CI [{trunc4(lo):.4f}, {trunc4(hi):.4f}]")

print(f"Percentile bootstrap, {N_BOOT} resamples, seed {SEED}\n")
print("== MAIN (BRFSS 2021, 200 high-risk) ==")
report("per-query", os.path.join(MAIN, "perquery_cf_metrics.csv"))
report("global",    os.path.join(MAIN, "global_cf_metrics.csv"))
print("\n== EXTERNAL (BRFSS 2015, 200 high-risk) ==")
report("per-query", os.path.join(EXT, "perquery_cf_metrics.csv"))
report("global",    os.path.join(EXT, "global_cf_metrics.csv"))
