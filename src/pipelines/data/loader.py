"""Dataset loader for diabetes risk prediction.

Currently supports BRFSS 2021 cleaned schema. For future datasets (NHIS in P5,
NHANES in P5/P6), add schema constants + dispatch in load_dataset(). Function
name dataset-agnostic on purpose.

BRFSS 2021 spec:
- 236,378 rows, 21 features + 1 target (Diabetes_binary)
- Income encoding 2021: 1-11 (year-specific)
- Source: julnazz-style cleaned CSV
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# BRFSS 2021 schema constants
BRFSS_2021_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
TARGET_COL = "Diabetes_binary"


def load_dataset(
    csv_path: str | Path,
    sample_n: Optional[int] = None,
    sample_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset CSV and return (X, y).

    Args:
        csv_path: Path to cleaned dataset CSV (BRFSS 2021 schema).
        sample_n: if not None, randomly subsample N rows after schema check.
            Use for dev iteration; set None for paper-authoritative.
        sample_seed: RNG seed for reproducible subsampling.

    Returns:
        X: DataFrame of features.
        y: Series of binary target.

    Raises:
        FileNotFoundError: csv_path does not exist.
        ValueError: schema mismatch.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'")

    missing = set(BRFSS_2021_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected features: {sorted(missing)}")

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=sample_seed).reset_index(drop=True)

    X = df[BRFSS_2021_FEATURES].copy()
    y = df[TARGET_COL].astype(int)
    return X, y