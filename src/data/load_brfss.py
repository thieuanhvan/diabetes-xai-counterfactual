"""Load BRFSS 2021 cleaned dataset.

Reuse from P2 (diabetes-xai-agreement). Dataset spec:
- 236,378 rows, 21 features + 1 target (Diabetes_binary)
- Income encoding 2021: 1-11 (year-specific)
- Source: BRFSS 2021 julnazz-style cleaned CSV
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


EXPECTED_FEATURES_2021 = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
TARGET_COL = "Diabetes_binary"


def load_brfss_2021(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load BRFSS 2021 and return (X, y).

    Args:
        csv_path: Path to BRFSS 2021 cleaned CSV.

    Returns:
        X: DataFrame of 21 features.
        y: Series of binary target (Diabetes_binary).

    Raises:
        FileNotFoundError: csv_path does not exist.
        ValueError: schema mismatch with expected 2021 columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"BRFSS 2021 CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'")

    missing = set(EXPECTED_FEATURES_2021) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected BRFSS 2021 features: {sorted(missing)}")

    X = df[EXPECTED_FEATURES_2021].copy()
    y = df[TARGET_COL].astype(int)
    return X, y
