"""Seed all randomness for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PYTHONHASHSEED for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
