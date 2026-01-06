from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 0) -> None:
    """Best-effort reproducibility for Week 0 experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
