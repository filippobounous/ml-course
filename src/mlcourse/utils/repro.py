"""Reproducibility helpers: seeds, deterministic flags, device detection.

The heavy DL stack (`torch`) is optional, so its seeding is best-effort.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 0, *, deterministic_torch: bool = False) -> int:
    """Seed Python, NumPy, and (if installed) PyTorch for reproducibility.

    Returns the seed that was applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed


def detect_device() -> str:
    """Return a string identifier for the best available torch device.

    Preference order: cuda -> mps -> cpu. Returns "cpu" if torch is not installed.
    """
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"
