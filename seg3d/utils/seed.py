from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed `random`, `numpy` and `torch` (CPU and every CUDA device)."""
    random.seed(seed)
    np.random.seed(seed)

    import torch  # imported here so commands that never touch torch start quickly

    torch.manual_seed(seed)  # also seeds every CUDA device
