"""Shared constants for the experiments."""

from __future__ import annotations

import math

N_VALUES: list[int] = [1000, 5000, 10000]
TRIALS: int = 30
LOAD_FACTORS: list[float] = [0.5, 0.75, 1.0, 2.0]
BASE_SEED: int = 42

# Family sizes (k) to try for universal hashing. k=1 reproduces plain
# single-function universal hashing exactly; a list so the experiment
# runner can sweep multiple family sizes in one run.
UNIVERSAL_K_VALUES: list[int] = [3]

DISTRIBUTIONS: list[str] = ["uniform", "sorted", "adversarial", "near_duplicate"]


def table_size_for_load_factor(n: int, alpha: float) -> int:
    """m = ceil(n / alpha), since alpha = n / m."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    m = math.ceil(n / alpha) if n > 0 else 1
    return max(m, 1)
