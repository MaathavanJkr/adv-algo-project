"""Shared constants for the experiments."""

from __future__ import annotations

import math

N_VALUES: list[int] = [1000, 5000, 10000]
N_VALUES_CHAIN: list[int] = [10, 100, 1000, 5000, 100000]
TRIALS: int = 1
LOAD_FACTORS: list[float] = [0.5, 0.75, 1.0, 2.0]
BASE_SEED: int = 42


def table_size_for_load_factor(n: int, alpha: float) -> int:
    """m = ceil(n / alpha), since alpha = n / m."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    m = math.ceil(n / alpha) if n > 0 else 1
    return max(m, 1)
