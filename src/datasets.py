"""Seedable key generators for the four test distributions."""

from __future__ import annotations

import random


def uniform_dataset(n: int, seed: int | None = None) -> list[int]:
    """n random keys from [1, 10n]."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return []

    upper = max(10 * n, 1)
    rng = random.Random(seed)
    return [rng.randint(1, upper) for _ in range(n)]


def sorted_dataset(n: int) -> list[int]:
    """1, 2, ..., n."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return list(range(1, n + 1))


def adversarial_dataset(n: int, m: int) -> list[int]:
    """k_i = i * m. Every key maps to bucket 0 under h(k) = k mod m."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    return [i * m for i in range(1, n + 1)]


def near_duplicate_dataset(n: int, seed: int | None = None) -> list[int]:
    """n keys drawn from a pool of size n // 10, so values repeat a lot."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return []

    pool_size = max(1, n // 10)
    pool = list(range(1, pool_size + 1))
    rng = random.Random(seed)
    return [rng.choice(pool) for _ in range(n)]
