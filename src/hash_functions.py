"""Fixed hash and Carter-Wegman universal hash.

Both are callable as f(key) -> int so HashTable can use either one
interchangeably.
"""

from __future__ import annotations

import random
from functools import partial
from typing import Callable

# Mersenne prime, comfortably bigger than any key this project uses.
DEFAULT_PRIME = (1 << 61) - 1


def _is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin primality test (exact for n < 2**64)."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for sp in small_primes:
        if n == sp:
            return True
        if n % sp == 0:
            return False

    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for witness in small_primes:
        x = pow(witness, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def fixed_hash(key: int, m: int) -> int:
    """h(k) = k mod m. Deterministic, so an adversary who knows m can
    craft keys that all collide."""
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    return key % m


def make_fixed_hash(m: int) -> Callable[[int], int]:
    """Binds m so fixed_hash matches the f(key) -> int interface."""
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    return partial(fixed_hash, m=m)


class UniversalHash:
    """Carter-Wegman universal hash: h(k) = ((a*k + b) mod p) mod m.

    a and b are drawn once when the instance is created, never per
    key -- that's what gives the collision-probability guarantee.
    """

    def __init__(
        self,
        m: int,
        p: int = DEFAULT_PRIME,
        rng: random.Random | None = None,
        seed: int | None = None,
        a: int | None = None,
        b: int | None = None,
    ) -> None:
        # a/b can be passed explicitly to pin coefficients in tests;
        # normally they're drawn from rng/seed.
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if p < 2:
            raise ValueError(f"p must be >= 2, got {p}")
        if not _is_probable_prime(p):
            raise ValueError(f"p must be prime, got {p}")
        if p <= m:
            raise ValueError(f"p must be larger than m, got p={p}, m={m}")

        self.m = m
        self.p = p

        generator = rng if rng is not None else random.Random(seed)

        if a is not None:
            if not (1 <= a <= p - 1):
                raise ValueError(f"a must be in [1, p-1]={{1..{p - 1}}}, got {a}")
            self.a = a
        else:
            self.a = generator.randint(1, p - 1)

        if b is not None:
            if not (0 <= b <= p - 1):
                raise ValueError(f"b must be in [0, p-1]={{0..{p - 1}}}, got {b}")
            self.b = b
        else:
            self.b = generator.randint(0, p - 1)

    def __call__(self, key: int) -> int:
        return ((self.a * key + self.b) % self.p) % self.m

    def __repr__(self) -> str:
        return f"UniversalHash(a={self.a}, b={self.b}, p={self.p}, m={self.m})"
