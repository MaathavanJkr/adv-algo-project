"""Separate-chaining hash table backed by a family of hash functions.

A table is given a family of k>=1 hash callables sharing one m-bucket
array. Insert picks one family member uniformly at random (per-table
placement RNG) to place the key; search probes every family member's
candidate bucket (deduped, deterministic order) so a key is found no
matter which member placed it. k=1 collapses to classic single-hash
behaviour: "pick one at random" is trivial and "probe all" probes the
single bucket, so a fixed-hash table (family of length 1) behaves
byte-for-byte like the pre-refactor single-hash table.
"""

from __future__ import annotations

import random
import statistics
from typing import Callable, Sequence


class HashTable:
    """Separate-chaining hash table that stores a set of integer keys.

    hash_family is a list of f(key) -> int callables, all sharing this
    table's m-bucket array. Fixed hashing is a family of size 1;
    universal hashing is a family of size k. Only the injected family
    differs between the two methods -- everything else is one code
    path.
    """

    def __init__(
        self,
        size: int,
        hash_family: Sequence[Callable[[int], int]],
        seed: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        if not hash_family:
            raise ValueError("hash_family must contain at least one hash function")

        self.size = size
        self.hash_family = list(hash_family)
        self.buckets: list[list[int]] = [[] for _ in range(size)]
        self.unique_keys = 0
        self.comparison_count = 0

        # Placement RNG: independent of whatever RNG drew the family's
        # own coefficients, but fixed for this table instance so which
        # member places a given key is reproducible.
        self._placement_rng = rng if rng is not None else random.Random(seed)

    def _bucket_index(self, hash_function: Callable[[int], int], key: int) -> int:
        index = hash_function(key)
        if not (0 <= index < self.size):
            raise ValueError(
                f"hash_function returned out-of-range index {index} "
                f"for table size {self.size}"
            )
        return index

    def _candidate_buckets(self, key: int) -> list[int]:
        """Every family member's bucket for this key, deduped, in a
        deterministic order (first occurrence, following family order)."""
        seen: set[int] = set()
        order: list[int] = []
        for hash_function in self.hash_family:
            index = self._bucket_index(hash_function, key)
            if index not in seen:
                seen.add(index)
                order.append(index)
        return order

    def insert(self, key: int) -> bool:
        """Insert key if not already present. Returns True if it was new.

        Presence is checked across every candidate bucket (the key
        could have been placed by any family member), then one member
        is chosen uniformly at random to place a genuinely new key.
        """
        candidates = self._candidate_buckets(key)
        for index in candidates:
            if key in self.buckets[index]:
                return False

        chosen_function = self._placement_rng.choice(self.hash_family)
        chosen_index = self._bucket_index(chosen_function, key)
        self.buckets[chosen_index].append(key)
        self.unique_keys += 1
        return True

    def search(self, key: int) -> bool:
        """Look up key, counting one comparison per element checked.

        Probes every candidate bucket (deduped), short-circuiting as
        soon as the key is found. A full miss scans -- and counts --
        every element in every candidate bucket.
        """
        for index in self._candidate_buckets(key):
            bucket = self.buckets[index]
            for candidate in bucket:
                self.comparison_count += 1
                if candidate == key:
                    return True
        return False

    def delete(self, key: int) -> bool:
        """Remove key if present. Returns True if it was found."""
        for index in self._candidate_buckets(key):
            bucket = self.buckets[index]
            for i, candidate in enumerate(bucket):
                if candidate == key:
                    del bucket[i]
                    self.unique_keys -= 1
                    return True
        return False

    def chain_length(self, index: int) -> int:
        if not (0 <= index < self.size):
            raise ValueError(f"index {index} out of range for size {self.size}")
        return len(self.buckets[index])

    def chain_lengths(self) -> list[int]:
        return [len(bucket) for bucket in self.buckets]

    def max_chain_length(self) -> int:
        return max((len(bucket) for bucket in self.buckets), default=0)

    def average_chain_length(self) -> float:
        return statistics.mean(len(bucket) for bucket in self.buckets)

    def reset_counters(self) -> None:
        self.comparison_count = 0
