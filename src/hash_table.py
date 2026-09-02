"""Separate-chaining hash table with a pluggable hash function."""

from __future__ import annotations

import statistics
from typing import Callable


class HashTable:
    """Separate-chaining hash table that stores a set of integer keys.

    The hash function is injected, so the same table works with a
    fixed hash or a universal hash without any other code changing.
    """

    def __init__(self, size: int, hash_function: Callable[[int], int]) -> None:
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")

        self.size = size
        self.hash_function = hash_function
        self.buckets: list[list[int]] = [[] for _ in range(size)]
        self.unique_keys = 0
        self.comparison_count = 0

    def _bucket_index(self, key: int) -> int:
        index = self.hash_function(key)
        if not (0 <= index < self.size):
            raise ValueError(
                f"hash_function returned out-of-range index {index} "
                f"for table size {self.size}"
            )
        return index

    def insert(self, key: int) -> bool:
        """Insert key if not already present. Returns True if it was new."""
        index = self._bucket_index(key)
        bucket = self.buckets[index]
        if key in bucket:
            return False
        bucket.append(key)
        self.unique_keys += 1
        return True

    def search(self, key: int) -> bool:
        """Look up key, counting one comparison per element checked."""
        index = self._bucket_index(key)
        bucket = self.buckets[index]
        for candidate in bucket:
            self.comparison_count += 1
            if candidate == key:
                return True
        return False

    def delete(self, key: int) -> bool:
        """Remove key if present. Returns True if it was found."""
        index = self._bucket_index(key)
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
