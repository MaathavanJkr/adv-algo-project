"""Chain/collision statistics for a populated HashTable."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from src.datasets import adversarial_dataset
from src.hash_functions import make_fixed_family
from src.hash_table import HashTable


@dataclass(frozen=True)
class ChainStats:
    max_chain_length: int
    average_chain_length: float
    non_empty_buckets: int
    empty_buckets: int
    chain_length_stdev: float
    collision_count: int


def compute_chain_stats(table: HashTable) -> ChainStats:
    """collision_count = unique keys beyond the first in each bucket, summed."""
    lengths = table.chain_lengths()

    non_empty = sum(1 for length in lengths if length > 0)
    empty = len(lengths) - non_empty
    collisions = sum(max(length - 1, 0) for length in lengths)
    stdev = statistics.pstdev(lengths) if len(lengths) > 0 else 0.0

    return ChainStats(
        max_chain_length=max(lengths, default=0),
        average_chain_length=statistics.mean(lengths) if lengths else 0.0,
        non_empty_buckets=non_empty,
        empty_buckets=empty,
        chain_length_stdev=stdev,
        collision_count=collisions,
    )


def validate_adversarial_concentration(n: int, m: int) -> int:
    """Check that the adversarial dataset all lands in bucket 0 under
    fixed hashing, and return the resulting max chain length."""
    keys = adversarial_dataset(n, m)
    table = HashTable(size=m, hash_family=make_fixed_family(m))
    for key in keys:
        table.insert(key)

    for index, bucket in enumerate(table.buckets):
        if index != 0:
            assert not bucket, f"expected bucket {index} to be empty, found {bucket}"

    assert table.chain_length(0) == table.unique_keys

    return table.max_chain_length()
