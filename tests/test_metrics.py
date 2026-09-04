"""Tests for src.metrics: ChainStats computation and adversarial
validation helper."""

from __future__ import annotations

from src.hash_functions import make_fixed_hash
from src.hash_table import HashTable
from src.metrics import compute_chain_stats, validate_adversarial_concentration


def constant_hash(index: int):
    def _hash(key: int) -> int:
        return index

    return _hash


class TestComputeChainStats:
    def test_hand_checkable_small_table(self) -> None:
        # size=4 buckets. Insert so bucket contents are known exactly:
        # bucket 0: [0, 4, 8]  (3 keys)
        # bucket 1: [1]        (1 key)
        # bucket 2: []         (0 keys)
        # bucket 3: [3, 7]     (2 keys)
        table = HashTable(size=4, hash_family=[make_fixed_hash(4)])
        for key in [0, 4, 8, 1, 3, 7]:
            table.insert(key)

        stats = compute_chain_stats(table)

        assert stats.max_chain_length == 3
        assert stats.non_empty_buckets == 3
        assert stats.empty_buckets == 1
        # average = (3 + 1 + 0 + 2) / 4 = 1.5
        assert stats.average_chain_length == 1.5
        # collisions = (3-1) + (1-1) + max(0-1,0) + (2-1) = 2 + 0 + 0 + 1 = 3
        assert stats.collision_count == 3

    def test_empty_table_all_zero(self) -> None:
        table = HashTable(size=5, hash_family=[make_fixed_hash(5)])
        stats = compute_chain_stats(table)
        assert stats.max_chain_length == 0
        assert stats.average_chain_length == 0.0
        assert stats.non_empty_buckets == 0
        assert stats.empty_buckets == 5
        assert stats.collision_count == 0
        assert stats.chain_length_stdev == 0.0

    def test_no_collisions_when_evenly_distributed(self) -> None:
        table = HashTable(size=4, hash_family=[make_fixed_hash(4)])
        for key in [0, 1, 2, 3]:
            table.insert(key)
        stats = compute_chain_stats(table)
        assert stats.collision_count == 0
        assert stats.max_chain_length == 1

    def test_collision_count_all_in_one_bucket(self) -> None:
        table = HashTable(size=4, hash_family=[constant_hash(0)])
        for key in [10, 20, 30, 40, 50]:
            table.insert(key)
        stats = compute_chain_stats(table)
        # 5 unique keys in one bucket -> 4 collisions.
        assert stats.collision_count == 4
        assert stats.max_chain_length == 5

    def test_duplicates_do_not_inflate_collision_count(self) -> None:
        table = HashTable(size=4, hash_family=[constant_hash(0)])
        table.insert(1)
        table.insert(1)
        table.insert(1)
        stats = compute_chain_stats(table)
        assert stats.collision_count == 0
        assert stats.max_chain_length == 1


class TestAdversarialValidation:
    def test_n100_m10_all_land_in_bucket_zero(self) -> None:
        max_chain = validate_adversarial_concentration(n=100, m=10)
        assert max_chain == 100

    def test_small_case(self) -> None:
        max_chain = validate_adversarial_concentration(n=5, m=3)
        assert max_chain == 5
