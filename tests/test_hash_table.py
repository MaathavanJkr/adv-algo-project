"""Tests for src.hash_table.HashTable."""

from __future__ import annotations

import pytest

from src.hash_functions import UniversalHash, make_fixed_hash
from src.hash_table import HashTable


def constant_hash(index: int):
    """Hash function that always returns index, to force collisions."""

    def _hash(key: int) -> int:
        return index

    return _hash


class TestInsertAndSearch:
    def test_inserted_keys_are_findable(self) -> None:
        table = HashTable(size=16, hash_function=make_fixed_hash(16))
        keys = [1, 2, 3, 17, 33, 1000]
        for key in keys:
            table.insert(key)
        for key in keys:
            assert table.search(key) is True

    def test_missing_key_not_found(self) -> None:
        table = HashTable(size=16, hash_function=make_fixed_hash(16))
        table.insert(5)
        assert table.search(999) is False

    def test_colliding_keys_all_searchable(self) -> None:
        # Force every key into bucket 0 regardless of value.
        table = HashTable(size=4, hash_function=constant_hash(0))
        keys = [10, 20, 30, 40, 50]
        for key in keys:
            table.insert(key)
        assert table.chain_length(0) == len(keys)
        for key in keys:
            assert table.search(key) is True

    def test_duplicate_insert_is_noop_and_tracked_as_unique(self) -> None:
        table = HashTable(size=8, hash_function=make_fixed_hash(8))
        assert table.insert(5) is True
        assert table.insert(5) is False
        assert table.insert(5) is False
        assert table.unique_keys == 1
        assert table.chain_length(5 % 8) == 1

    def test_duplicate_policy_identical_for_universal_hash(self) -> None:
        h = UniversalHash(m=8, seed=1)
        table = HashTable(size=8, hash_function=h)
        assert table.insert(42) is True
        assert table.insert(42) is False
        assert table.unique_keys == 1

    def test_delete_removes_key(self) -> None:
        table = HashTable(size=8, hash_function=make_fixed_hash(8))
        table.insert(3)
        assert table.search(3) is True
        assert table.delete(3) is True
        assert table.search(3) is False
        assert table.unique_keys == 0

    def test_delete_missing_key_returns_false(self) -> None:
        table = HashTable(size=8, hash_function=make_fixed_hash(8))
        assert table.delete(123) is False


class TestComparisonCounter:
    def test_counter_increments_on_search(self) -> None:
        # All keys forced into bucket 0, inserted in order.
        table = HashTable(size=4, hash_function=constant_hash(0))
        for key in [10, 20, 30]:
            table.insert(key)

        table.reset_counters()
        found = table.search(30)  # 3rd element in the chain.
        assert found is True
        assert table.comparison_count == 3

    def test_counter_counts_full_miss(self) -> None:
        table = HashTable(size=4, hash_function=constant_hash(0))
        for key in [10, 20, 30]:
            table.insert(key)

        table.reset_counters()
        found = table.search(999)
        assert found is False
        assert table.comparison_count == 3

    def test_counter_accumulates_across_searches(self) -> None:
        table = HashTable(size=4, hash_function=constant_hash(0))
        for key in [1, 2]:
            table.insert(key)

        table.reset_counters()
        table.search(1)  # 1 comparison
        table.search(2)  # 2 comparisons
        assert table.comparison_count == 3

    def test_reset_counters_returns_to_zero(self) -> None:
        table = HashTable(size=4, hash_function=constant_hash(0))
        table.insert(1)
        table.search(1)
        assert table.comparison_count > 0
        table.reset_counters()
        assert table.comparison_count == 0


class TestChainMetricsAccess:
    def test_max_chain_length_empty_table(self) -> None:
        table = HashTable(size=5, hash_function=make_fixed_hash(5))
        assert table.max_chain_length() == 0

    def test_average_chain_length(self) -> None:
        table = HashTable(size=4, hash_function=constant_hash(0))
        for key in [1, 2, 3, 4]:
            table.insert(key)
        # 4 keys all in bucket 0, 3 empty buckets -> avg = 4/4 = 1.0
        assert table.average_chain_length() == 1.0

    def test_chain_lengths_list(self) -> None:
        table = HashTable(size=3, hash_function=make_fixed_hash(3))
        table.insert(0)
        table.insert(3)  # same bucket as 0
        table.insert(1)
        assert table.chain_lengths() == [2, 1, 0]


class TestInvalidInputs:
    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError):
            HashTable(size=0, hash_function=make_fixed_hash(1))
        with pytest.raises(ValueError):
            HashTable(size=-1, hash_function=make_fixed_hash(1))

    def test_out_of_range_hash_raises(self) -> None:
        table = HashTable(size=4, hash_function=lambda k: 100)
        with pytest.raises(ValueError):
            table.insert(1)

    def test_size_one_table_works(self) -> None:
        table = HashTable(size=1, hash_function=make_fixed_hash(1))
        table.insert(42)
        table.insert(43)
        assert table.unique_keys == 2
        assert table.chain_length(0) == 2

    def test_large_integer_keys(self) -> None:
        big = 10**18 + 7
        table = HashTable(size=101, hash_function=make_fixed_hash(101))
        table.insert(big)
        assert table.search(big) is True
