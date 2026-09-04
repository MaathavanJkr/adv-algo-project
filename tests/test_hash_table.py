"""Tests for src.hash_table.HashTable."""

from __future__ import annotations

import pytest

from src.hash_functions import (
    UniversalHash,
    make_fixed_family,
    make_fixed_hash,
    make_universal_family,
)
from src.hash_table import HashTable


def constant_hash(index: int):
    """Hash function that always returns index, to force collisions."""

    def _hash(key: int) -> int:
        return index

    return _hash


class TestInsertAndSearch:
    def test_inserted_keys_are_findable(self) -> None:
        table = HashTable(size=16, hash_family=[make_fixed_hash(16)])
        keys = [1, 2, 3, 17, 33, 1000]
        for key in keys:
            table.insert(key)
        for key in keys:
            assert table.search(key) is True

    def test_missing_key_not_found(self) -> None:
        table = HashTable(size=16, hash_family=[make_fixed_hash(16)])
        table.insert(5)
        assert table.search(999) is False

    def test_colliding_keys_all_searchable(self) -> None:
        # Force every key into bucket 0 regardless of value.
        table = HashTable(size=4, hash_family=[constant_hash(0)])
        keys = [10, 20, 30, 40, 50]
        for key in keys:
            table.insert(key)
        assert table.chain_length(0) == len(keys)
        for key in keys:
            assert table.search(key) is True

    def test_duplicate_insert_is_noop_and_tracked_as_unique(self) -> None:
        table = HashTable(size=8, hash_family=[make_fixed_hash(8)])
        assert table.insert(5) is True
        assert table.insert(5) is False
        assert table.insert(5) is False
        assert table.unique_keys == 1
        assert table.chain_length(5 % 8) == 1

    def test_duplicate_policy_identical_for_universal_hash(self) -> None:
        h = UniversalHash(m=8, seed=1)
        table = HashTable(size=8, hash_family=[h])
        assert table.insert(42) is True
        assert table.insert(42) is False
        assert table.unique_keys == 1

    def test_delete_removes_key(self) -> None:
        table = HashTable(size=8, hash_family=[make_fixed_hash(8)])
        table.insert(3)
        assert table.search(3) is True
        assert table.delete(3) is True
        assert table.search(3) is False
        assert table.unique_keys == 0

    def test_delete_missing_key_returns_false(self) -> None:
        table = HashTable(size=8, hash_family=[make_fixed_hash(8)])
        assert table.delete(123) is False


class TestComparisonCounter:
    def test_counter_increments_on_search(self) -> None:
        # All keys forced into bucket 0, inserted in order.
        table = HashTable(size=4, hash_family=[constant_hash(0)])
        for key in [10, 20, 30]:
            table.insert(key)

        table.reset_counters()
        found = table.search(30)  # 3rd element in the chain.
        assert found is True
        assert table.comparison_count == 3

    def test_counter_counts_full_miss(self) -> None:
        table = HashTable(size=4, hash_family=[constant_hash(0)])
        for key in [10, 20, 30]:
            table.insert(key)

        table.reset_counters()
        found = table.search(999)
        assert found is False
        assert table.comparison_count == 3

    def test_counter_accumulates_across_searches(self) -> None:
        table = HashTable(size=4, hash_family=[constant_hash(0)])
        for key in [1, 2]:
            table.insert(key)

        table.reset_counters()
        table.search(1)  # 1 comparison
        table.search(2)  # 2 comparisons
        assert table.comparison_count == 3

    def test_reset_counters_returns_to_zero(self) -> None:
        table = HashTable(size=4, hash_family=[constant_hash(0)])
        table.insert(1)
        table.search(1)
        assert table.comparison_count > 0
        table.reset_counters()
        assert table.comparison_count == 0


class TestChainMetricsAccess:
    def test_max_chain_length_empty_table(self) -> None:
        table = HashTable(size=5, hash_family=[make_fixed_hash(5)])
        assert table.max_chain_length() == 0

    def test_average_chain_length(self) -> None:
        table = HashTable(size=4, hash_family=[constant_hash(0)])
        for key in [1, 2, 3, 4]:
            table.insert(key)
        # 4 keys all in bucket 0, 3 empty buckets -> avg = 4/4 = 1.0
        assert table.average_chain_length() == 1.0

    def test_chain_lengths_list(self) -> None:
        table = HashTable(size=3, hash_family=[make_fixed_hash(3)])
        table.insert(0)
        table.insert(3)  # same bucket as 0
        table.insert(1)
        assert table.chain_lengths() == [2, 1, 0]


class TestInvalidInputs:
    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError):
            HashTable(size=0, hash_family=[make_fixed_hash(1)])
        with pytest.raises(ValueError):
            HashTable(size=-1, hash_family=[make_fixed_hash(1)])

    def test_out_of_range_hash_raises(self) -> None:
        table = HashTable(size=4, hash_family=[lambda k: 100])
        with pytest.raises(ValueError):
            table.insert(1)

    def test_size_one_table_works(self) -> None:
        table = HashTable(size=1, hash_family=[make_fixed_hash(1)])
        table.insert(42)
        table.insert(43)
        assert table.unique_keys == 2
        assert table.chain_length(0) == 2

    def test_large_integer_keys(self) -> None:
        big = 10**18 + 7
        table = HashTable(size=101, hash_family=[make_fixed_hash(101)])
        table.insert(big)
        assert table.search(big) is True


class TestHashFamily:
    def test_family_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError):
            HashTable(size=8, hash_family=[])

    def test_k1_family_matches_single_hash_behavior(self) -> None:
        # A family of length 1 should behave byte-for-byte like the
        # pre-refactor single-hash table: same bucket, same counters.
        m = 16
        family_table = HashTable(size=m, hash_family=make_fixed_family(m))
        single_table = HashTable(size=m, hash_family=[make_fixed_hash(m)])

        keys = [1, 2, 3, 17, 33, 1000]
        for key in keys:
            assert family_table.insert(key) == single_table.insert(key)
        assert family_table.chain_lengths() == single_table.chain_lengths()

        family_table.reset_counters()
        single_table.reset_counters()
        for key in keys:
            assert family_table.search(key) is True
            assert single_table.search(key) is True
        assert family_table.comparison_count == single_table.comparison_count

    def test_k1_universal_family_finds_every_key(self) -> None:
        m = 101
        family = make_universal_family(k=1, m=m, seed=1)
        table = HashTable(size=m, hash_family=family, seed=2)
        keys = list(range(500))
        for key in keys:
            table.insert(key)
        for key in keys:
            assert table.search(key) is True

    @pytest.mark.parametrize("k", [3, 5])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_multi_function_no_false_negatives(self, k: int, seed: int) -> None:
        m = 97
        family = make_universal_family(k=k, m=m, seed=seed)
        table = HashTable(size=m, hash_family=family, seed=seed + 1000)

        keys = list(range(1, 401))
        for key in keys:
            table.insert(key)

        for key in keys:
            assert table.search(key) is True, f"false negative for key={key}, k={k}, seed={seed}"

    def test_duplicate_insert_still_noop_with_multi_function_family(self) -> None:
        family = make_universal_family(k=4, m=50, seed=7)
        table = HashTable(size=50, hash_family=family, seed=8)
        assert table.insert(123) is True
        assert table.insert(123) is False
        assert table.insert(123) is False
        assert table.unique_keys == 1
        assert table.search(123) is True

    def test_placement_uses_shared_m_bucket_array(self) -> None:
        # All k members map into the same size-m array, so total
        # inserted keys across buckets equals unique_keys regardless
        # of which member placed each one.
        m = 30
        family = make_universal_family(k=5, m=m, seed=3)
        table = HashTable(size=m, hash_family=family, seed=4)
        for key in range(200):
            table.insert(key)
        assert sum(table.chain_lengths()) == table.unique_keys
        assert len(table.buckets) == m

    def test_search_counts_comparisons_across_probed_buckets(self) -> None:
        # Force each family member to map every key to a distinct
        # bucket, so a miss must scan all k buckets in full.
        def const(index: int):
            def _hash(key: int) -> int:
                return index
            return _hash

        family = [const(0), const(1), const(2)]
        table = HashTable(size=4, hash_family=family, seed=1)

        # Manually seed each bucket so probing all 3 costs 1+1+1.
        table.buckets[0].append(10)
        table.buckets[1].append(20)
        table.buckets[2].append(30)

        table.reset_counters()
        found = table.search(999)  # not present anywhere
        assert found is False
        assert table.comparison_count == 3

    def test_search_short_circuits_on_find(self) -> None:
        def const(index: int):
            def _hash(key: int) -> int:
                return index
            return _hash

        family = [const(0), const(1)]
        table = HashTable(size=4, hash_family=family, seed=1)
        table.buckets[0].append(42)  # placed by the first family member

        table.reset_counters()
        found = table.search(42)
        assert found is True
        assert table.comparison_count == 1

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            make_universal_family(k=0, m=10, seed=1)
        with pytest.raises(ValueError):
            make_universal_family(k=-2, m=10, seed=1)

    def test_family_members_draw_coefficients_once_each(self) -> None:
        family = make_universal_family(k=4, m=101, seed=5)
        before = [(h.a, h.b) for h in family]
        for h in family:
            for key in range(1000):
                h(key)
        after = [(h.a, h.b) for h in family]
        assert before == after

    def test_family_members_have_independent_coefficients(self) -> None:
        family = make_universal_family(k=5, m=101, seed=5)
        pairs = {(h.a, h.b) for h in family}
        assert len(pairs) > 1
