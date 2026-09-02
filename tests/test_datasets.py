"""Tests for src.datasets generators."""

from __future__ import annotations

import pytest

from src.datasets import (
    adversarial_dataset,
    near_duplicate_dataset,
    sorted_dataset,
    uniform_dataset,
)


class TestUniformDataset:
    def test_yields_n_keys(self) -> None:
        for n in [0, 1, 5, 100]:
            assert len(uniform_dataset(n, seed=1)) == n

    def test_keys_in_expected_range(self) -> None:
        n = 50
        keys = uniform_dataset(n, seed=2)
        for key in keys:
            assert 1 <= key <= 10 * n

    def test_seed_reproducible(self) -> None:
        a = uniform_dataset(100, seed=123)
        b = uniform_dataset(100, seed=123)
        assert a == b

    def test_different_seeds_can_differ(self) -> None:
        a = uniform_dataset(100, seed=1)
        b = uniform_dataset(100, seed=2)
        assert a != b

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError):
            uniform_dataset(-1)


class TestSortedDataset:
    def test_exact_sequence(self) -> None:
        assert sorted_dataset(5) == [1, 2, 3, 4, 5]

    def test_n_zero(self) -> None:
        assert sorted_dataset(0) == []

    def test_n_one(self) -> None:
        assert sorted_dataset(1) == [1]

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError):
            sorted_dataset(-1)


class TestAdversarialDataset:
    def test_follows_formula(self) -> None:
        n, m = 10, 7
        keys = adversarial_dataset(n, m)
        assert keys == [i * m for i in range(1, n + 1)]

    def test_all_keys_are_multiples_of_m(self) -> None:
        n, m = 20, 13
        keys = adversarial_dataset(n, m)
        assert all(key % m == 0 for key in keys)

    def test_n_zero(self) -> None:
        assert adversarial_dataset(0, 10) == []

    def test_invalid_m_raises(self) -> None:
        with pytest.raises(ValueError):
            adversarial_dataset(10, 0)

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError):
            adversarial_dataset(-1, 10)


class TestNearDuplicateDataset:
    def test_yields_n_keys(self) -> None:
        for n in [0, 1, 10, 200]:
            assert len(near_duplicate_dataset(n, seed=1)) == n

    def test_pool_size_is_n_over_10(self) -> None:
        n = 200
        keys = near_duplicate_dataset(n, seed=5)
        distinct = set(keys)
        assert len(distinct) <= max(1, n // 10)

    def test_seed_reproducible(self) -> None:
        a = near_duplicate_dataset(100, seed=7)
        b = near_duplicate_dataset(100, seed=7)
        assert a == b

    def test_n_one_degenerate_pool(self) -> None:
        keys = near_duplicate_dataset(1, seed=1)
        assert len(keys) == 1

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError):
            near_duplicate_dataset(-1)
