"""Tests for src.hash_functions: fixed_hash and UniversalHash."""

from __future__ import annotations

import random

import pytest

from src.hash_functions import (
    DEFAULT_PRIME,
    UniversalHash,
    fixed_hash,
    make_fixed_family,
    make_fixed_hash,
    make_universal_family,
)


class TestFixedHash:
    def test_known_values(self) -> None:
        assert fixed_hash(20, 10) == 0
        assert fixed_hash(31, 10) == 1

    def test_result_in_range(self) -> None:
        m = 7
        for key in range(0, 100):
            result = fixed_hash(key, m)
            assert 0 <= result < m

    def test_deterministic(self) -> None:
        assert fixed_hash(123, 17) == fixed_hash(123, 17)

    def test_invalid_m_raises(self) -> None:
        with pytest.raises(ValueError):
            fixed_hash(5, 0)
        with pytest.raises(ValueError):
            fixed_hash(5, -3)

    def test_make_fixed_hash_adapter_matches(self) -> None:
        m = 13
        adapted = make_fixed_hash(m)
        for key in range(50):
            assert adapted(key) == fixed_hash(key, m)

    def test_make_fixed_hash_invalid_m_raises(self) -> None:
        with pytest.raises(ValueError):
            make_fixed_hash(0)


class TestUniversalHash:
    def test_output_always_in_range(self) -> None:
        m = 11
        h = UniversalHash(m=m, seed=1)
        for key in range(0, 1000):
            result = h(key)
            assert 0 <= result < m

    def test_a_never_zero(self) -> None:
        for seed in range(200):
            h = UniversalHash(m=10, seed=seed)
            assert h.a != 0

    def test_deterministic_for_fixed_params(self) -> None:
        h1 = UniversalHash(m=10, seed=99)
        h2 = UniversalHash(m=10, seed=99)
        # Same seed -> same (a, b) -> same outputs for every key.
        assert h1.a == h2.a
        assert h1.b == h2.b
        for key in range(100):
            assert h1(key) == h2(key)

    def test_same_instance_deterministic_repeated_calls(self) -> None:
        h = UniversalHash(m=10, seed=7)
        for key in [1, 2, 3, 100, 12345]:
            assert h(key) == h(key)

    def test_invalid_m_raises(self) -> None:
        with pytest.raises(ValueError):
            UniversalHash(m=0)
        with pytest.raises(ValueError):
            UniversalHash(m=-5)

    def test_explicit_a_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            UniversalHash(m=10, a=0)

    def test_explicit_a_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            UniversalHash(m=10, p=13, a=13)  # a must be <= p - 1

    def test_explicit_b_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            UniversalHash(m=10, p=13, b=13)  # b must be <= p - 1

    def test_explicit_a_and_b_are_used_verbatim(self) -> None:
        h = UniversalHash(m=10, p=13, a=3, b=5)
        assert h.a == 3
        assert h.b == 5
        assert h(7) == ((3 * 7 + 5) % 13) % 10

    def test_non_prime_p_raises(self) -> None:
        with pytest.raises(ValueError):
            UniversalHash(m=5, p=100)  # 100 is not prime

    def test_p_too_small_raises(self) -> None:
        # p must be > m; here p == m.
        with pytest.raises(ValueError):
            UniversalHash(m=13, p=13)

    def test_p_less_than_two_raises(self) -> None:
        with pytest.raises(ValueError):
            UniversalHash(m=5, p=1)

    def test_default_prime_is_used_and_valid(self) -> None:
        h = UniversalHash(m=10, seed=3)
        assert h.p == DEFAULT_PRIME

    def test_coefficients_drawn_once_per_instance_not_per_key(self) -> None:
        h = UniversalHash(m=97, seed=5)
        a_before, b_before, p_before = h.a, h.b, h.p
        for key in range(10_000):
            h(key)
        assert h.a == a_before
        assert h.b == b_before
        assert h.p == p_before

    def test_two_instances_can_have_different_coefficients(self) -> None:
        seen_pairs = {
            (UniversalHash(m=101, seed=None).a, UniversalHash(m=101, seed=None).b)
            for _ in range(20)
        }
        # 20 fresh instances should not all draw the same (a, b).
        assert len(seen_pairs) > 1

    def test_explicit_rng_allows_independent_instances(self) -> None:
        rng = random.Random(42)
        h1 = UniversalHash(m=50, rng=rng)
        h2 = UniversalHash(m=50, rng=rng)
        assert (h1.a, h1.b) != (h2.a, h2.b)

    def test_repr_contains_params(self) -> None:
        h = UniversalHash(m=10, seed=1)
        text = repr(h)
        assert "a=" in text and "b=" in text and "p=" in text and "m=" in text


class TestMakeFixedFamily:
    def test_length_one_containing_fixed_hash(self) -> None:
        family = make_fixed_family(13)
        assert len(family) == 1
        for key in range(50):
            assert family[0](key) == fixed_hash(key, 13)

    def test_invalid_m_raises(self) -> None:
        with pytest.raises(ValueError):
            make_fixed_family(0)


class TestMakeUniversalFamily:
    def test_family_has_k_members(self) -> None:
        for k in (1, 2, 3, 5):
            family = make_universal_family(k=k, m=97, seed=1)
            assert len(family) == k
            assert all(isinstance(h, UniversalHash) for h in family)

    def test_k_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError):
            make_universal_family(k=0, m=10, seed=1)
        with pytest.raises(ValueError):
            make_universal_family(k=-1, m=10, seed=1)

    def test_all_members_output_in_range(self) -> None:
        family = make_universal_family(k=4, m=23, seed=2)
        for h in family:
            for key in range(200):
                assert 0 <= h(key) < 23

    def test_members_are_independent_not_identical(self) -> None:
        family = make_universal_family(k=6, m=101, seed=9)
        pairs = {(h.a, h.b) for h in family}
        assert len(pairs) > 1

    def test_reproducible_given_same_seed(self) -> None:
        family1 = make_universal_family(k=4, m=101, seed=123)
        family2 = make_universal_family(k=4, m=101, seed=123)
        assert [(h.a, h.b) for h in family1] == [(h.a, h.b) for h in family2]

    def test_k1_matches_single_universal_hash_given_same_rng_state(self) -> None:
        family = make_universal_family(k=1, m=50, seed=77)
        solo = UniversalHash(m=50, seed=77)
        assert (family[0].a, family[0].b) == (solo.a, solo.b)
