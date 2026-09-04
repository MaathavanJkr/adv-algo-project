"""Runs the timed trials comparing fixed hashing vs universal hashing.

For each (n, load_factor, distribution, k) setting, this runs
config.TRIALS trials. Each trial builds one dataset and inserts it
into both a fixed-hash table and a universal-hash table (family size
k), so the two methods are compared on the exact same input.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass

import config
from src.datasets import (
    adversarial_dataset,
    near_duplicate_dataset,
    sorted_dataset,
    uniform_dataset,
)
from src.hash_functions import make_fixed_family, make_universal_family
from src.hash_table import HashTable
from src.metrics import compute_chain_stats

# Separate seeds (offset from the trial seed) so that generating the
# dataset, drawing universal coefficients, choosing where to place
# keys, and sampling search keys don't affect each other.
_UNIVERSAL_PLACEMENT_OFFSET = 1_000_000
_FIXED_PLACEMENT_OFFSET = 2_000_000
_SEARCH_SAMPLE_OFFSET = 3_000_000


@dataclass(frozen=True)
class TrialResult:
    """One row of results: one method, one trial."""

    method: str
    distribution: str
    n: int
    m: int
    load_factor: float
    num_hash_functions: int
    trial: int
    insert_time_total: float
    insert_time_per_op: float
    search_time_total: float
    search_time_per_op: float
    search_comparisons_total: int
    search_comparisons_avg: float
    max_chain_length: int
    average_chain_length: float
    non_empty_buckets: int
    empty_buckets: int
    chain_length_stdev: float
    collision_count: int
    unique_keys: int
    p: int | None = None
    coefficients: str | None = None


def generate_keys(distribution: str, n: int, m: int, seed: int) -> list[int]:
    """Builds the dataset for one trial."""
    if distribution == "uniform":
        return uniform_dataset(n, seed=seed)
    elif distribution == "sorted":
        return sorted_dataset(n)
    elif distribution == "adversarial":
        return adversarial_dataset(n, m)
    elif distribution == "near_duplicate":
        return near_duplicate_dataset(n, seed=seed)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def _time_inserts(table: HashTable, keys: list[int]) -> float:
    start = time.perf_counter()
    for key in keys:
        table.insert(key)
    return time.perf_counter() - start


def _time_searches(table: HashTable, keys: list[int]) -> float:
    table.reset_counters()
    start = time.perf_counter()
    for key in keys:
        table.search(key)
    return time.perf_counter() - start


def _row_from_table(
    table: HashTable,
    *,
    method: str,
    distribution: str,
    n: int,
    m: int,
    load_factor: float,
    num_hash_functions: int,
    trial: int,
    insert_time: float,
    search_time: float,
    num_search_keys: int,
    p: int | None,
    coefficients: str | None,
) -> TrialResult:
    """Builds one result row from a table after insert + search have run."""
    stats = compute_chain_stats(table)
    return TrialResult(
        method=method,
        distribution=distribution,
        n=n,
        m=m,
        load_factor=load_factor,
        num_hash_functions=num_hash_functions,
        trial=trial,
        insert_time_total=insert_time,
        insert_time_per_op=insert_time / n if n else 0.0,
        search_time_total=search_time,
        search_time_per_op=search_time / num_search_keys if num_search_keys else 0.0,
        search_comparisons_total=table.comparison_count,
        search_comparisons_avg=(
            table.comparison_count / num_search_keys if num_search_keys else 0.0
        ),
        max_chain_length=stats.max_chain_length,
        average_chain_length=stats.average_chain_length,
        non_empty_buckets=stats.non_empty_buckets,
        empty_buckets=stats.empty_buckets,
        chain_length_stdev=stats.chain_length_stdev,
        collision_count=stats.collision_count,
        unique_keys=table.unique_keys,
        p=p,
        coefficients=coefficients,
    )


def run_single_trial(
    distribution: str,
    n: int,
    alpha: float,
    m: int,
    k: int,
    trial: int,
    base_seed: int,
) -> list[TrialResult]:
    """Runs one trial and returns [fixed_row, universal_row].

    Both tables get the same dataset and the same search-key sample,
    so the only difference between the two rows is the hash family.
    """
    seed = base_seed + trial

    keys = generate_keys(distribution, n, m, seed)
    assert len(keys) == n, f"expected {n} keys, got {len(keys)}"

    # Fixed hash: family of size 1.
    fixed_table = HashTable(
        size=m,
        hash_family=make_fixed_family(m),
        seed=seed + _FIXED_PLACEMENT_OFFSET,
    )
    fixed_insert_time = _time_inserts(fixed_table, keys)

    # Universal hash: family of size k. Coefficients are drawn once,
    # before any inserts, and never change during the trial.
    coeff_rng = random.Random(seed)
    universal_family = make_universal_family(k=k, m=m, rng=coeff_rng)
    coefficients = [(h.a, h.b) for h in universal_family]

    universal_table = HashTable(
        size=m,
        hash_family=universal_family,
        seed=seed + _UNIVERSAL_PLACEMENT_OFFSET,
    )
    universal_insert_time = _time_inserts(universal_table, keys)

    assert [(h.a, h.b) for h in universal_family] == coefficients, (
        "universal coefficients changed during the trial"
    )

    # Same n/2 search keys for both tables, sampled once.
    num_search_keys = n // 2
    sample_rng = random.Random(seed + _SEARCH_SAMPLE_OFFSET)
    search_keys = sample_rng.choices(keys, k=num_search_keys) if num_search_keys else []

    fixed_search_time = _time_searches(fixed_table, search_keys)
    fixed_row = _row_from_table(
        fixed_table,
        method="fixed",
        distribution=distribution,
        n=n,
        m=m,
        load_factor=alpha,
        num_hash_functions=1,
        trial=trial,
        insert_time=fixed_insert_time,
        search_time=fixed_search_time,
        num_search_keys=num_search_keys,
        p=None,
        coefficients=None,
    )

    universal_search_time = _time_searches(universal_table, search_keys)
    universal_row = _row_from_table(
        universal_table,
        method="universal",
        distribution=distribution,
        n=n,
        m=m,
        load_factor=alpha,
        num_hash_functions=k,
        trial=trial,
        insert_time=universal_insert_time,
        search_time=universal_search_time,
        num_search_keys=num_search_keys,
        p=universal_family[0].p,
        coefficients=json.dumps(coefficients),
    )

    return [fixed_row, universal_row]


def run_all_experiments(
    n_values: list[int] | None = None,
    trials: int | None = None,
    distributions: list[str] | None = None,
    load_factors: list[float] | None = None,
    k_values: list[int] | None = None,
    base_seed: int | None = None,
    progress: bool = True,
) -> list[TrialResult]:
    """Runs every (n, load_factor, distribution, k) combination.

    Any argument left as None falls back to the matching default in
    config.py.
    """
    n_values = n_values if n_values is not None else config.N_VALUES
    trials = trials if trials is not None else config.TRIALS
    distributions = distributions if distributions is not None else config.DISTRIBUTIONS
    load_factors = load_factors if load_factors is not None else config.LOAD_FACTORS
    k_values = k_values if k_values is not None else config.UNIVERSAL_K_VALUES
    base_seed = base_seed if base_seed is not None else config.BASE_SEED

    results: list[TrialResult] = []
    for n in n_values:
        for alpha in load_factors:
            m = config.table_size_for_load_factor(n, alpha)
            for distribution in distributions:
                for k in k_values:
                    if progress:
                        print(
                            f"[experiment] n={n} alpha={alpha} m={m} "
                            f"distribution={distribution} k={k} "
                            f"({trials} trials)"
                        )
                    for trial in range(trials):
                        results.extend(
                            run_single_trial(distribution, n, alpha, m, k, trial, base_seed)
                        )
    return results


def results_as_dicts(results: list[TrialResult]) -> list[dict]:
    """Converts result rows to plain dicts, for building a DataFrame."""
    return [asdict(r) for r in results]
