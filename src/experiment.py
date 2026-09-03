"""Runs trials across n / load factor / dataset / hash strategy."""

from __future__ import annotations

import time
import random
from dataclasses import dataclass

import config
from src.datasets import uniform_dataset, sorted_dataset, adversarial_dataset, near_duplicate_dataset
from src.hash_functions import make_fixed_hash, UniversalHash
from src.hash_table import HashTable
from src.metrics import compute_chain_stats, ChainStats


@dataclass(frozen=True)
class ExperimentResult:
    n: int
    load_factor: float
    m: int
    dataset_name: str
    hash_strategy: str
    trial_index: int
    insert_time: float
    search_time: float
    stats: ChainStats


def _generate_keys(dataset_name: str, n: int, m: int, seed: int) -> list[int]:
    if dataset_name == "uniform":
        return uniform_dataset(n, seed=seed)
    elif dataset_name == "sorted":
        return sorted_dataset(n)
    elif dataset_name == "adversarial":
        return adversarial_dataset(n, m)
    elif dataset_name == "near_duplicate":
        return near_duplicate_dataset(n, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_all_experiments() -> list[ExperimentResult]:
    """Runs all trials and returns the collected results."""
    results: list[ExperimentResult] = []
    
    dataset_names = ["uniform", "sorted", "adversarial", "near_duplicate"]
    
    for n in config.N_VALUES:
        for alpha in config.LOAD_FACTORS:
            m = config.table_size_for_load_factor(n, alpha)
            for dataset_name in dataset_names:
                for trial in range(config.TRIALS):
                    # Derive per-trial seed
                    seed = config.BASE_SEED + trial
                    
                    keys = _generate_keys(dataset_name, n, m, seed)
                    
                    # 1. Fixed Hash
                    fixed_table = HashTable(size=m, hash_function=make_fixed_hash(m))
                    start = time.perf_counter()
                    for key in keys:
                        fixed_table.insert(key)
                    fixed_insert_time = time.perf_counter() - start
                    
                    fixed_table.reset_counters()
                    start = time.perf_counter()
                    for key in keys:
                        fixed_table.search(key)
                    fixed_search_time = time.perf_counter() - start
                    
                    fixed_stats = compute_chain_stats(fixed_table)
                    
                    results.append(ExperimentResult(
                        n=n, 
                        load_factor=alpha, 
                        m=m, 
                        dataset_name=dataset_name,
                        hash_strategy="fixed", 
                        trial_index=trial,
                        insert_time=fixed_insert_time, 
                        search_time=fixed_search_time,
                        stats=fixed_stats
                    ))
                    
                    # 2. Universal Hash
                    rng = random.Random(seed)
                    univ_hash = UniversalHash(m=m, rng=rng)
                    univ_table = HashTable(size=m, hash_function=univ_hash)
                    
                    start = time.perf_counter()
                    for key in keys:
                        univ_table.insert(key)
                    univ_insert_time = time.perf_counter() - start
                    
                    univ_table.reset_counters()
                    start = time.perf_counter()
                    for key in keys:
                        univ_table.search(key)
                    univ_search_time = time.perf_counter() - start
                    
                    univ_stats = compute_chain_stats(univ_table)
                    
                    results.append(ExperimentResult(
                        n=n, 
                        load_factor=alpha, 
                        m=m, 
                        dataset_name=dataset_name,
                        hash_strategy="universal", 
                        trial_index=trial,
                        insert_time=univ_insert_time, 
                        search_time=univ_search_time,
                        stats=univ_stats
                    ))
                    
    return results


if __name__ == "__main__":
    print("Running experiments...")
    results = run_all_experiments()
    print(f"Generated {len(results)} experiment results.")
