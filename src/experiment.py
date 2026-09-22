"""Runs trials across n / load factor / dataset / hash strategy."""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from pathlib import Path

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
    iterations: int
    insert_times: list[float]
    search_times: list[float]
    stats: ChainStats

@dataclass(frozen=True)
class ChainGrowthResult:
    n: int
    load_factor: float
    m: int
    dataset_name: str
    hash_strategy: str
    trial_index: int
    chain_growth: list[int]
    collision_growth: list[int]

def benchmark_table(table: HashTable, keys: list[int], iters: int) -> tuple[list[float], list[float]]:
    insert_times = []
    for key in keys:
        t_sum = 0
        for _ in range(iters):
            start = time.perf_counter_ns()
            table.insert(key)
            t_sum += (time.perf_counter_ns() - start)
            table.delete(key)
        
        insert_times.append(t_sum / float(iters))
        table.insert(key)
        
    search_times = []
    table.reset_counters()
    for key in keys:
        t_sum = 0
        for _ in range(iters):
            start = time.perf_counter_ns()
            table.search(key)
            t_sum += (time.perf_counter_ns() - start)
        search_times.append(t_sum / float(iters))
        
    return insert_times, search_times


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
                all_trials_keys = []
                for trial in range(config.TRIALS):
                    # Derive per-trial seed
                    seed = config.BASE_SEED + trial
                    
                    keys = _generate_keys(dataset_name, n, m, seed)
                    all_trials_keys.append(keys)
                    
                    for iters in [10, 20, 30, 40, 50]:
                        # 1. Fixed Hash
                        fixed_table = HashTable(size=m, hash_function=make_fixed_hash(m))
                        fixed_insert_times, fixed_search_times = benchmark_table(fixed_table, keys, iters)
                        fixed_stats = compute_chain_stats(fixed_table)
                        
                        results.append(ExperimentResult(
                            n=n, 
                            load_factor=alpha, 
                            m=m, 
                            dataset_name=dataset_name,
                            hash_strategy="fixed", 
                            trial_index=trial,
                            iterations=iters,
                            insert_times=fixed_insert_times, 
                            search_times=fixed_search_times,
                            stats=fixed_stats
                        ))
                        
                        # 2. Universal Hash
                        rng = random.Random(seed)
                        univ_hash = UniversalHash(m=m, rng=rng)
                        univ_table = HashTable(size=m, hash_function=univ_hash)
                        univ_insert_times, univ_search_times = benchmark_table(univ_table, keys, iters)
                        univ_stats = compute_chain_stats(univ_table)
                        
                        results.append(ExperimentResult(
                            n=n, 
                            load_factor=alpha, 
                            m=m, 
                            dataset_name=dataset_name,
                            hash_strategy="universal", 
                            trial_index=trial,
                            iterations=iters,
                            insert_times=univ_insert_times, 
                            search_times=univ_search_times,
                            stats=univ_stats
                        ))
                    
                # Output combined dataset for this parameter combination
                dataset_dir = Path("results/datasets")
                dataset_dir.mkdir(parents=True, exist_ok=True)
                dataset_file = dataset_dir / f"{dataset_name}_n{n}_m{m}.csv"
                with open(dataset_file, "w") as f:
                    headers = ",".join(f"trial_{t}" for t in range(config.TRIALS))
                    f.write(headers + "\n")
                    for row in zip(*all_trials_keys):
                        f.write(",".join(map(str, row)) + "\n")
                    
    return results


def run_chain_length_experiments() -> list[ChainGrowthResult]:
    """Runs chain length growth tracking over 1..N."""
    results: list[ChainGrowthResult] = []
    dataset_names = ["uniform", "sorted", "adversarial", "near_duplicate"]
    
    for n in config.N_VALUES_CHAIN:
        for alpha in config.LOAD_FACTORS:
            m = config.table_size_for_load_factor(n, alpha)
            for dataset_name in dataset_names:
                for trial in range(config.TRIALS):
                    seed = config.BASE_SEED + trial
                    keys = _generate_keys(dataset_name, n, m, seed)
                    
                    # 1. Fixed Hash
                    fixed_table = HashTable(size=m, hash_function=make_fixed_hash(m))
                    fixed_growth = []
                    fixed_collision_growth = []
                    current_max = 0
                    current_collisions = 0
                    seen_fixed = set()
                    for key in keys:
                        if key not in seen_fixed:
                            seen_fixed.add(key)
                            idx = fixed_table._bucket_index(key)
                            if len(fixed_table.buckets[idx]) > 0:
                                current_collisions += 1
                            fixed_table.buckets[idx].append(key)
                            current_max = max(current_max, len(fixed_table.buckets[idx]))
                        fixed_growth.append(current_max)
                        fixed_collision_growth.append(current_collisions)
                        
                    results.append(ChainGrowthResult(
                        n=n, load_factor=alpha, m=m, 
                        dataset_name=dataset_name, hash_strategy="fixed",
                        trial_index=trial, chain_growth=fixed_growth,
                        collision_growth=fixed_collision_growth
                    ))
                    
                    # 2. Universal Hash
                    rng = random.Random(seed)
                    univ_hash = UniversalHash(m=m, rng=rng)
                    univ_table = HashTable(size=m, hash_function=univ_hash)
                    univ_growth = []
                    univ_collision_growth = []
                    current_max = 0
                    current_collisions = 0
                    seen_univ = set()
                    for key in keys:
                        if key not in seen_univ:
                            seen_univ.add(key)
                            idx = univ_table._bucket_index(key)
                            if len(univ_table.buckets[idx]) > 0:
                                current_collisions += 1
                            univ_table.buckets[idx].append(key)
                            current_max = max(current_max, len(univ_table.buckets[idx]))
                        univ_growth.append(current_max)
                        univ_collision_growth.append(current_collisions)
                        
                    results.append(ChainGrowthResult(
                        n=n, load_factor=alpha, m=m, 
                        dataset_name=dataset_name, hash_strategy="universal",
                        trial_index=trial, chain_growth=univ_growth,
                        collision_growth=univ_collision_growth
                    ))
                    
    return results


if __name__ == "__main__":
    print("Running experiments...")
    results = run_all_experiments()
    print(f"Generated {len(results)} experiment results.")
