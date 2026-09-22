"""Aggregates trial results into summary tables."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from collections import defaultdict
from src.experiment import ExperimentResult

def aggregate_and_save(results: list[ExperimentResult], output_dir: str | Path = "results") -> list[dict]:
    """Aggregates trial results and saves them to a CSV."""
    if not results:
        raise ValueError("No results provided.")
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group results by dataset_name, hash_strategy, n, load_factor, m
    grouped = defaultdict(list)
    for r in results:
        key = (r.dataset_name, r.hash_strategy, r.n, r.load_factor, r.m, r.iterations)
        grouped[key].append(r)
        
    aggregated = []
    for key, group in grouped.items():
        dataset_name, hash_strategy, n, load_factor, m, iterations = key
        
        insert_times = [statistics.mean(r.insert_times) for r in group]
        search_times = [statistics.mean(r.search_times) for r in group]
        max_chain_lengths = [r.stats.max_chain_length for r in group]
        collision_counts = [r.stats.collision_count for r in group]
        
        def safe_stdev(data):
            return statistics.stdev(data) if len(data) > 1 else 0.0
            
        row = {
            "dataset_name": dataset_name,
            "hash_strategy": hash_strategy,
            "n": n,
            "load_factor": load_factor,
            "m": m,
            "iterations": iterations,
            "insert_time_mean": statistics.mean(insert_times),
            "insert_time_std": safe_stdev(insert_times),
            "search_time_mean": statistics.mean(search_times),
            "search_time_std": safe_stdev(search_times),
            "max_chain_length_mean": statistics.mean(max_chain_lengths),
            "max_chain_length_std": safe_stdev(max_chain_lengths),
            "collision_count_mean": statistics.mean(collision_counts),
            "collision_count_std": safe_stdev(collision_counts),
        }
        aggregated.append(row)
        
    csv_path = output_path / "summary.csv"
    with open(csv_path, mode="w", newline="") as f:
        fieldnames = list(aggregated[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated:
            writer.writerow(row)
            
    print(f"Summary statistics saved to {csv_path}")
    return aggregated
