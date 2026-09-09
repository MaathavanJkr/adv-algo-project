"""Main script to run experiments, compute statistics, and generate plots."""

from __future__ import annotations

import time
from src.experiment import run_all_experiments
from src.statistics import aggregate_and_save
from src.visualization import generate_all_plots

def main() -> None:
    print("Starting experiments...")
    start_time = time.perf_counter()
    
    # 1. Run experiments
    results = run_all_experiments()
    print(f"Completed {len(results)} trials in {time.perf_counter() - start_time:.2f} seconds.")
    
    # 2. Aggregate statistics and save to CSV
    print("Aggregating statistics...")
    summary_df = aggregate_and_save(results, output_dir="results")
    
    # 3. Generate visualizations
    print("Generating plots...")
    try:
        generate_all_plots(summary_df, output_dir="plots")
        print("All tasks completed successfully! Check the 'results' and 'plots' directories.")
    except ImportError as e:
        print(f"Skipping plot generation: {e}. Please install matplotlib.")
        print("Experiment datasets and statistics were generated successfully in 'results/'.")

if __name__ == "__main__":
    main()
