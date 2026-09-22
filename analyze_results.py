"""Main script to run experiments, compute statistics, and generate plots."""

from __future__ import annotations

import time
from src.experiment import run_all_experiments, run_chain_length_experiments
from src.statistics import aggregate_and_save
from src.visualization import generate_all_plots, generate_raw_plots_by_n

def main() -> None:
    print("Starting experiments...")
    start_time = time.perf_counter()
    
    # 1. Run experiments
    results = run_all_experiments()
    print(f"Completed {len(results)} trials in {time.perf_counter() - start_time:.2f} seconds.")
    
    # 2. Aggregate statistics and save to CSV
    print("Aggregating statistics...")
    summary_df = aggregate_and_save(results, output_dir="results")
    
    # 3. Run chain growth experiments
    print("Running chain growth tracking experiments (1...N)...")
    chain_start_time = time.perf_counter()
    chain_results = run_chain_length_experiments()
    print(f"Completed chain growth trials in {time.perf_counter() - chain_start_time:.2f} seconds.")
    
    from src.visualization import generate_all_plots, generate_per_key_line_plots, generate_raw_plots_by_n, generate_chain_growth_plots
    
    print("Generating plots...")
    try:
        datasets = ['uniform', 'sorted', 'adversarial', 'near_duplicate']
        generate_per_key_line_plots(results, datasets_to_plot=datasets, output_dir="plots")
        generate_raw_plots_by_n(results, datasets_to_plot=datasets, output_dir="plots")
        generate_all_plots(summary_df, output_dir="plots")
        generate_chain_growth_plots(chain_results, datasets_to_plot=datasets, output_dir="plots")
        print("All tasks completed successfully! Check the 'results' and 'plots' directories.")
    except ImportError as e:
        print(f"Skipping plot generation: {e}. Please install matplotlib.")
        print("Experiment datasets and statistics were generated successfully in 'results/'.")

if __name__ == "__main__":
    main()
