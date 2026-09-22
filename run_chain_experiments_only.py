"""Run only the chain tracking experiments."""

import time
from src.experiment import run_chain_length_experiments
from src.visualization import generate_chain_growth_plots, generate_collision_growth_plots

def main() -> None:
    print("Running chain growth tracking experiments (1...N)...")
    chain_start_time = time.perf_counter()
    chain_results = run_chain_length_experiments()
    print(f"Completed chain growth trials in {time.perf_counter() - chain_start_time:.2f} seconds.")
    
    print("Generating chain growth and collision plots...")
    datasets = ['uniform', 'sorted', 'adversarial', 'near_duplicate']
    generate_chain_growth_plots(chain_results, datasets_to_plot=datasets, output_dir="plots")
    generate_collision_growth_plots(chain_results, datasets_to_plot=datasets, output_dir="plots")
    print("Chain growth and collision plots generated successfully in 'plots' directory.")

if __name__ == "__main__":
    main()
