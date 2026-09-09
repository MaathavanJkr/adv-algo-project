"""Plots for the aggregated results using matplotlib."""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

def generate_all_plots(aggregated_data: list[dict], output_dir: str | Path = "plots") -> None:
    """Generates comparison PNG plots from the aggregated data."""
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group by dataset
    datasets = defaultdict(list)
    for row in aggregated_data:
        if row["load_factor"] == 1.0:
            datasets[row["dataset_name"]].append(row)
            
    for dataset, rows in datasets.items():
        # group by hash strategy
        series_collisions = defaultdict(list)
        series_chains = defaultdict(list)
        series_search = defaultdict(list)
        
        for row in rows:
            strat = row["hash_strategy"]
            n = row["n"]
            series_collisions[strat].append((n, row["collision_count_mean"], row["collision_count_std"]))
            series_chains[strat].append((n, row["max_chain_length_mean"], row["max_chain_length_std"]))
            series_search[strat].append((n, row["search_time_mean"], row["search_time_std"]))
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Dataset: {dataset.title()} (Load Factor = 1.0)")
        
        for strategy in ['fixed', 'universal']:
            if strategy not in series_collisions:
                continue
                
            # Collisions
            col_data = sorted(series_collisions[strategy], key=lambda p: p[0])
            x_col = [p[0] for p in col_data]
            y_col = [p[1] for p in col_data]
            std_col = [p[2] for p in col_data]
            axes[0].plot(x_col, y_col, marker='o', label=strategy)
            axes[0].fill_between(x_col, 
                                 [y - s for y, s in zip(y_col, std_col)],
                                 [y + s for y, s in zip(y_col, std_col)], alpha=0.2)
                                 
            # Chains
            chain_data = sorted(series_chains[strategy], key=lambda p: p[0])
            x_ch = [p[0] for p in chain_data]
            y_ch = [p[1] for p in chain_data]
            std_ch = [p[2] for p in chain_data]
            axes[1].plot(x_ch, y_ch, marker='o', label=strategy)
            axes[1].fill_between(x_ch, 
                                 [y - s for y, s in zip(y_ch, std_ch)],
                                 [y + s for y, s in zip(y_ch, std_ch)], alpha=0.2)
                                 
            # Search Time
            search_data = sorted(series_search[strategy], key=lambda p: p[0])
            x_st = [p[0] for p in search_data]
            y_st = [p[1] for p in search_data]
            std_st = [p[2] for p in search_data]
            axes[2].plot(x_st, y_st, marker='o', label=strategy)
            axes[2].fill_between(x_st, 
                                 [y - s for y, s in zip(y_st, std_st)],
                                 [y + s for y, s in zip(y_st, std_st)], alpha=0.2)
                                 
        axes[0].set_title('Collision Count')
        axes[0].set_xlabel('n (number of keys)')
        axes[0].set_ylabel('Collisions')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_title('Max Chain Length')
        axes[1].set_xlabel('n (number of keys)')
        axes[1].set_ylabel('Max Chain Length')
        axes[1].legend()
        axes[1].grid(True)
        
        axes[2].set_title('Search Time')
        axes[2].set_xlabel('n (number of keys)')
        axes[2].set_ylabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plot_file = output_path / f"{dataset}_plots.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved plot: {plot_file}")
