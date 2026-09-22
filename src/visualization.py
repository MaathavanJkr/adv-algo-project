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

def generate_raw_plots_by_n(results, datasets_to_plot=['sorted'], output_dir="plots") -> None:
    """Generates scatter plots for all data points for each N, using Matplotlib."""
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter by dataset and group by N
    grouped = defaultdict(list)
    for r in results:
        if r.dataset_name in datasets_to_plot:
            grouped[(r.dataset_name, r.n)].append(r)
            
    for (dataset, n), group_results in grouped.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Dataset: {dataset.title()} | n = {n} (All Data Points)")
        
        strategies = ['fixed', 'universal']
        colors = {'fixed': 'blue', 'universal': 'orange'}
        offsets = {'fixed': -0.02, 'universal': 0.02}
        
        for r in group_results:
            strat = r.hash_strategy
            lf = r.load_factor
            x_pos = lf + offsets.get(strat, 0)
            
            avg_search_time = sum(r.search_times) / len(r.search_times) if r.search_times else 0.0
            axes[0].scatter(x_pos, r.stats.collision_count, color=colors[strat], alpha=0.5, s=15)
            axes[1].scatter(x_pos, r.stats.max_chain_length, color=colors[strat], alpha=0.5, s=15)
            axes[2].scatter(x_pos, avg_search_time, color=colors[strat], alpha=0.5, s=15)
            
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='w', markerfacecolor=colors[s], marker='o', markersize=8) for s in strategies]
        
        axes[0].set_title('Collision Count')
        axes[0].set_xlabel('Load Factor')
        axes[0].set_ylabel('Collisions')
        axes[0].legend(custom_lines, strategies)
        
        axes[1].set_title('Max Chain Length')
        axes[1].set_xlabel('Load Factor')
        axes[1].set_ylabel('Max Chain Length')
        axes[1].legend(custom_lines, strategies)
        
        axes[2].set_title('Search Time')
        axes[2].set_xlabel('Load Factor')
        axes[2].set_ylabel('Time (s)')
        axes[2].legend(custom_lines, strategies)
        
        plt.tight_layout()
        plot_file = output_path / f"{dataset}_scatter_n{n}.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved raw plot: {plot_file}")

def generate_per_key_line_plots(results, datasets_to_plot=['sorted'], output_dir="plots") -> None:
    """Generates per-key insertion and search time line graphs for each N."""
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter by dataset and load factor 1.0 for simplicity
    grouped = defaultdict(list)
    for r in results:
        if r.dataset_name in datasets_to_plot and r.load_factor == 1.0:
            grouped[(r.dataset_name, r.n, r.iterations)].append(r)
            
    for (dataset, n, iterations), group_results in grouped.items():
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f"Dataset: {dataset.title()} | n = {n} | iters = {iterations} (Per-Key Timings, LF=1.0)")
        
        for r in group_results:
            strat = r.hash_strategy
            color = 'blue' if strat == 'fixed' else 'orange'
            
            x_vals = list(range(1, n + 1))
            # Just take the first n elements in case there's an off-by-one
            axes[0].plot(x_vals, r.insert_times[:n], color=color, alpha=0.7, label=strat)
            axes[1].plot(x_vals, r.search_times[:n], color=color, alpha=0.7, label=strat)
            
        axes[0].set_title('Insertion Time per Key')
        axes[0].set_xlabel('Key Index')
        axes[0].set_ylabel('Time (ns)')
        axes[0].grid(True, alpha=0.3)
        # Avoid duplicate legend entries
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[0].legend(unique.values(), unique.keys())
        
        axes[1].set_title('Search Time per Key')
        axes[1].set_xlabel('Key Index')
        axes[1].set_ylabel('Time (ns)')
        axes[1].grid(True, alpha=0.3)
        handles, labels = axes[1].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[1].legend(unique.values(), unique.keys())
        
        plt.tight_layout()
        plot_file = output_path / f"{dataset}_per_key_n{n}_iters{iterations}.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved per-key plot: {plot_file}")

def generate_chain_growth_plots(chain_results, datasets_to_plot=['sorted'], output_dir="plots") -> None:
    """Generates trajectory plots tracking max chain length across 1..N insertions."""
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter by dataset and group by (dataset, N)
    grouped = defaultdict(list)
    for r in chain_results:
        if r.dataset_name in datasets_to_plot and r.load_factor == 1.0:
            grouped[(r.dataset_name, r.n)].append(r)
            
    for (dataset, n), group_results in grouped.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Dataset: {dataset.title()} | n = {n} (Max Chain Growth, LF=1.0)")
        
        for r in group_results:
            strat = r.hash_strategy
            color = 'blue' if strat == 'fixed' else 'orange'
            
            x_vals = list(range(1, len(r.chain_growth) + 1))
            ax.plot(x_vals, r.chain_growth, color=color, alpha=0.7, label=strat)
            
        ax.set_title('Max Chain Length Growth')
        ax.set_xlabel('Insertion Step (1..N)')
        ax.set_ylabel('Max Chain Length')
        ax.grid(True, alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        
        plt.tight_layout()
        plot_file = output_path / f"{dataset}_chain_growth_n{n}.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved chain growth plot: {plot_file}")

def generate_collision_growth_plots(chain_results, datasets_to_plot=['sorted'], output_dir="plots") -> None:
    """Generates trajectory plots tracking total collisions across 1..N insertions."""
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    grouped = defaultdict(list)
    for r in chain_results:
        if r.dataset_name in datasets_to_plot and r.load_factor == 1.0:
            grouped[(r.dataset_name, r.n)].append(r)
            
    for (dataset, n), group_results in grouped.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f"Dataset: {dataset.title()} | n = {n} (Total Collisions, LF=1.0)")
        
        for r in group_results:
            strat = r.hash_strategy
            color = 'blue' if strat == 'fixed' else 'orange'
            
            x_vals = list(range(1, len(r.collision_growth) + 1))
            ax.plot(x_vals, r.collision_growth, color=color, alpha=0.7, label=strat)
            
        ax.set_title('Total Collisions Growth')
        ax.set_xlabel('Insertion Step (1..N)')
        ax.set_ylabel('Total Collisions')
        ax.grid(True, alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        
        plt.tight_layout()
        plot_file = output_path / f"{dataset}_collision_growth_n{n}.png"
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved collision growth plot: {plot_file}")
