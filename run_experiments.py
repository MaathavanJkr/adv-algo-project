#!/usr/bin/env python3
"""CLI: run the fixed-vs-universal hashing experiments end-to-end.

Writes results/raw_results.csv, results/summary_statistics.csv, and
results/experiment_summary.txt.
"""

from __future__ import annotations

import argparse
import os

import config
from src.datasets import adversarial_dataset
from src.hash_functions import make_universal_family
from src.hash_table import HashTable
from src.experiment import run_all_experiments
from src.metrics import validate_adversarial_concentration
from src.statistics import results_to_dataframe, summarize

RESULTS_DIR = "results"

# n, m for the pre-flight adversarial sanity check -- large enough
# that "universal happens to also concentrate everything in one
# bucket" is not a realistic outcome, small enough to run instantly.
_SANITY_N = 1000
_SANITY_M = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-hash vs universal-hash experiments."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny end-to-end config to verify the whole pipeline fast.",
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=None,
        choices=config.DISTRIBUTIONS,
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        dest="k_values",
        help="Universal hash family sizes to test.",
    )
    parser.add_argument("--load-factors", type=float, nargs="+", default=None)
    return parser.parse_args()


def run_adversarial_sanity_check(k_values: list[int], base_seed: int) -> None:
    """Fixed hashing must concentrate k_i = i*m into bucket 0 on the
    adversarial dataset; universal hashing (independent of that
    construction) should not. Fails loudly if either doesn't hold."""
    print("[sanity check] adversarial dataset concentration...")

    fixed_max = validate_adversarial_concentration(n=_SANITY_N, m=_SANITY_M)
    assert fixed_max == _SANITY_N, (
        f"expected fixed hash to concentrate all {_SANITY_N} keys into "
        f"bucket 0, got max_chain_length={fixed_max}"
    )

    keys = adversarial_dataset(_SANITY_N, _SANITY_M)
    k = max(k_values)
    family = make_universal_family(k=k, m=_SANITY_M, seed=base_seed)
    universal_table = HashTable(size=_SANITY_M, hash_family=family, seed=base_seed + 1)
    for key in keys:
        universal_table.insert(key)
    universal_max = universal_table.max_chain_length()

    print(
        f"[sanity check] fixed max_chain_length={fixed_max}  "
        f"universal(k={k}) max_chain_length={universal_max}"
    )
    assert universal_max < fixed_max, (
        "expected universal hashing to avoid concentrating the adversarial "
        f"key set the way fixed hashing does (fixed={fixed_max}, "
        f"universal={universal_max})"
    )
    print("[sanity check] passed\n")


def write_experiment_summary(df, summary, path: str) -> None:
    """Writes concise, data-driven findings from the measured results."""
    lines = ["Experiment Summary", "=" * 60, f"Total trial rows: {len(df)}", ""]

    adversarial = summary[summary["distribution"] == "adversarial"]
    if not adversarial.empty:
        lines.append("Adversarial set -- fixed vs universal max chain length:")
        cols = ["n", "m", "method", "num_hash_functions", "max_chain_length_mean"]
        for _, row in adversarial.sort_values(["n", "method", "num_hash_functions"]).iterrows():
            lines.append(
                f"  n={int(row['n'])} m={int(row['m'])} method={row['method']} "
                f"k={int(row['num_hash_functions'])}: "
                f"mean max_chain_length={row['max_chain_length_mean']:.2f}"
            )
        lines.append("")

    universal = summary[summary["method"] == "universal"]
    if not universal.empty:
        lines.append("Search comparisons vs universal family size k:")
        by_k = universal.groupby("num_hash_functions")["search_comparisons_avg_mean"].mean()
        for k, value in by_k.sort_index().items():
            lines.append(f"  k={int(k)}: mean avg search comparisons = {value:.3f}")
        lines.append("")

    fixed = summary[summary["method"] == "fixed"]
    if not fixed.empty and not universal.empty:
        lines.append("Fixed vs universal, overall (mean across all measured configs):")
        lines.append(
            f"  fixed:     max_chain_length={fixed['max_chain_length_mean'].mean():.3f}  "
            f"search_comparisons_avg={fixed['search_comparisons_avg_mean'].mean():.3f}"
        )
        lines.append(
            f"  universal: max_chain_length={universal['max_chain_length_mean'].mean():.3f}  "
            f"search_comparisons_avg={universal['search_comparisons_avg_mean'].mean():.3f}"
        )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    if args.quick:
        n_values = args.n_values or [200]
        trials = args.trials or 3
        distributions = args.distributions or ["uniform", "adversarial"]
        k_values = args.k_values or [3]
        load_factors = args.load_factors or [1.0]
    else:
        n_values = args.n_values or config.N_VALUES
        trials = args.trials or config.TRIALS
        distributions = args.distributions or config.DISTRIBUTIONS
        k_values = args.k_values or config.UNIVERSAL_K_VALUES
        load_factors = args.load_factors or config.LOAD_FACTORS

    base_seed = args.seed if args.seed is not None else config.BASE_SEED

    run_adversarial_sanity_check(k_values, base_seed)

    print(
        f"Running experiments: n={n_values} trials={trials} "
        f"distributions={distributions} k={k_values} "
        f"load_factors={load_factors} seed={base_seed}"
    )
    results = run_all_experiments(
        n_values=n_values,
        trials=trials,
        distributions=distributions,
        load_factors=load_factors,
        k_values=k_values,
        base_seed=base_seed,
    )
    print(f"Collected {len(results)} trial rows.")

    df = results_to_dataframe(results)
    summary = summarize(df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(RESULTS_DIR, "raw_results.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary_statistics.csv")
    summary_txt_path = os.path.join(RESULTS_DIR, "experiment_summary.txt")

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_experiment_summary(df, summary, summary_txt_path)

    print(f"Wrote {raw_path} ({len(df)} rows)")
    print(f"Wrote {summary_path} ({len(summary)} rows)")
    print(f"Wrote {summary_txt_path}")


if __name__ == "__main__":
    main()
