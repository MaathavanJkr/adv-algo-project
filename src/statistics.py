"""Aggregates trial results into summary tables.

Named src.statistics, not statistics, so it doesn't shadow the stdlib
module elsewhere in the project. Pure aggregation only -- no timing,
no plotting.
"""

from __future__ import annotations

import pandas as pd

from src.experiment import TrialResult, results_as_dicts

# The configuration a set of trials shares. Grouping by these columns
# turns per-trial rows into per-configuration summary rows.
GROUP_COLUMNS = ["method", "distribution", "n", "m", "load_factor", "num_hash_functions"]

# Columns that are numeric but aren't trial metrics -- excluded from
# aggregation even though pandas would otherwise treat them as numbers
# to average.
_NON_METRIC_NUMERIC_COLUMNS = {"trial", "p"}


def results_to_dataframe(results: list[TrialResult] | list[dict]) -> pd.DataFrame:
    """Builds a DataFrame from raw trial rows. Kept out of the timing
    loops entirely -- this runs once, after all trials are collected."""
    if results and isinstance(results[0], TrialResult):
        rows = results_as_dicts(results)
    else:
        rows = results
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Groups by GROUP_COLUMNS and computes mean/median/std for every
    numeric trial metric."""
    numeric_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in GROUP_COLUMNS and column not in _NON_METRIC_NUMERIC_COLUMNS
    ]

    grouped = df.groupby(GROUP_COLUMNS, dropna=False)[numeric_columns].agg(
        ["mean", "median", "std"]
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    return grouped
