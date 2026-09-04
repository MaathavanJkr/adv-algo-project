"""Tests for src.statistics: aggregation over raw trial rows."""

from __future__ import annotations

from src.experiment import run_all_experiments
from src.statistics import GROUP_COLUMNS, results_to_dataframe, summarize


class TestResultsToDataframe:
    def test_row_count_and_columns(self) -> None:
        results = run_all_experiments(
            n_values=[40],
            trials=2,
            distributions=["uniform"],
            load_factors=[1.0],
            k_values=[3],
            base_seed=1,
            progress=False,
        )
        df = results_to_dataframe(results)
        assert len(df) == len(results)
        assert "num_hash_functions" in df.columns
        assert "method" in df.columns


class TestSummarize:
    def test_grouped_rows_have_mean_median_std_columns(self) -> None:
        results = run_all_experiments(
            n_values=[40],
            trials=3,
            distributions=["uniform"],
            load_factors=[1.0],
            k_values=[3],
            base_seed=1,
            progress=False,
        )
        df = results_to_dataframe(results)
        summary = summarize(df)

        for col in GROUP_COLUMNS:
            assert col in summary.columns
        assert "max_chain_length_mean" in summary.columns
        assert "max_chain_length_median" in summary.columns
        assert "max_chain_length_std" in summary.columns

        # One group per (method, distribution, n, m, load_factor, k):
        # fixed and universal each get their own row.
        assert len(summary) == 2

    def test_trial_column_excluded_from_aggregation(self) -> None:
        results = run_all_experiments(
            n_values=[40],
            trials=3,
            distributions=["uniform"],
            load_factors=[1.0],
            k_values=[3],
            base_seed=1,
            progress=False,
        )
        df = results_to_dataframe(results)
        summary = summarize(df)
        assert not any(col.startswith("trial_") for col in summary.columns)

    def test_mean_matches_manual_average(self) -> None:
        results = run_all_experiments(
            n_values=[40],
            trials=5,
            distributions=["uniform"],
            load_factors=[1.0],
            k_values=[3],
            base_seed=1,
            progress=False,
        )
        df = results_to_dataframe(results)
        summary = summarize(df)

        fixed_rows = df[df["method"] == "fixed"]
        fixed_summary = summary[summary["method"] == "fixed"].iloc[0]
        expected_mean = fixed_rows["max_chain_length"].mean()
        assert fixed_summary["max_chain_length_mean"] == expected_mean
