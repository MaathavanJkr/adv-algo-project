"""Tests for src.experiment: single-trial fairness and result shape.

Kept small (small n, few trials) so the suite stays fast -- full-scale
runs are exercised via run_experiments.py, not pytest.
"""

from __future__ import annotations

import json

from src.experiment import TrialResult, run_all_experiments, run_single_trial


class TestRunSingleTrial:
    def test_returns_one_fixed_and_one_universal_row(self) -> None:
        rows = run_single_trial("uniform", n=100, alpha=1.0, m=100, k=3, trial=0, base_seed=1)
        assert len(rows) == 2
        methods = {row.method for row in rows}
        assert methods == {"fixed", "universal"}

    def test_num_hash_functions_matches_method(self) -> None:
        rows = run_single_trial("uniform", n=100, alpha=1.0, m=100, k=4, trial=0, base_seed=1)
        by_method = {row.method: row for row in rows}
        assert by_method["fixed"].num_hash_functions == 1
        assert by_method["universal"].num_hash_functions == 4

    def test_fixed_row_has_no_universal_params(self) -> None:
        rows = run_single_trial("uniform", n=50, alpha=1.0, m=50, k=3, trial=0, base_seed=1)
        by_method = {row.method: row for row in rows}
        assert by_method["fixed"].p is None
        assert by_method["fixed"].coefficients is None

    def test_universal_row_records_p_and_coefficients_for_every_member(self) -> None:
        k = 5
        rows = run_single_trial("uniform", n=50, alpha=1.0, m=50, k=k, trial=0, base_seed=1)
        by_method = {row.method: row for row in rows}
        universal = by_method["universal"]
        assert universal.p is not None
        coeffs = json.loads(universal.coefficients)
        assert len(coeffs) == k
        assert all(len(pair) == 2 for pair in coeffs)

    def test_same_dataset_used_by_both_methods_via_unique_keys(self) -> None:
        # sorted_dataset has no duplicates, so both tables should end
        # up with exactly n unique keys inserted from the same n keys.
        n = 200
        rows = run_single_trial("sorted", n=n, alpha=1.0, m=n, k=3, trial=0, base_seed=1)
        for row in rows:
            assert row.unique_keys == n

    def test_search_sample_size_is_n_over_2(self) -> None:
        n = 137  # odd, so n // 2 is the relevant floor division
        rows = run_single_trial("uniform", n=n, alpha=1.0, m=n, k=3, trial=0, base_seed=1)
        for row in rows:
            expected = n // 2
            assert row.search_comparisons_total >= 0
            # search_time_per_op is search_time_total / (n // 2); back
            # out the sample size indirectly isn't reliable from time,
            # so just confirm the average is total / expected sample.
            if expected:
                assert row.search_comparisons_avg == row.search_comparisons_total / expected

    def test_universal_coefficients_constant_within_trial_vary_across_trials(self) -> None:
        rows_t0 = run_single_trial("uniform", n=50, alpha=1.0, m=50, k=3, trial=0, base_seed=1)
        rows_t1 = run_single_trial("uniform", n=50, alpha=1.0, m=50, k=3, trial=1, base_seed=1)
        universal_t0 = next(r for r in rows_t0 if r.method == "universal")
        universal_t1 = next(r for r in rows_t1 if r.method == "universal")
        assert universal_t0.coefficients != universal_t1.coefficients

    def test_reproducible_given_same_seed(self) -> None:
        # Everything except wall-clock timing must match exactly given
        # the same seed -- timing legitimately varies run to run.
        timing_fields = {
            "insert_time_total",
            "insert_time_per_op",
            "search_time_total",
            "search_time_per_op",
        }
        rows_a = run_single_trial("uniform", n=50, alpha=1.0, m=50, k=3, trial=2, base_seed=7)
        rows_b = run_single_trial("uniform", n=50, alpha=1.0, m=50, k=3, trial=2, base_seed=7)
        for row_a, row_b in zip(rows_a, rows_b):
            dict_a = {k: v for k, v in row_a.__dict__.items() if k not in timing_fields}
            dict_b = {k: v for k, v in row_b.__dict__.items() if k not in timing_fields}
            assert dict_a == dict_b

    def test_adversarial_fixed_max_chain_beats_universal(self) -> None:
        n, m = 300, 30
        rows = run_single_trial("adversarial", n=n, alpha=n / m, m=m, k=3, trial=0, base_seed=1)
        by_method = {row.method: row for row in rows}
        assert by_method["fixed"].max_chain_length == n
        assert by_method["universal"].max_chain_length < by_method["fixed"].max_chain_length


class TestRunAllExperiments:
    def test_row_count_matches_grid_size(self) -> None:
        results = run_all_experiments(
            n_values=[50],
            trials=2,
            distributions=["uniform", "sorted"],
            load_factors=[1.0],
            k_values=[2, 3],
            base_seed=1,
            progress=False,
        )
        # 1 n * 1 load_factor * 2 distributions * 2 k values * 2 trials
        # * 2 rows (fixed + universal) per trial.
        assert len(results) == 1 * 1 * 2 * 2 * 2 * 2
        assert all(isinstance(r, TrialResult) for r in results)

    def test_fixed_rows_always_have_num_hash_functions_one(self) -> None:
        results = run_all_experiments(
            n_values=[40],
            trials=1,
            distributions=["uniform"],
            load_factors=[1.0],
            k_values=[2, 5],
            base_seed=1,
            progress=False,
        )
        fixed_rows = [r for r in results if r.method == "fixed"]
        assert fixed_rows
        assert all(r.num_hash_functions == 1 for r in fixed_rows)
