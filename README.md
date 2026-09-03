# Fixed Hash Function vs. Universal Hashing

Advanced Algorithms project comparing a deterministic fixed hash
(`h(k) = k mod m`) against a randomized Carter-Wegman universal hash,
on the *same* separate chaining hash table implementation, across
four input distributions.

**Research question:** does randomizing the hash function's
coefficients per table instance protect against the collision
patterns that a fixed modulus hash is structurally vulnerable to,
without materially hurting performance on "well behaved" inputs?

## Status

The hash table, both hash strategies, the dataset generators, the
metrics module, and the full test suite are implemented and passing.
The experiment runner and results/plots are not built yet:

- `src/experiment.py` and `src/statistics.py` are stubs — no
  experiments have been run, so there are no results in `results/`
  yet.
- `src/visualization.py` and an `analyze_results.py` script are
  planned but not started, so `plots/` is currently empty.

## Fixed hashing

`h(k) = k mod m` (`src/hash_functions.py::fixed_hash`). This is the
deterministic baseline: the same key always maps to the same bucket,
and the mapping is fully determined by `m`. That determinism is
exactly what makes it exploitable — anyone who knows `m` can construct
a set of keys that all collide (see [Dataset descriptions](#dataset-descriptions),
adversarial set).

## Universal hashing

Carter-Wegman universal hashing (`src/hash_functions.py::UniversalHash`):

```
h(k) = ((a*k + b) mod p) mod m
```

- `p` is prime and larger than the key range in use (defaults to the
  fixed Mersenne prime `2**61 - 1`, validated with a deterministic
  Miller Rabin test).
- `a` is drawn uniformly from `{1, ..., p-1}` (`a != 0`, since `a = 0`
  would collapse `h` to a constant function of `b` alone).
- `b` is drawn uniformly from `{0, ..., p-1}`.
- **Coefficients are drawn once, at instance construction — never per
  key.** A `UniversalHash` instance accepts an injected `random.Random`
  or `seed` so trials can be reproducible while still letting
  different instances draw independent `(a, b)` pairs.

Universal hashing gives a **probabilistic** guarantee, not a
zero-collision guarantee: for any two fixed, distinct keys, the
probability (over the random choice of `a, b` at construction time)
that they collide is at most `1/m`. A specific instance can still be
unlucky for a specific input; the guarantee is about the *expected*
behaviour over the random choice of the function, not about any one
draw being collision-free.

## Hash table architecture

`src/hash_table.py::HashTable` uses **separate chaining**: `m`
buckets, each a plain Python `list`. The hash strategy is
**dependency-injected** — `HashTable(size=m, hash_function=...)` — so
the exact same data structure and code path is used for both fixed
and universal hashing; only the injected callable differs. Both
strategies are exposed as a single-argument callable `f(key) -> int`
(`make_fixed_hash(m)` adapts `fixed_hash(key, m)` to that shape;
`UniversalHash` instances are already callable as `h(key)`).

**Duplicate policy:** the table behaves like a *set* of keys.
Re-inserting a key already present in its bucket is a no-op — it does
not create a second entry and does not raise. This policy is applied
identically regardless of which hash strategy is injected, so
duplicate handling never differs between the two methods being
compared. See [Design decisions](#design-decisions).

**Search instrumentation:** `search()` counts every key comparison
made while scanning the target bucket. Finding the k-th element in a
chain costs `k` comparisons; a full miss costs `len(bucket)`
comparisons. The running total is `comparison_count`, resettable via
`reset_counters()`.

## Dataset descriptions

All four generators live in `src/datasets.py`, are seedable where
randomness is involved, and run in O(n) time.

1. **Uniform** (`uniform_dataset`): `n` random integers drawn from
   `[1, 10n]`. The "generic" workload — no structure for either hash
   to exploit or be defeated by.
2. **Sorted** (`sorted_dataset`): exactly `1, 2, ..., n`. Tests
   behaviour on already-ordered, densely-packed key sequences.
3. **Adversarial** (`adversarial_dataset`, requires `m`): `k_i = i *
   m` for `i = 1..n`. Under the fixed hash, `i*m mod m == 0` for
   every `i`, so **every key maps to bucket 0** — this is the
   textbook worst case for `h(k) = k mod m`, and it is constructible
   by anyone who knows `m`. A universal hash's `(a, b)` are chosen
   independently of this construction, so it is very unlikely to
   concentrate these same keys into one bucket.
4. **Near-duplicate** (`near_duplicate_dataset`): `n` keys drawn (with
   heavy repetition) from a pool of size `max(1, n // 10)`, so each
   pool value repeats roughly 10 times on average. Still yields
   exactly `n` keys. Exercises the set-based duplicate policy under
   load.

## Metrics

`src/metrics.py::compute_chain_stats` computes, from a populated
`HashTable`:

- `max_chain_length` — longest chain.
- `average_chain_length` — mean chain length across all buckets
  (equals `unique_keys / size`, the achieved load factor).
- `non_empty_buckets` / `empty_buckets`.
- `chain_length_stdev` — population standard deviation of chain
  lengths.
- `collision_count` — inserted unique keys beyond the first in each
  bucket, summed over all buckets (a bucket with `k` unique keys
  contributes `max(k - 1, 0)`).

`src/metrics.py::validate_adversarial_concentration(n, m)` is a
sanity-check helper: it builds a fixed-hash table over the
adversarial dataset, asserts every unique key landed in bucket 0, and
returns the measured max chain length, so it can be reused as a
regression check once experiments are running.

Timing and cross-trial statistical aggregation aren't in this module —
that's what `src/experiment.py` and `src/statistics.py` are for.

## Complexity note

Under the standard "simple uniform hashing" assumption (each key
equally likely to land in any bucket, independent of other keys),
separate chaining gives expected `O(1 + alpha)` insert/search, where
`alpha = n/m` is the load factor. The worst case is `O(n)`, reached
when every key lands in the same bucket, degenerating search into a
linear scan of a single chain.

The adversarial dataset triggers exactly that worst case **for fixed
hashing**, because `k_i = i*m` is constructed specifically so that
`k_i mod m == 0` for every key — the simple uniform hashing
assumption fails completely, since the mapping is deterministic and
known. Universal hashing makes this worst case unlikely (not
impossible) for the *same* key set: its coefficients `(a, b)` are
chosen independently of how the keys were constructed, so the
Carter-Wegman guarantee (collision probability `<= 1/m` per pair,
in expectation over the random choice of the function) still applies
to this input.

## Installation

- Python 3.11+ (developed and tested against 3.13). All algorithm
  code uses the standard library only — `random`, `statistics`,
  `math`.
- `pytest` for running the test suite (the only non-stdlib
  dependency, and only needed to run tests).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pytest
```

## How to run the tests

```bash
pytest -v
```

(Or, from outside a `source`-activated venv: `.venv/bin/pytest -v`.)

All tests currently pass (67 tests across `tests/test_hash_functions.py`,
`tests/test_hash_table.py`, `tests/test_datasets.py`, and
`tests/test_metrics.py`).

## Reproducibility

`config.py` defines `BASE_SEED = 42` as the project-wide seeding
convention, intended to derive per-trial seeds (e.g.
`BASE_SEED + trial_index`) once the experiment runner exists.

- `uniform_dataset(n, seed=...)` and `near_duplicate_dataset(n,
  seed=...)` accept an explicit seed and are fully deterministic for
  a given seed.
- `sorted_dataset(n)` and `adversarial_dataset(n, m)` are inherently
  deterministic (no randomness involved).
- `UniversalHash(m, ..., seed=...)` (or an injected `rng`) makes the
  one-time `(a, b)` draw reproducible, while still allowing separate
  instances to draw independently by using separate seeds or a
  shared, sequentially-advancing `random.Random`.

## Design decisions

Places where more than one reasonable implementation choice existed,
and which one I went with (full list with reasoning in
`DECISIONS.md`):

- **Duplicate policy is set-based**: re-inserting an existing key is a
  no-op, applied identically for both hash strategies.
- **`collision_count` = unique keys beyond the first, per bucket,
  summed**: the standard chaining-collision definition, hand-verified
  in `tests/test_metrics.py`.
- **Search comparisons include the matching element**: a hit on the
  k-th element costs `k` comparisons, matching the standard `O(1 +
  alpha)` analysis.
- **Universal hash `p` defaults to `2**61 - 1`**, validated via
  deterministic Miller-Rabin, comfortably larger than any key range
  the datasets here produce.
- **`UniversalHash` accepts optional explicit `a`/`b`** (validated) so
  invalid-parameter cases like `a = 0` can be exercised directly in
  tests, since normal random draws never produce `a = 0`.

## What's next

- `src/experiment.py`: run `config.TRIALS` trials per `(n,
  load_factor, dataset, hash_strategy)` combination, timing via
  `time.perf_counter`, reusing the same generated dataset across both
  hash strategies within a trial.
- `src/statistics.py`: aggregate those trials into summary tables in
  `results/*.csv`.
- `src/visualization.py` + `analyze_results.py`: turn the CSVs into
  comparison plots (`plots/*.png`) for collisions, chain lengths,
  search comparisons, and timings.
