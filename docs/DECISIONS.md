# Design Decisions

Notes on the places where more than one reasonable implementation
choice existed, and which one I went with. Keeping this around so the
reasoning isn't lost by the time the final report gets written.

- **Duplicate policy**: the table is treated as a *set* of keys.
  Re-inserting an existing key is a no-op (not an error, not a second
  entry). Applied identically for both hash strategies so comparisons
  are apples-to-apples. See `HashTable.insert`.

- **collision_count definition**: for each bucket, `max(len(bucket) -
  1, 0)`, summed across buckets. I.e. the first key placed in a
  bucket is "free"; every key placed after it in the same bucket
  counts as one collision. This is the standard chaining-collision
  definition and is easy to hand-verify on a small table (see
  `tests/test_metrics.py`).

- **Search comparison counting**: a comparison is charged for every
  element inspected while scanning a bucket, including the matching
  element itself. Finding the k-th element costs k comparisons; a
  full miss costs `len(bucket)` comparisons. This matches the
  standard analysis of expected search cost in chaining (`O(1 +
  alpha)`).

- **Universal hash prime `p`**: defaults to the fixed Mersenne prime
  `2**61 - 1`, validated with a deterministic Miller-Rabin test. This
  is comfortably larger than any key range the datasets here produce
  (keys up to roughly `10 * n`, n in the tens of thousands), so there's
  no need to compute a fresh prime per key range. A caller can still
  pass a different `p` explicitly; it gets validated (primality,
  `p > m`) either way.

- **Universal hash coefficients (`a`, `b`)**: drawn once at instance
  construction from an injected `random.Random` (or `seed`), never
  redrawn per key or per call. This is what the Carter-Wegman
  guarantee assumes; the tests check that repeated calls on one
  instance never change `(a, b)`, while separate instances can differ.
  `UniversalHash` also accepts explicit `a`/`b` (validated) so tests
  can pin coefficients or exercise the `a == 0` invalid case directly,
  since `a` is never drawn as 0 in the normal random path.

- **Hash strategy interface**: both strategies are injected into
  `HashTable` as a single-argument callable `f(key) -> int`.
  `fixed_hash(key, m)` takes `m` explicitly, so `make_fixed_hash(m)`
  adapts it via `functools.partial` to the same single-argument shape
  a `UniversalHash` instance already has (it's callable as `h(key)`).
  This keeps `HashTable` itself completely hash-strategy-agnostic.

- **Adversarial dataset requires `m`**: `adversarial_dataset(n, m)`
  takes the table size as a parameter (unlike the other three
  generators) because the construction `k_i = i * m` is only
  adversarial *relative to* a specific modulus. This is called out in
  its docstring.

- **`near_duplicate_dataset` pool size**: `max(1, n // 10)`, so `n=1`
  through `n=9` all get a pool of size 1 rather than 0.

- **Dev environment**: the system default `python3` was 3.9.6, but this
  project targets 3.11+. Used a Homebrew-installed `python3.13` to
  create the project's `.venv` instead.
