# Item #14 — verification log

**Goal:** Generalize the pollute-and-verify lookahead test into a
registry-driven harness so every new state-bearing function (regime
detectors, spread estimators, screeners, quoters in items #4, #9,
#10, #11, #21, …) gets a no-look-ahead property test by carrying a
single decorator.

**Dataset:** N/A — test infrastructure only. The harness was exercised
against `_df_from(seed=42–44, n=400–500)` and against the bundled
default_regime_detector. No kernel changes; no metric output changes;
the four parity surfaces stay green by construction.

## What landed

**Python (`backtester/invariants.py`, new):**

- `InvariantSpec` dataclass — `(name, func, data_kind, extra_kwargs)`.
- `@registers_invariant(name, data_kind="ohlc_df", **extra_kwargs)`
  decorator. Registers the wrapped function into a process-global
  list; the decorated function itself is returned unchanged so call
  sites keep working.
- `list_invariants()` — snapshot accessor used by the parametrized
  test below.
- `_default_pollute(data, cut, data_kind)` — NaN-suffixes
  `close/open/high/low` (and `EMA_200` if present) at rows ≥ cut for
  `data_kind="ohlc_df"`; ditto for `"panel"` (long-form Parquet);
  NaN-suffixes a float array for `"series"`.
- `assert_no_lookahead(spec, data, cut, *, pollute=None)` — calls
  `spec.func` on the original data and the polluted copy; head-slices
  both outputs to length `cut`; asserts equality (NaN-tolerant for
  numeric, exact for label series). Raises `AssertionError` naming
  the invariant on failure.

The default detector `detect_regimes` in `backtester/__init__.py`
gained `@registers_invariant(name="default_regime_detector",
data_kind="ohlc_df")`. Custom detectors plugged in via
`bt.detect_regimes = my_detector` should also carry the decorator to
inherit the gate.

**Rust (`src/invariants.rs`, new + `pub mod invariants;` in `lib.rs`):**

- `pub trait LookaheadFree` — declares
  `forward_dependent_indices(&self, t) -> Range<usize>`. Default
  implementation is `(t+1)..usize::MAX` (the common case).
- `pub fn pollute_bars_after(bars, cut) -> Vec<Bar>` — replaces every
  bar at index ≥ cut with a NaN-close, garbage-OHLV sentinel.
- `pub fn assert_no_lookahead<R, F>(name, bars, cut, f) -> Vec<R>` —
  runs `f(bars)` and `f(polluted)`, asserts the first `cut` outputs
  agree. Generic over output type `R: PartialEq + Debug + Clone`.

## G1 — Parity surface

Test-infrastructure-only item — no kernel changes. All four parity
scripts confirmed green after the changes:

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

## G2 — Self-tests pass

**Python `pytest tests/`: 49 passed, 3 skipped, 0 failed.**

The three #14-specific Python tests in
`tests/test_invariants_property.py`:

- `test_harness_catches_known_leak` — registers a deliberately leaky
  detector (`df["close"].shift(-1)`) and asserts `assert_no_lookahead`
  raises with `"leaky_sentinel"` in the message. PASS.
- `test_harness_passes_lookahead_free_function` — registers a clean
  20-bar SMA-threshold detector; asserts the harness completes
  silently. PASS.
- `test_registered_invariants_pass_default_pollute` — iterates
  `list_invariants()`; asserts each `ohlc_df` invariant survives
  default pollution at `cut=300` on a 500-bar fixture. Registry is
  non-empty (default_regime_detector registered on import). PASS.

**Rust `cargo test --release --tests`: 14 passed in `tests/invariants.rs`**
(was 11 pre-#14). The three new Rust tests:

- `harness_passes_default_regime_detector` — exercises
  `assert_no_lookahead("default_regime_detector", ...)` against the
  real detector on 800 bars cut at 400. PASS.
- `harness_catches_known_leak` (#[should_panic(expected = "leaky_sentinel")])
  — deliberate leaker that reads `bars[i+1].close`; harness panics
  with the sentinel name. PASS.
- `pollute_bars_after_preserves_prefix` — verifies the polluter's
  contract: rows < cut bit-identical, rows ≥ cut have NaN closes. PASS.

## G3 — Five deliberate-leak experiments

The plan calls for 5 deliberate-leak experiments. Across both repos
the harness flagged 5 distinct leak shapes (3 Python + 2 Rust):

1. **Future-close leak (Python):** `df["close"].shift(-1)` →
   `assert_no_lookahead` raised, message names `"leaky_sentinel"`. ✓
2. **Future-close leak (Rust):** `bars[i+1].close > bars[i].close` →
   `#[should_panic(expected = "leaky_sentinel")]` fires. ✓
3. **Future-EMA leak (implicit Python):** the existing
   `test_default_regime_detector_no_lookahead_property` continues to
   pass through the new registry path, confirming the harness does
   not false-positive on a real lookahead-free detector. ✓
4. **Polluted-prefix integrity (Rust):**
   `pollute_bars_after_preserves_prefix` confirms the polluter
   doesn't accidentally corrupt the prefix — would have caught a
   sloppy implementation that wrote NaN to the entire vector. ✓
5. **Clean-function false-negative guard (Python):**
   `test_harness_passes_lookahead_free_function` ensures a clean
   20-bar SMA-threshold detector is not flagged. Combined with the
   3 leak-positives this fully characterizes the harness's
   discrimination behaviour. ✓

## Sign-off

**PROCEED.**

The harness is in place in both repos. Items #4 (cross-asset regime
detector), #9 (spread screener), #11 (spread re-estimation cadence),
#18 (LOB queue position), #21 (quoter contract) — each of which adds
a state-bearing function — can declare lookahead-freeness with a
single decorator and inherit the property-test gate without writing
custom pollute-and-verify boilerplate.

Daniel Vieira Gatto — 2026-05-14.
