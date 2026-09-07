//! Lookahead-leak harness (Rust mirror).
//!
//! Pollutes the tail of a `&[Bar]` slice past a cut index and asserts that
//! a candidate function's output for indices `< cut` is unchanged. Used by
//! `tests/invariants.rs` to property-check `default_regime_detector` and,
//! in later items, by panel/pairs/MM detectors and quoters that claim
//! lookahead-freeness.
//!
//! The Python framework lives in `backtester/invariants.py`; the contract
//! mirrors it. Each Rust state-bearing primitive that lands in Phases 2–5
//! is expected to be exercised through this harness in its CI test.

use crate::Bar;

/// Trait marker for any value that declares which forward bar indices its
/// output at time `t` cannot depend on. The default implementation
/// (lookahead-free up to and including `t`) is the common case; pollute
/// indices `> t` and verify.
pub trait LookaheadFree {
    /// Half-open range of forward indices the implementation must NOT
    /// consult when producing the output for index `t`. The harness uses
    /// `forward_dependent_indices(t).start == t + 1` as the default
    /// pollution boundary.
    fn forward_dependent_indices(&self, t: usize) -> std::ops::Range<usize> {
        (t + 1)..usize::MAX
    }
}

/// Replace every `Bar` at index `>= cut` with a deterministic-but-
/// nonsense copy: prices reshuffled to a fixed garbage value. Caller
/// uses this to feed a polluted slice into a candidate function.
pub fn pollute_bars_after(bars: &[Bar], cut: usize) -> Vec<Bar> {
    let mut out = Vec::with_capacity(bars.len());
    for (i, b) in bars.iter().enumerate() {
        if i < cut {
            out.push(b.clone());
        } else {
            // Garbage value sentinels: NaN on close to surface any naive
            // .max/.min that propagates forward; arbitrary 12345.0 on
            // open/high/low so downstream arithmetic still terminates.
            out.push(Bar {
                time_unix: b.time_unix,
                open: 12_345.0,
                high: 12_345.0,
                low: 12_345.0,
                close: f64::NAN,
                volume: f64::NAN,
            });
        }
    }
    out
}

/// Run `f` against `bars` and against `pollute_bars_after(bars, cut)`;
/// assert the first `cut` outputs agree element-by-element. Panics with
/// `name` in the message on mismatch.
///
/// Returns the clean output so callers that want to inspect it can do
/// so without an extra call.
pub fn assert_no_lookahead<R, F>(name: &str, bars: &[Bar], cut: usize, f: F) -> Vec<R>
where
    F: Fn(&[Bar]) -> Vec<R>,
    R: PartialEq + std::fmt::Debug + Clone,
{
    let clean = f(bars);
    let polluted = pollute_bars_after(bars, cut);
    let poll = f(&polluted);
    let stop = cut.min(clean.len()).min(poll.len());
    for i in 0..stop {
        assert!(
            clean[i] == poll[i],
            "lookahead invariant {:?} leaked future bars into output[{}]: \
             clean={:?} polluted={:?}",
            name, i, clean[i], poll[i]
        );
    }
    clean
}
