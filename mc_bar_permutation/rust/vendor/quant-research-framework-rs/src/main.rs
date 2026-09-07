//! Reference binary: EMA(20) vs EMA(lb) crossover strategy.
//!
//! This is the 1-to-1 port of `backtester.py`'s baseline strategy in the
//! sibling Python repo. Edit `ema_crossover` below to plug in a different
//! strategy, or add a new binary under `examples/` (see `examples/atr_cross.rs`).

use quant_research_framework_rs::{compute_ema, run_with_csv, Bar};

/// Raw-signals function. Must return a `Vec<i8>` the same length as `bars`
/// where each element is:
///   *  +1 if the bar should be **long**,
///   *  -1 if the bar should be **short**,
///   *   0 if there is no signal yet (indicator not warmed up, etc).
///
/// The backtester later runs `parse_signals` over this vector to detect
/// position flips and convert them into entry/exit codes (1 = enter long,
/// 3 = enter short, 2 = exit long, 4 = exit short).
///
/// Index convention: `raw[i]` is the desired position *at bar i*, computed
/// from information available at bar `i-1` (no look-ahead).
fn ema_crossover(bars: &[Bar], lb: usize) -> Vec<i8> {
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ema_fast = compute_ema(&close, 20);
    let ema_slow = compute_ema(&close, lb);
    let n = bars.len();
    let mut raw = vec![0i8; n];
    for i in 1..n {
        if ema_fast[i - 1] > ema_slow[i - 1] {
            raw[i] = 1;
        } else if ema_fast[i - 1] < ema_slow[i - 1] {
            raw[i] = -1;
        }
    }
    raw
}

fn main() {
    run_with_csv("data/SOLUSDT_1h.csv", "EMA-crossover", ema_crossover);
}
