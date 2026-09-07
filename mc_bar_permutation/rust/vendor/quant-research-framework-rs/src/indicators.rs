//! Shared TradingView/pandas-style indicators (roadmap item 5, indicator
//! parity). Single source of truth for the example strategies so they stop
//! re-deriving indicators inline and drifting from the Python reference.
//!
//! Every function here is a bit-for-bit (to f64 noise) mirror of the
//! corresponding function in the Python reference
//! `backtester/indicators.py`. The cross-language guarantee is the
//! dedicated `tools/parity_indicators.py` harness (1e-3 gate on a
//! real-shaped OHLC fixture that forces the RSI zero, Stochastic inf,
//! Stochastic NaN-window, and indicator-of-indicator branches), NOT the
//! stdout-metric parity surface, none of these are on the default
//! EMA-cross parity path.
//!
//! Gated behind the `indicators` Cargo feature so it never touches the
//! default build / default parity surface. Used by `examples/batch_runner.rs`
//! and `examples/_parity_indicators.rs`.
//!
//! Provenance: `ewm_adjusted`, `compute_sma`, `ema`, `compute_atr` are lifted
//! (arithmetic identical; re-signatured from `&[Bar]` to `&[f64]` slice args)
//! from the already-correct `examples/atr_cross.rs`. `compute_rsi` is lifted
//! the same way but DROPS the `avg_loss==0 -> 100` guard that file carried ,
//! pandas has no such guard, so the guard diverged on flat data (pandas:
//! 0/0 -> NaN). `compute_ema`, `compute_macd`, `compute_stoch` are added here
//! to round out the shared surface against `backtester/indicators.py`.

use crate::Bar;

/// Pandas-style adjusted EWM with `min_periods`. Matches
/// `series.ewm(alpha=alpha, min_periods=mp).mean()` (default `adjust=True`,
/// `ignore_na=False`): a NaN entry decays the running weights but contributes
/// no value, and the output is NaN until `min_periods` non-NaN inputs have
/// been seen. Shared building block for ATR and RSI.
pub fn ewm_adjusted(series: &[f64], alpha: f64, min_periods: usize) -> Vec<f64> {
    let n = series.len();
    let mut out = vec![f64::NAN; n];
    let gamma = 1.0 - alpha;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut seen = 0usize;
    for i in 0..n {
        let x = series[i];
        num *= gamma;
        den *= gamma;
        if !x.is_nan() {
            num += x;
            den += 1.0;
            seen += 1;
        }
        if seen >= min_periods && den > 0.0 {
            out[i] = num / den;
        }
    }
    out
}

/// Simple moving average. Matches `data[col].rolling(window=length).mean()`:
/// first value at index `length-1`, and NaN PROPAGATES (any NaN in the
/// window -> NaN). This is the `atr_cross.rs` semantics, NOT the running-sum
/// shortcut in the old `batch_runner.rs` (which silently swallowed NaNs).
pub fn compute_sma(series: &[f64], length: usize) -> Vec<f64> {
    let n = series.len();
    let mut out = vec![f64::NAN; n];
    if length == 0 || n < length {
        return out;
    }
    for i in (length - 1)..n {
        let window = &series[i + 1 - length..=i];
        if window.iter().any(|v| v.is_nan()) {
            continue;
        }
        out[i] = window.iter().sum::<f64>() / length as f64;
    }
    out
}

/// EMA, NaN-aware, seeded at the first non-NaN. Equivalent to
/// `series.ewm(span=span, adjust=False).mean()` on a series that may have a
/// LEADING-NaN warmup region. Use this (NOT `compute_ema`, which seeds
/// `out[0]=series[0]` and would poison the whole output if `series[0]` is
/// NaN) for indicator-of-indicator chains (EMA-of-ATR, MACD signal line over
/// a leading-NaN macd). On clean (NaN-free) input the two are bit-identical.
pub fn ema(series: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span as f64 + 1.0);
    let n = series.len();
    let mut out = vec![f64::NAN; n];
    let mut state = f64::NAN;
    for i in 0..n {
        let v = series[i];
        if v.is_nan() {
            continue;
        }
        state = if state.is_nan() {
            v
        } else if state == v {
            // pandas' ewm constant-run guard; see `compute_ema` in src/lib.rs.
            v
        } else {
            alpha * v + (1.0 - alpha) * state
        };
        out[i] = state;
    }
    out
}

/// EMA on a CLEAN (NaN-free) close series. Matches
/// `data['close'].ewm(span=length, adjust=False).mean()` and is byte-identical
/// to `crate::compute_ema` (`src/lib.rs:388`, same `out[0]=close[0]` seed).
/// PRECONDITION: `close` has no leading NaN, a leading NaN would poison the
/// whole output (debug_assert below); for warmup'd inputs use `ema` instead.
pub fn compute_ema(close: &[f64], span: usize) -> Vec<f64> {
    debug_assert!(
        close.is_empty() || !close[0].is_nan(),
        "compute_ema expects a clean (no leading-NaN) close series; \
         use `ema` for indicator-of-indicator inputs"
    );
    let alpha = 2.0 / (span as f64 + 1.0);
    let n = close.len();
    let mut out = vec![0.0f64; n];
    if n == 0 {
        return out;
    }
    out[0] = close[0];
    for i in 1..n {
        // Same constant-run guard as `crate::compute_ema`; see the comment
        // there for why the unguarded form drifts by one ulp.
        out[i] = if out[i - 1] == close[i] {
            close[i]
        } else {
            alpha * close[i] + (1.0 - alpha) * out[i - 1]
        };
    }
    out
}

/// MACD line + signal line. Matches `compute_macd` in
/// `backtester/indicators.py`:
///   fast = EMA(close, fast_length);  slow = EMA(close, slow_length)
///   macd = fast - slow;  signal = EMA(macd, signal_length)
/// All EMAs are `adjust=False`. Operates on a CLEAN close series (uses the
/// `compute_ema` seed `out[0]=close[0]`, matching the pandas span EMA on a
/// NaN-free input). Both close-EMAs are finite from index 0, so `macd` has no
/// leading NaN and the signal-line seed agrees with pandas. Returns
/// `(macd, signal)`.
pub fn compute_macd(
    close: &[f64],
    fast_length: usize,
    slow_length: usize,
    signal_length: usize,
) -> (Vec<f64>, Vec<f64>) {
    let fast = compute_ema(close, fast_length);
    let slow = compute_ema(close, slow_length);
    let macd: Vec<f64> = fast
        .iter()
        .zip(slow.iter())
        .map(|(a, b)| a - b)
        .collect();
    let signal = compute_ema(&macd, signal_length);
    (macd, signal)
}

/// ATR = adjusted EWM (alpha = 1/length, min_periods = length) of true range.
/// Matches `compute_atr` in `backtester/indicators.py`:
///   TR_i = max(H_i-L_i, |H_i-C_{i-1}|, |L_i-C_{i-1}|),  TR_0 = H_0-L_0
///   (pandas `DataFrame.max(axis=1)` skips the NaN hc/lc at index 0)
///   ATR  = TR.ewm(alpha=1/length, min_periods=length).mean()
pub fn compute_atr(high: &[f64], low: &[f64], close: &[f64], length: usize) -> Vec<f64> {
    let n = high.len();
    let mut tr = vec![f64::NAN; n];
    if n == 0 {
        return tr;
    }
    tr[0] = high[0] - low[0];
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }
    ewm_adjusted(&tr, 1.0 / length as f64, length)
}

/// Bar-slice convenience wrapper for `compute_atr` (used by examples).
pub fn compute_atr_bars(bars: &[Bar], length: usize) -> Vec<f64> {
    let high: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let low: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    compute_atr(&high, &low, &close, length)
}

/// RSI = 100 - 100/(1 + avg_gain/avg_loss), averages = pandas adjusted EWM
/// with `com=length-1` (<=> alpha = 1/length, adjust=True, min_periods=length).
/// Matches `compute_rsi` in `backtester/indicators.py` EXACTLY, including the
/// flat/saturated edge cases:
///   delta[0] = NaN => gain[0] = loss[0] = 0 (NaN>0 and NaN<0 are both False);
///   NaN for indices 0..length-1 (warmup);
///   NO zero-guard, `rs = avg_gain/avg_loss` is computed directly, so:
///     avg_gain==0 && avg_loss==0 (flat)        => rs = 0/0 = NaN  => RSI = NaN
///     avg_gain>0  && avg_loss==0 (monotone-up) => rs = x/0 = inf  => RSI = 100
///   These arise from IEEE-754 division identically on both sides, verified
///   against pandas. (The old `atr_cross.rs` port hard-coded
///   `if avg_loss==0 {100}`, which returns 100 on the FLAT case where pandas
///   returns NaN; that guard is intentionally REMOVED here.)
pub fn compute_rsi(close: &[f64], length: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < 2 || length == 0 {
        return out;
    }
    let mut gain = vec![0.0f64; n];
    let mut loss = vec![0.0f64; n];
    for i in 1..n {
        let d = close[i] - close[i - 1];
        if d > 0.0 {
            gain[i] = d;
        } else if d < 0.0 {
            loss[i] = -d;
        }
    }
    let alpha = 1.0 / length as f64;
    let avg_gain = ewm_adjusted(&gain, alpha, length);
    let avg_loss = ewm_adjusted(&loss, alpha, length);
    for i in 0..n {
        if avg_gain[i].is_nan() || avg_loss[i].is_nan() {
            continue;
        }
        // No guard: let IEEE-754 produce NaN (0/0, flat) or 100 (x/0 -> inf),
        // exactly as pandas `avg_gain/avg_loss` then `100 - 100/(1+rs)` does.
        let rs = avg_gain[i] / avg_loss[i];
        out[i] = 100.0 - 100.0 / (1.0 + rs);
    }
    out
}

/// Bar-slice convenience wrapper for `compute_rsi` (close only).
pub fn compute_rsi_bars(bars: &[Bar], length: usize) -> Vec<f64> {
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    compute_rsi(&close, length)
}

/// Stochastic %K. Matches `compute_stoch` in `backtester/indicators.py`:
///   lo = low.rolling(length).min();  hi = high.rolling(length).max()
///   %K = 100 * (close - lo) / (hi - lo)
/// Rolling min/max use `min_periods = length` (pandas default), so output is
/// NaN for indices 0..length-1. NaN PROPAGATES from the rolling min/max: any
/// NaN in the window -> NaN K (verified against pandas `rolling().min/max()`),
/// so we skip windows containing a NaN rather than letting `f64::min/max`
/// silently swallow it. For VALID OHLC a flat window forces close == hi == lo,
/// so the numerator is also 0 => 0.0/0.0 = NaN on BOTH sides (matching pandas
/// `0/0 -> NaN`). The malformed corner (hi==lo, close!=lo) -> +/-inf on both
/// sides via the same direct division (no early guard).
pub fn compute_stoch(high: &[f64], low: &[f64], close: &[f64], length: usize) -> Vec<f64> {
    let n = high.len();
    let mut out = vec![f64::NAN; n];
    if length == 0 || n < length {
        return out;
    }
    for i in (length - 1)..n {
        let hw = &high[i + 1 - length..=i];
        let lw = &low[i + 1 - length..=i];
        // pandas rolling().min/max() propagate NaN; f64::min/max do NOT, so
        // guard explicitly (mirrors compute_sma's NaN propagation).
        if hw.iter().any(|v| v.is_nan()) || lw.iter().any(|v| v.is_nan()) || close[i].is_nan() {
            continue;
        }
        let lo = lw.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = hw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // Direct: 100*(close-lo)/(hi-lo). When hi==lo the denominator is 0;
        // for valid OHLC the numerator (close-lo) is then also 0 => 0/0 = NaN
        // (matching pandas). Malformed hi==lo,close!=lo => +/-inf, also matching.
        out[i] = 100.0 * (close[i] - lo) / (hi - lo);
    }
    out
}

/// Bar-slice convenience wrapper for `compute_stoch`.
pub fn compute_stoch_bars(bars: &[Bar], length: usize) -> Vec<f64> {
    let high: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let low: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    compute_stoch(&high, &low, &close, length)
}
