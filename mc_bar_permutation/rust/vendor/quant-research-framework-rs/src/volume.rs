//! Volume indicators (v0.6.0). Rust mirror of
//! `backtester/volume_indicators.py`. Pure functions; every value at index
//! `i` uses only bars `0..=i` (no look-ahead). Seeds and session-reset
//! boundaries are specified IDENTICALLY to the Python module so the two
//! engines match within f64 noise (parity gate 1e-3 paper / 1e-9 default in
//! tools/parity_volume.py).
//!
//! This module is UNCONDITIONAL (always compiled, like src/metrics.rs) so the
//! volume example strategies stay auto-discovered without a Cargo feature.
//!
//! Convention notes (kept in lockstep with the Python mirror):
//!   * Volume SMA / z-score: trailing window ENDING at i; NaN before full
//!     window; z-score uses POPULATION std (ddof=0) via a naive two-pass loop
//!     IDENTICAL to Python, guarded on var <= VAR_FLOOR (not exact ==0).
//!   * Volume EMA: ewm(span, adjust=False) seeded at volume[0].
//!   * OBV: OBV[0] = 0 (TradingView); tie close[i]==close[i-1] adds 0.
//!   * A/D: AD[0] = mfm[0]*vol[0]; mfm = 0 when high==low.
//!   * MFI: NaN until `length` typical-price deltas exist (i < length);
//!     tie tp[i]==tp[i-1] contributes to neither pos nor neg; a fully-flat
//!     window (pos_sum==0 && neg_sum==0) returns NaN (TradingView-faithful).
//!   * VWAP rolling-N: window ENDING at i; NaN before full window or zero
//!     window-volume.
//!   * VWAP session-anchored: cumulative, reset when `reset[i]` is true.

use crate::Bar;

/// Magnitude floor for the z-score zero-variance guard. Matched in Python.
const VAR_FLOOR: f64 = 1e-300;

/// Typical price (high + low + close) / 3.
#[inline]
pub fn typical_price(b: &Bar) -> f64 {
    (b.high + b.low + b.close) / 3.0
}

/// Volume simple moving average. Trailing window ending at `i`. NaN until
/// `length` bars are accumulated.
pub fn volume_sma(volume: &[f64], length: usize) -> Vec<f64> {
    let n = volume.len();
    let mut out = vec![f64::NAN; n];
    if length == 0 || n < length {
        return out;
    }
    for i in (length - 1)..n {
        let window = &volume[i + 1 - length..=i];
        out[i] = window.iter().sum::<f64>() / length as f64;
    }
    out
}

/// Volume exponential moving average, ewm(span, adjust=False) seeded at
/// volume[0]. Matches compute_ema (src/lib.rs:388).
pub fn volume_ema(volume: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span as f64 + 1.0);
    let n = volume.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    out[0] = volume[0];
    for i in 1..n {
        // pandas' ewm constant-run guard; see `compute_ema` in src/lib.rs.
        // Flat-volume stretches are common, so this path hits often.
        out[i] = if out[i - 1] == volume[i] {
            volume[i]
        } else {
            alpha * volume[i] + (1.0 - alpha) * out[i - 1]
        };
    }
    out
}

/// Relative volume = volume / volume_sma(length). NaN where the SMA is NaN
/// or zero (avoid div-by-zero).
pub fn relative_volume(volume: &[f64], length: usize) -> Vec<f64> {
    let sma = volume_sma(volume, length);
    volume
        .iter()
        .zip(sma.iter())
        .map(|(&v, &s)| {
            if s.is_nan() || s == 0.0 {
                f64::NAN
            } else {
                v / s
            }
        })
        .collect()
}

/// Volume z-score over a trailing window of `length` ending at i. POPULATION
/// std (ddof = 0) via a naive two-pass loop IDENTICAL to Python. NaN before
/// the full window or when var <= VAR_FLOOR.
pub fn volume_zscore(volume: &[f64], length: usize) -> Vec<f64> {
    let n = volume.len();
    let mut out = vec![f64::NAN; n];
    if length == 0 || n < length {
        return out;
    }
    for i in (length - 1)..n {
        let window = &volume[i + 1 - length..=i];
        let mean = window.iter().sum::<f64>() / length as f64;
        let var = window.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / length as f64;
        if var <= VAR_FLOOR {
            continue; // leave NaN
        }
        out[i] = (volume[i] - mean) / var.sqrt();
    }
    out
}

/// On-Balance Volume. OBV[0] = 0 (TradingView). Tie close[i]==close[i-1]
/// adds nothing.
pub fn obv(bars: &[Bar], volume: &[f64]) -> Vec<f64> {
    let n = bars.len();
    let mut out = vec![0.0f64; n];
    if n == 0 {
        return out;
    }
    out[0] = 0.0;
    for i in 1..n {
        let d = bars[i].close - bars[i - 1].close;
        out[i] = if d > 0.0 {
            out[i - 1] + volume[i]
        } else if d < 0.0 {
            out[i - 1] - volume[i]
        } else {
            out[i - 1]
        };
    }
    out
}

/// Accumulation / Distribution line. Cumulative from bar 0.
/// mfm = ((close - low) - (high - close)) / (high - low); mfm = 0 if high==low.
pub fn ad_line(bars: &[Bar], volume: &[f64]) -> Vec<f64> {
    let n = bars.len();
    let mut out = vec![0.0f64; n];
    if n == 0 {
        return out;
    }
    let mfm = |b: &Bar| -> f64 {
        let rng = b.high - b.low;
        if rng == 0.0 {
            0.0
        } else {
            ((b.close - b.low) - (b.high - b.close)) / rng
        }
    };
    out[0] = mfm(&bars[0]) * volume[0];
    for i in 1..n {
        out[i] = out[i - 1] + mfm(&bars[i]) * volume[i];
    }
    out
}

/// Money Flow Index. NaN until `length` typical-price deltas exist
/// (i < length). Trailing window of `length` raw-money-flow terms. Tie
/// tp[i]==tp[i-1] contributes to neither pos nor neg. Fully-flat window
/// (pos_sum==0 && neg_sum==0) returns NaN (TradingView-faithful).
pub fn mfi(bars: &[Bar], volume: &[f64], length: usize) -> Vec<f64> {
    let n = bars.len();
    let mut out = vec![f64::NAN; n];
    if n < 2 || length == 0 {
        return out;
    }
    let tp: Vec<f64> = bars.iter().map(typical_price).collect();
    let mut pos = vec![0.0f64; n];
    let mut neg = vec![0.0f64; n];
    for i in 1..n {
        let rmf = tp[i] * volume[i];
        if tp[i] > tp[i - 1] {
            pos[i] = rmf;
        } else if tp[i] < tp[i - 1] {
            neg[i] = rmf;
        }
    }
    // Window of `length` deltas ending at i: indices (i-length+1)..=i.
    // First valid i is `length` (need `length` deltas; deltas start at 1).
    for i in length..n {
        let lo = i + 1 - length;
        let pos_sum: f64 = pos[lo..=i].iter().sum();
        let neg_sum: f64 = neg[lo..=i].iter().sum();
        out[i] = if pos_sum == 0.0 && neg_sum == 0.0 {
            f64::NAN // flat window: TradingView returns na
        } else if neg_sum == 0.0 {
            100.0
        } else {
            let mr = pos_sum / neg_sum;
            100.0 - 100.0 / (1.0 + mr)
        };
    }
    out
}

/// Rolling-N VWAP. Window ending at i: sum(tp*vol)/sum(vol) over
/// (i-length+1)..=i. NaN before the full window or zero window-volume.
pub fn vwap_rolling(bars: &[Bar], volume: &[f64], length: usize) -> Vec<f64> {
    let n = bars.len();
    let mut out = vec![f64::NAN; n];
    if length == 0 || n < length {
        return out;
    }
    let tp: Vec<f64> = bars.iter().map(typical_price).collect();
    for i in (length - 1)..n {
        let lo = i + 1 - length;
        let mut pv = 0.0f64;
        let mut vv = 0.0f64;
        for k in lo..=i {
            pv += tp[k] * volume[k];
            vv += volume[k];
        }
        if vv != 0.0 {
            out[i] = pv / vv;
        }
    }
    out
}

/// Session-anchored VWAP. `reset[i] == true` means bar i opens a new session
/// (the accumulator restarts AT i, including bar i). The caller supplies the
/// reset flags (computed from NY-tz wall clock) so the boundary is identical
/// to Python. NaN only if cumulative session volume is 0 at i (degenerate).
pub fn vwap_session(bars: &[Bar], volume: &[f64], reset: &[bool]) -> Vec<f64> {
    let n = bars.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let tp: Vec<f64> = bars.iter().map(typical_price).collect();
    let mut pv = 0.0f64;
    let mut vv = 0.0f64;
    for i in 0..n {
        if reset[i] || i == 0 {
            pv = 0.0;
            vv = 0.0;
        }
        pv += tp[i] * volume[i];
        vv += volume[i];
        if vv != 0.0 {
            out[i] = pv / vv;
        }
    }
    out
}

/// Session reset flags from NY-tz wall-clock. `reset[i]` is true when bar i's
/// NY calendar date differs from bar i-1, OR bar i crosses the anchor hour
/// (prev_hour < anchor_hour <= cur_hour). reset[0] is always a session open.
/// Uses chrono-tz America/New_York (already a direct dep, see src/lib.rs:45).
pub fn ny_session_resets(times_unix: &[i64], anchor_hour: u32) -> Vec<bool> {
    use chrono::{Datelike, TimeZone, Timelike, Utc};
    use chrono_tz::America::New_York;
    let n = times_unix.len();
    let mut out = vec![false; n];
    if n == 0 {
        return out;
    }
    let ny = |ts: i64| {
        Utc.timestamp_opt(ts, 0)
            .single()
            .expect("invalid unix timestamp")
            .with_timezone(&New_York)
    };
    let mut prev = ny(times_unix[0]);
    out[0] = true;
    for i in 1..n {
        let cur = ny(times_unix[i]);
        let date_changed = (cur.year(), cur.ordinal()) != (prev.year(), prev.ordinal());
        let crossed_anchor = prev.hour() < anchor_hour && cur.hour() >= anchor_hour;
        out[i] = date_changed || crossed_anchor;
        prev = cur;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bar;

    fn bar(o: f64, h: f64, l: f64, c: f64) -> Bar {
        Bar { time_unix: 0, open: o, high: h, low: l, close: c, volume: 0.0 }
    }

    #[test]
    fn obv_seed_zero_and_direction() {
        let bars = vec![bar(1.0, 1.0, 1.0, 10.0), bar(1.0, 1.0, 1.0, 11.0), bar(1.0, 1.0, 1.0, 10.5)];
        let vol = vec![100.0, 200.0, 50.0];
        let o = obv(&bars, &vol);
        assert_eq!(o[0], 0.0);
        assert_eq!(o[1], 200.0);
        assert_eq!(o[2], 150.0);
    }

    #[test]
    fn ad_line_guards_zero_range() {
        let bars = vec![bar(5.0, 5.0, 5.0, 5.0)]; // high==low
        let vol = vec![123.0];
        let a = ad_line(&bars, &vol);
        assert_eq!(a[0], 0.0);
    }

    #[test]
    fn volsma_nan_until_full_window() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let s = volume_sma(&v, 3);
        assert!(s[0].is_nan() && s[1].is_nan());
        assert!((s[2] - 2.0).abs() < 1e-12);
        assert!((s[3] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn vwap_rolling_matches_manual() {
        let bars = vec![bar(0.0, 2.0, 1.0, 1.5), bar(0.0, 4.0, 2.0, 3.0)];
        let vol = vec![10.0, 20.0];
        let w = vwap_rolling(&bars, &vol, 2);
        let tp0 = (2.0 + 1.0 + 1.5) / 3.0;
        let tp1 = (4.0 + 2.0 + 3.0) / 3.0;
        let expect = (tp0 * 10.0 + tp1 * 20.0) / 30.0;
        assert!((w[1] - expect).abs() < 1e-12);
    }

    #[test]
    fn mfi_flat_window_is_nan() {
        // identical tp on every bar -> all deltas zero -> NaN window
        let bars = vec![bar(1.0, 1.0, 1.0, 1.0); 5];
        let vol = vec![10.0; 5];
        let m = mfi(&bars, &vol, 3);
        assert!(m[4].is_nan());
    }

    #[test]
    fn zscore_zero_variance_is_nan() {
        let v = vec![7.0, 7.0, 7.0, 7.0];
        let z = volume_zscore(&v, 3);
        assert!(z[2].is_nan() && z[3].is_nan());
    }
}
