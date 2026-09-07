// Crate-wide clippy carve-outs. Each is intentional and reviewed; collected
// here so CI can run `clippy -- -D warnings` without spurious failures.
//
// * `needless_range_loop`, the WFO/regime/news-injection inner loops use
//   index arithmetic deliberately (parity with the Python numba kernels,
//   which are also index-based). Iterator rewrites change locality and
//   make the parity diff harder to read.
// * `too_many_arguments`, `run_wfo_window` mirrors the Python
//   `_run_wfo_window` signature 1:1; refactoring to a struct would diverge
//   from the reference.
// * `manual_rem_euclid` / `manual_is_multiple_of`, the manual forms are
//   identical to what the Python source does and the lints flag stylistic
//   modernisation that would only obscure parity.
// * `field_reassign_with_default`, the `Metrics { pf: INFINITY, .. }`
//   builder used in `compute_metrics_for` is post-hoc on the empty branch
//   and not worth a struct-update rewrite.
// * `neg_cmp_op_on_partial_ord`, the regime detector explicitly uses
//   `!(a > b)` semantics; NaN guards are inserted ahead of the comparison
//   so the negation is intentional. See comment at the call site.
// * `new_without_default`, `Config::new()` is the public constructor that
//   reads the engine's compile-time constants; a `Default::default()` would
//   look like it could be reassigned to "all zeros" and is confusing here.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_rem_euclid)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::new_without_default)]

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write as IoWrite};
use std::path::Path;
mod ledger_lock;
use ledger_lock::LedgerGuard;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Set by `run` / `run_with_regime` from cfg.use_forex so prettyprint can
/// format ROI / Exp / MaxDD in R-units (forex) or dollars (crypto).
static DISPLAY_FOREX: AtomicBool = AtomicBool::new(false);

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use chrono::{TimeZone, Timelike};
use chrono_tz::America::New_York;

pub mod monte_carlo;
pub use monte_carlo::{run_trade_monte_carlo, MonteCarloMode, MonteCarloResult};
pub mod bar_permutation;
pub use bar_permutation::permute_bars;

// ============================================================================
// CONFIGURATION, mirrors backtester.py constants
// ============================================================================
// CSV path is taken from $BT_CSV at runtime (see src/main.rs); there is
// no compile-time constant for it.
const ACCOUNT_SIZE: f64 = 100_000.0;
const RISK_AMOUNT: f64 = 2_500.0;
const SLIPPAGE_PCT_DEFAULT: f64 = 0.03;
const FEE_PCT_DEFAULT: f64 = 0.02;
pub const FUNDING_FEE: f64 = 0.01;
pub const DEFAULT_LB: usize = 50;
pub const BACKTEST_CANDLES: usize = 10_000;
const OOS_CANDLES_BASE: usize = 90_000;
const USE_OOS2: bool = false;
/// `Consistency` is a project-specific example designed by Daniel Gatto. The
/// name does not refer to an established financial metric; see README.
pub const OPT_METRIC: &str = "Sharpe";
pub const MIN_TRADES: usize = 10;
pub const SMART_OPTIMIZATION: bool = true;
const DRAWDOWN_CONSTRAINT: Option<f64> = None;
const USE_MONTE_CARLO: bool = true;
const MC_RUNS: usize = 1000;
const MC_MODE: MonteCarloMode = MonteCarloMode::Resampling;
const MC_SEED: u64 = 42;
const USE_SL: bool = true;
const SL_PERCENTAGE: f64 = 1.0;
const USE_TP_DEFAULT: bool = true;
const TP_PERCENTAGE_DEFAULT: f64 = 3.0;
pub const OPTIMIZE_RRR: bool = true;
pub const USE_WFO: bool = true;
const WFO_TRIGGER_MODE: &str = "candles";
pub const WFO_TRIGGER_VAL: usize = 5000;
const FAST_EMA_SPAN: usize = 20;

// Forex mode: when true, funding fees are skipped (FX brokers don't charge
// crypto-style perpetual funding). PnL semantics follow the Python reference.
const USE_FOREX: bool = false;
/// Sharpe convention. false = per-trade statistic (default, unchanged);
/// true = standard bar-based calendar-time Sharpe on the mark-to-market equity
/// curve. Mirrors Python `SHARPE_MODE` ("trade" | "bar").
const SHARPE_BAR: bool = false;

// Session mode: when true, only entries inside the NY tz
// [SESSION_START_HOUR, SESSION_END_HOUR) window are taken; positions are
// force-closed on the last in-session bar of each NY-tz day. Times are
// interpreted in America/New_York with DST (matches Python's load_ohlc
// + in_session semantics).
const USE_SESSIONS: bool = false;
const SESSION_START_HOUR: u32 = 8;    // NY local hour
const SESSION_END_HOUR: u32 = 17;     // NY local hour (exclusive); covers the
                                      // 16:00..16:59 range which matches
                                      // Python's "8:00".."16:50" window for
                                      // any bar whose minute is < 50.

// Robustness scenario flag: news-candle injection (sparse high-vol wicks).
// When the scenario list contains "NEWS_CANDLES_INJECTION", inject_news_candles
// produces a perturbed copy of the bar series before backtest.
const NEWS_INJECTION_SEED: u64 = 42;
// INDICATOR_VARIANCE seed: previously unseeded (paper-time review noted that
// the +/- 1 LB perturbation propagated to a different optimised parameter
// across host architectures). Seeded to 42 here so the perturbation is
// deterministic and the cross-architecture parity claim covers IND too.
const IND_VARIANCE_SEED: u64 = 42;

fn robustness_scenarios() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("Test 1", vec!["ENTRY_DRIFT"]),
        ("Test 2", vec!["FEE_SHOCK"]),
        ("Test 3", vec!["SLIPPAGE_SHOCK"]),
        ("Test 4", vec!["ENTRY_DRIFT", "INDICATOR_VARIANCE"]),
    ]
}
const MAX_ROBUSTNESS_SCENARIOS: usize = 5;
const METRICS_LIST: [&str; 6] = ["ROI", "PF", "Sharpe", "WinRate", "Exp", "MaxDrawdown"];
const AGE_DATASET: usize = 0;

// ============================================================================
// DATA STRUCTURES
// ============================================================================
// Mimic Python pandas iloc negative index behavior
fn python_iloc_idx(idx: isize, length: usize) -> usize {
    if idx >= 0 { (idx as usize).min(length) }
    else if (-idx) as usize <= length { (length as isize + idx) as usize }
    else { 0 }
}

/// Resolve a Python iloc slice [start_raw:end_raw] into (usize, usize).
/// If resolved start >= resolved end, returns (x, x) i.e. empty range.
fn python_iloc_slice(start_raw: i64, end_raw: i64, length: usize) -> (usize, usize) {
    let s = python_iloc_idx(start_raw as isize, length);
    let e = python_iloc_idx(end_raw as isize, length);
    if s >= e { (s, s) } else { (s, e) }
}

pub mod invariants;
pub mod metrics;
pub mod objectives;
pub mod orchestrator;

#[cfg(feature = "panel")]
pub mod panel;

#[cfg(feature = "pairs")]
pub mod pairs;

#[cfg(feature = "carry")]
pub mod carry;

// Deflated Sharpe Ratio post-processing (roadmap item 09, Rust mirror of
// `backtester/dsr.py`). Standalone scalar utility gated behind its own
// lightweight `dsr` feature so it does not drag in the panel/pairs stack.
#[cfg(feature = "dsr")]
pub mod dsr;

// Shared TradingView/pandas-style indicators (roadmap item 5, indicator
// parity). Single source of truth for the example strategies; gated behind
// the `indicators` feature so it never touches the default build or the
// default EMA-cross parity surface. Guarded by tools/parity_indicators.py.
#[cfg(feature = "indicators")]
pub mod indicators;

// Overfitting-statistics layer (roadmap item 3). PBO/CSCV + multiple-testing
// haircut. Gated behind `overfit` (pulls in `dsr` for PSR/MinTRL). Outside
// the engine parity surface; guarded by tools/parity_pbo.py / parity_multitest.py.
#[cfg(feature = "overfit")]
pub mod pbo;
#[cfg(feature = "overfit")]
pub mod multitest;
#[cfg(feature = "overfit")]
pub mod haircut;

// Volume indicators (v0.6.0). UNCONDITIONAL like src/metrics.rs.
pub mod volume;

// IS parameter-robustness isosurface emit (pure std + f64).
pub mod opt_surface;

#[derive(Clone)]
pub struct Bar {
    pub time_unix: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// (v0.6.0): per-bar volume; 0.0 when 6th CSV column absent.
    pub volume: f64,
}

impl Bar {
    /// Construct a clean OHLC bar with volume defaulted to 0.0.
    pub fn ohlc(time_unix: i64, open: f64, high: f64, low: f64, close: f64) -> Self {
        Bar { time_unix, open, high, low, close, volume: 0.0 }
    }
}

#[derive(Clone, Debug)]
pub struct Trade {
    pub side: i8,
    pub entry_idx: i32,
    pub exit_idx: i32,
    pub entry_price: f64,
    pub exit_price: f64,
    pub qty: f64,
    pub pnl: f64,
    /// leg index within a multi-leg trade group. Always 0 in
    /// single-leg single-asset mode. Set by the backtest core at every
    /// `trades.push(...)` site; aggregation lives downstream.
    pub leg_id: u32,
    /// shared group id across legs of the same logical trade.
    /// Single-leg mode emits `trade_group_id == trades.len()` at push
    /// time so every leg row is its own group. The trade ledger CSV
    /// schema stays at 15 columns (these fields are internal); parity
    /// vs Python is preserved.
    pub trade_group_id: u64,
    /// cost decomposition. fee + slippage + funding +
    /// net_pnl == gross_pnl by construction, satisfying the leg-level
    /// identity to floating-point tolerance. net_pnl == pnl. In forex
    /// mode slippage and funding are 0 (R-unit math folds them in).
    pub fee: f64,
    pub slippage: f64,
    pub funding: f64,
    pub gross_pnl: f64,
    pub net_pnl: f64,
}

#[derive(Clone, Debug)]
pub struct Metrics {
    pub trades: usize,
    pub roi: f64,
    pub pf: f64,
    pub win_rate: f64,
    pub exp: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    /// Daniel Gatto's project-specific example objective, not a standard
    /// financial metric.
    pub consistency: f64,
    pub rrr: Option<usize>,
}
impl Default for Metrics {
    fn default() -> Self {
        Metrics { trades:0, roi:0.0, pf:0.0, win_rate:0.0, exp:0.0, sharpe:0.0,
                  max_drawdown:0.0, consistency:0.0, rrr:None }
    }
}
impl Metrics {
    fn get(&self, key: &str) -> f64 {
        match key {
            "ROI" => self.roi, "PF" => self.pf, "Sharpe" => self.sharpe,
            "WinRate" => self.win_rate, "Exp" => self.exp,
            "MaxDrawdown" => self.max_drawdown, "Consistency" => self.consistency, _ => 0.0,
        }
    }
}

#[derive(Clone)]
pub struct Config {
    /// Per-run ledger destination. Defaults to BT_EXPORT_PATH or trade_list.csv.
    pub export_path: String,
    pub tp_percentage: f64,
    pub use_tp: bool,
    pub use_sl: bool,
    pub fee_pct: f64,
    pub slippage_pct: f64,
    pub oos_candles: usize,
    pub position_size: f64,
    pub use_forex: bool,
    /// false = per-trade Sharpe (default); true = bar-based standard Sharpe.
    pub sharpe_bar: bool,
    /// Run completed-trade Monte Carlo diagnostics on the first IS sample.
    pub use_monte_carlo: bool,
    /// One null model per invocation. Resampling is the default because it
    /// produces distributions for order-invariant as well as path metrics.
    pub mc_mode: MonteCarloMode,
    /// Number of trade-level simulations. Bar permutation uses its own runner.
    pub mc_runs: usize,
    /// Stable seed for trade-level Monte Carlo diagnostics.
    pub mc_seed: u64,
    pub use_sessions: bool,
    pub session_start_hour: u32,
    pub session_end_hour: u32,
    pub use_oos2: bool,
    /// When true, the backtest core skips the first 200 bars of each
    /// segment (warm-up), mirrors Python's `use_regime` flag passed to
    /// `_backtest_numba_core`. Set true automatically by the
    /// `run_with_regime` / `run_with_regime_cfg` entrypoints; otherwise
    /// stays false and the warmup does not fire. Affects every backtest
    /// in the run, not just the regime-rotated ones, matching Python's
    /// global-flag semantics.
    pub use_regime_seg: bool,
    /// pip size for forex mode. Default 0.0001 (4-decimal pairs); set to
    /// 0.01 for JPY pairs. Ignored unless `use_forex` is true.
    pub pip_size: f64,
    /// Forex-mode account/risk normaliser. In forex mode Python sets
    /// RISK = ACCOUNT = POSITION = 1.0 so all metrics are reported in
    /// R-units; we surface the same as a single field. Default 100_000
    /// preserves crypto-mode dollar semantics when use_forex is false.
    pub account_size: f64,
    /// Confluence-filter exit-side option, mirroring Python's MASK_EXITS
    /// flag. Default false: confluence rule applies only to entries (1, 3);
    /// exits (2, 4) pass unconditionally. The Rust port does not currently
    /// implement the confluence machinery (Python's `CONFLUENCES`); this
    /// field is reserved so the Config surface stays symmetric and
    /// strategies can be ported without API churn when confluence lands.
    pub mask_exits: bool,
    /// Legacy RRR-side bug toggle. Default false (corrected code path).
    /// Set true to reproduce numerical results of versions <= v0.2.4
    /// where the optimiser's side comparison always took the short
    /// branch. See CHANGELOG v0.3.1 / Python v0.2.5.
    pub legacy_side_bug: bool,
    /// maximum hold period in bars. 0 = no force-close
    /// (engine behaves as pre-#46). When > 0, the kernel emits a code-2
    /// / code-4 close at bar i for any open position with
    /// `(i - ent_bar) >= max_hold_bars`. Priority: news/session >
    /// hold-period > SL/TP intrabar > signal-driven.
    pub max_hold_bars: usize,
    /// Crypto-path gap-handling clamp. Default false (existing behaviour).
    /// When true, every realized trade's pnl-before-costs in the crypto
    /// path is clipped to [-1R, +RRR*R] where R = entry_price *
    /// sl_percentage / 100. Mirrors the R-unit clamping that forex mode
    /// applies unconditionally, so a bar that gaps through SL/TP cannot
    /// record a loss larger than 1R or a gain larger than the configured
    /// RRR. No effect when use_forex is true.
    pub clamp_results: bool,
    // ---- IS isosurface emit ----
    pub emit_opt_surface: bool,
    pub emit_opt_surface_sl: bool,
    pub sl_override: Option<f64>,
    // ---- benchmark v2: funding parameterization ----
    /// Funding rate (% per 8h per leg). Default = FUNDING_FEE so existing
    /// callers and the NET golden are unchanged; set 0.0 for a GROSS run.
    pub funding_fee: f64,
}
impl Config {
    pub fn new() -> Self {
        let oos = if USE_OOS2 { OOS_CANDLES_BASE * 2 } else { OOS_CANDLES_BASE };
        Config { export_path: std::env::var("BT_EXPORT_PATH").unwrap_or_else(|_| "trade_list.csv".into()),
                 tp_percentage: TP_PERCENTAGE_DEFAULT, use_tp: USE_TP_DEFAULT, use_sl: USE_SL,
                 fee_pct: FEE_PCT_DEFAULT, slippage_pct: SLIPPAGE_PCT_DEFAULT,
                 oos_candles: oos, position_size: RISK_AMOUNT,
                 use_forex: USE_FOREX,
                 sharpe_bar: std::env::var("BT_SHARPE_MODE").map(|v| v == "bar").unwrap_or(SHARPE_BAR),
                 use_monte_carlo: std::env::var("BT_USE_MONTE_CARLO")
                     .map(|v| v != "0" && v != "false").unwrap_or(USE_MONTE_CARLO),
                 mc_mode: std::env::var("BT_MC_MODE")
                     .map(|value| MonteCarloMode::parse(&value).unwrap_or_else(|error| panic!("{error}")))
                     .unwrap_or(MC_MODE),
                 mc_runs: std::env::var("BT_MC_RUNS")
                     .map(|value| value.parse::<usize>().expect("BT_MC_RUNS must be a positive integer"))
                     .unwrap_or(MC_RUNS),
                 mc_seed: std::env::var("BT_MC_SEED")
                     .map(|value| value.parse::<u64>().expect("BT_MC_SEED must be a non-negative integer"))
                     .unwrap_or(MC_SEED),
                 use_sessions: USE_SESSIONS,
                 session_start_hour: SESSION_START_HOUR,
                 session_end_hour: SESSION_END_HOUR,
                 use_oos2: USE_OOS2, use_regime_seg: false, pip_size: 0.0001,
                 account_size: ACCOUNT_SIZE, mask_exits: false,
                 legacy_side_bug: false,
                 max_hold_bars: 0, clamp_results: false,
                 emit_opt_surface: std::env::var("EMIT_OPT_SURFACE")
                     .map(|v| v == "1" || v == "true").unwrap_or(false),
                 emit_opt_surface_sl: std::env::var("EMIT_OPT_SURFACE_SL")
                     .map(|v| v == "1" || v == "true").unwrap_or(false),
                 sl_override: None,
                 funding_fee: FUNDING_FEE }
    }
    pub fn with_forex(mut self, on: bool) -> Self { self.use_forex = on; self }
    /// Switch the reported Sharpe to the standard bar-based calendar-time
    /// convention (default off = per-trade statistic).
    pub fn with_sharpe_bar(mut self, on: bool) -> Self { self.sharpe_bar = on; self }
    pub fn with_monte_carlo(mut self, on: bool) -> Self { self.use_monte_carlo = on; self }
    pub fn with_monte_carlo_mode(mut self, mode: MonteCarloMode) -> Self {
        self.mc_mode = mode;
        self
    }
    pub fn with_monte_carlo_runs(mut self, runs: usize) -> Self {
        assert!(runs > 0, "Monte Carlo runs must be positive");
        self.mc_runs = runs;
        self
    }
    pub fn with_monte_carlo_seed(mut self, seed: u64) -> Self { self.mc_seed = seed; self }
    /// Forex defaults: position_size = account_size = 1.0 (R-unit
    /// reporting), use_forex = true. Mirrors Python's module-level
    /// FOREX_MODE setup at import time.
    pub fn with_forex_defaults(mut self) -> Self {
        self.use_forex = true;
        self.position_size = 1.0;
        self.account_size = 1.0;
        self
    }
    /// Set `pip_size` from the dataset path, mirroring the Python
    /// reference's import-time `PIP_SIZE` resolution. Chain after
    /// `with_forex_defaults` so JPY-quoted pairs size correctly:
    /// `Config::new().with_forex_defaults().with_pip_size_for(&csv)`.
    pub fn with_pip_size_for(mut self, csv_path: &str) -> Self {
        self.pip_size = resolve_pip_size(csv_path);
        self
    }
    pub fn with_sessions(mut self, on: bool, start_h: u32, end_h: u32) -> Self {
        self.use_sessions = on; self.session_start_hour = start_h; self.session_end_hour = end_h; self
    }
    pub fn with_oos2(mut self, on: bool) -> Self {
        self.use_oos2 = on;
        self.oos_candles = if on { OOS_CANDLES_BASE * 2 } else { OOS_CANDLES_BASE };
        self
    }
    pub fn with_mask_exits(mut self, on: bool) -> Self { self.mask_exits = on; self }
    pub fn with_legacy_side_bug(mut self, on: bool) -> Self { self.legacy_side_bug = on; self }
    /// cap the number of bars a position can be held.
    /// 0 (default) disables; >0 makes the kernel force-close at bar i
    /// when `(i - ent_bar) >= max_hold_bars`. Mirrors Python's
    /// `MAX_HOLD_BARS` module constant.
    pub fn with_max_hold_bars(mut self, n: usize) -> Self {
        self.max_hold_bars = n;
        self
    }
    /// Enable crypto-path R-clamp for gap-prone instruments. No-op in forex
    /// mode (forex already clamps). See `Config::clamp_results` docs.
    pub fn with_clamp_results(mut self, on: bool) -> Self { self.clamp_results = on; self }
    pub fn with_emit_opt_surface(mut self, on: bool) -> Self { self.emit_opt_surface = on; self }
    pub fn with_emit_opt_surface_sl(mut self, on: bool) -> Self { self.emit_opt_surface_sl = on; self }
    fn fee_rate(&self) -> f64 { self.fee_pct / 100.0 }
    fn slip(&self) -> f64 { self.slippage_pct * 0.01 }
    fn funding_rate(&self) -> f64 { self.funding_fee / 100.0 }
    fn dd_constraint(&self) -> Option<f64> { DRAWDOWN_CONSTRAINT.map(|d| d / 100.0) }
}

// ============================================================================
// Time helpers: UTC for funding (matches Python's UTC funding schedule),
// New_York for sessions (matches Python's load_ohlc tz_convert + DST).
// ============================================================================
fn utc_hour_minute(unix_ts: i64) -> (u32, u32) {
    let secs_in_day = ((unix_ts % 86400) + 86400) % 86400;
    let hour = (secs_in_day / 3600) as u32;
    let minute = ((secs_in_day % 3600) / 60) as u32;
    (hour, minute)
}

fn ny_hour_minute(unix_ts: i64) -> (u32, u32) {
    let dt = chrono::Utc.timestamp_opt(unix_ts, 0).single()
        .expect("invalid unix timestamp");
    let ny = dt.with_timezone(&New_York);
    (ny.hour(), ny.minute())
}

// ============================================================================
// 1. LOAD DATA
// ============================================================================
/// Resolve the forex pip size for a dataset path.
///
/// Mirrors the Python reference's `_resolve_pip_size`: an explicit
/// `BT_PIP_SIZE` wins, otherwise a path containing "JPY" resolves to the
/// two-decimal pip of JPY-quoted crosses and anything else to 0.0001.
/// The substring heuristic is fragile (it fires on a file named
/// "FUJPYR.csv" too), so it warns, exactly as the Python side does.
pub fn resolve_pip_size(csv_path: &str) -> f64 {
    if let Ok(raw) = std::env::var("BT_PIP_SIZE") {
        match raw.parse::<f64>() {
            Ok(v) => return v,
            Err(_) => panic!("BT_PIP_SIZE={:?} is not a valid float", raw),
        }
    }
    if csv_path.to_uppercase().contains("JPY") {
        eprintln!(
            "warning: pip_size auto-set to 0.01 because CSV path {:?} contains \
             'JPY'. This substring heuristic is fragile; set BT_PIP_SIZE \
             explicitly to silence this warning.",
            csv_path
        );
        return 0.01;
    }
    0.0001
}

pub fn load_ohlc(path: &str) -> Vec<Bar> {
    let file = File::open(path).unwrap_or_else(|_| panic!("CSV file not found: {}", path));
    let reader = BufReader::new(file);
    let mut bars: Vec<Bar> = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("Failed to read line");
        if i == 0 { continue; }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 { continue; }
        let time_unix: i64 = fields[0].trim().parse().expect("bad time");
        let open: f64 = fields[1].trim().parse().expect("bad open");
        let high: f64 = fields[2].trim().parse().expect("bad high");
        let low: f64 = fields[3].trim().parse().expect("bad low");
        let close: f64 = fields[4].trim().parse().expect("bad close");
        let volume: f64 = fields.get(5)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        bars.push(Bar { time_unix, open, high, low, close, volume });
    }
    bars.sort_by_key(|b| b.time_unix);
    bars
}

fn age_dataset(bars: Vec<Bar>, age: usize) -> Vec<Bar> {
    if age == 0 { return bars; }
    bars[..bars.len() - age].to_vec()
}

// ============================================================================
// 2. INDICATORS - EMA matching pandas ewm(span=X, adjust=False)
// ============================================================================
pub fn compute_ema(close: &[f64], span: usize) -> Vec<f64> {
    // Ablation hook for the speed-up decomposition (tools/ablations/
    // rolling_materialized.sh). With QRF_ROLLING=materialized set, dispatch to
    // a naive O(n*w) window recompute that is numerically equivalent to the
    // shipped O(n) recursion below. That isolates the algorithmic term (rolling
    // statistics materialised the way pandas computes them) from the language
    // term. With the variable unset this is the shipped path, unchanged.
    if rolling_is_materialized() {
        return compute_ema_materialized(close, span);
    }
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut ema = vec![0.0f64; close.len()];
    if close.is_empty() { return ema; }
    ema[0] = close[0];
    for i in 1..close.len() {
        // pandas' ewm kernel skips the update whenever the running average
        // already equals the incoming observation, and we mirror it. Without
        // the guard, alpha*x + (1-alpha)*x does not round back to x at every
        // span and price level: it lands one unit in the last place away for
        // 125 of the 780 (span, price) pairs we probed, always by 1.42e-14 at
        // the magnitudes involved. On a run of constant-price bars, where the
        // fast and slow EMAs are equal in exact arithmetic, that residue is
        // enough to flip a strict indicator comparison and open a trade the
        // reference does not. Forward-filled non-trading hours make such runs
        // common in FX series.
        ema[i] = if ema[i - 1] == close[i] {
            close[i]
        } else {
            alpha * close[i] + (1.0 - alpha) * ema[i - 1]
        };
    }
    ema
}

/// True iff QRF_ROLLING=materialized is set.
#[inline]
fn rolling_is_materialized() -> bool {
    std::env::var("QRF_ROLLING")
        .map(|v| v == "materialized")
        .unwrap_or(false)
}

/// Window multiplier for the materialised EMA. With adjust=False the EMA is an
/// infinite geometric-weighted sum of past closes with ratio (1-alpha), so a
/// finite window of W = MULT*span truncates a tail of weight (1-alpha)^W. At
/// MULT=40 that tail is ~exp(-80) for any span, far inside the 1e-3 parity
/// tolerance, while the per-bar work is still a genuine O(n*w) recompute.
const EMA_MAT_WINDOW_MULT: usize = 40;

/// Effective window multiplier; QRF_ROLLING_MULT overrides the default. Dialling
/// it down tightens the window and shows the apparent algorithmic factor is a
/// function of the window, not a property of the engines.
#[inline]
fn ema_mat_window_mult() -> usize {
    std::env::var("QRF_ROLLING_MULT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&m| m >= 1)
        .unwrap_or(EMA_MAT_WINDOW_MULT)
}

/// Naive O(n*w) EMA. For every output bar, re-walk a finite window of preceding
/// closes and accumulate the geometric weights from scratch, mimicking how a
/// materialised rolling reduction recomputes each window.
fn compute_ema_materialized(close: &[f64], span: usize) -> Vec<f64> {
    let n = close.len();
    let mut ema = vec![0.0f64; n];
    if n == 0 { return ema; }
    let alpha = 2.0 / (span as f64 + 1.0);
    let one_minus = 1.0 - alpha;
    let window = span.saturating_mul(ema_mat_window_mult()).max(1);
    ema[0] = close[0];
    for i in 1..n {
        let start = i.saturating_sub(window);
        // Seed on the truncation boundary: below `window` bars of history the
        // window covers the whole series, so seed with close[0] exactly as the
        // recursion does; past it, the truncated tail is negligible.
        let mut acc = if start == 0 {
            close[0] * one_minus.powi(i as i32)
        } else {
            close[start] * one_minus.powi((i - start) as i32)
        };
        for k in (start + 1)..=i {
            acc += alpha * close[k] * one_minus.powi((i - k) as i32);
        }
        ema[i] = acc;
    }
    ema
}

// ============================================================================
// 3. RAW SIGNALS, provided by the caller. See src/main.rs for the reference
// EMA-crossover implementation and examples/atr_cross.rs for an ATR variant.
// ============================================================================
pub type RawSignalsFn = fn(&[Bar], usize) -> Vec<i8>;

/// Pluggable regime-detector contract, mirrors Python's `detect_regimes`.
/// Returns one label per bar drawn from `REGIME_LABELS` (encoded as a u8
/// index into that slice). Length must match `bars`. Detectors must be free
/// of look-ahead, only data from bars `0..i` may inform the label at bar `i`.
///
/// Plug a custom detector through `RegimeConfig::new(labels, detector)` and
/// run via `run_with_regime` / `run_with_regime_cfg`. The engine handles
/// per-regime LB optimisation and OOS LB rotation inside each WFO window.
pub type RegimeDetectorFn = fn(&[Bar]) -> Vec<u8>;
pub const REGIME_LABELS: &[&str] = &["Uptrend", "Downtrend", "Ranging"];

// ============================================================================
// 4. PARSE SIGNALS (flip detection)
// ============================================================================
pub fn parse_signals(raw: &[i8]) -> Vec<i8> {
    parse_signals_with_flags(raw, None)
}

/// Session-aware parse: when an out-of-session bar is hit, `pos` is dropped
/// and the *next* in-session bar snaps `pos` to that bar's raw value WITHOUT
/// emitting a flip (matches Python's `_parse_signals_numba` semantics).
/// `in_flags` must be the same length as `raw` if provided.
pub fn parse_signals_with_flags(raw: &[i8], in_flags: Option<&[bool]>) -> Vec<i8> {
    let n = raw.len();
    let mut sig = vec![0i8; n];
    let mut pos: i8 = 0;
    let initial_in = in_flags.map(|f| f.first().copied().unwrap_or(true)).unwrap_or(true);
    let mut in_prev = initial_in;
    for i in 0..n {
        let r = raw[i];
        let in_now = in_flags.map(|f| f[i]).unwrap_or(true);
        if !in_now {
            in_prev = false;
            continue;
        }
        if !in_prev {
            pos = r;
            in_prev = true;
            continue;
        }
        if r == 1 && pos != 1 { sig[i] = 1; pos = 1; }
        else if r == -1 && pos != -1 { sig[i] = 3; pos = -1; }
    }
    sig
}

/// Build the in-session mask for a slice of bars given a Config.
/// Matches the in-session computation used inside `backtest_core`.
pub fn compute_in_flags(bars: &[Bar], cfg: &Config) -> Vec<bool> {
    bars.iter().map(|b| {
        if !cfg.use_sessions { return true; }
        let (h, _m) = ny_hour_minute(b.time_unix);
        h >= cfg.session_start_hour && h < cfg.session_end_hour
    }).collect()
}

/// Engine-internal session-aware parse: applies the session mask up front
/// when `cfg.use_sessions` is set, mirroring Python's `parse_signals(raw, times)`.
fn parse_signals_for(raw: &[i8], bars: &[Bar], cfg: &Config) -> Vec<i8> {
    if cfg.use_sessions {
        let flags = compute_in_flags(bars, cfg);
        parse_signals_with_flags(raw, Some(&flags))
    } else {
        parse_signals_with_flags(raw, None)
    }
}

// ============================================================================
// 5. BACKTEST CORE
// ============================================================================
fn backtest_core(bars: &[Bar], sig: &[i8], cfg: &Config, use_sl: bool) -> (Vec<Trade>, Metrics, Vec<f64>, Vec<f64>) {
    let n = bars.len();
    let fee_rate = cfg.fee_rate();
    let funding_rate = cfg.funding_rate();
    // Forex switches sizing to 1 R-unit, slip to pip-distance, SL/TP to pip-distance.
    let position_size = if cfg.use_forex { 1.0 } else { cfg.position_size };
    let pip_size = cfg.pip_size;
    let slip = if cfg.use_forex { cfg.slippage_pct * pip_size } else { cfg.slip() };
    // surface SL sweep substitutes a per-cell stop via cfg.sl_override
    // (always None on default/parity paths, so this is a no-op there).
    let sl_base = cfg.sl_override.unwrap_or(SL_PERCENTAGE);
    let sl_perc = if cfg.use_forex { sl_base * pip_size } else { sl_base };
    let tp_perc = if cfg.use_forex { cfg.tp_percentage * pip_size } else { cfg.tp_percentage };
    let rrr_fx = if cfg.use_forex && sl_perc > 0.0 { tp_perc / sl_perc } else { 1.0 };
    let stop_pips_fx = if cfg.use_forex { sl_perc / pip_size } else { 1.0 };

    // Closure to compute exit PnL, forex uses pip-based capped R units;
    // crypto uses dollar qty * price diff. Both deduct entry+exit fees and
    // (crypto only) any accumulated funding.
    let pnl_for = |side: i8, entry_p: f64, exit_p: f64,
                    qty_v: f64, fee_e: f64, fee_x: f64, funding_v: f64| -> f64 {
        if cfg.use_forex {
            let price_move = if side == 1 { exit_p - entry_p } else { entry_p - exit_p };
            let price_move_pips = price_move / pip_size;
            let trade_res = ((price_move_pips / (rrr_fx * stop_pips_fx)) * rrr_fx)
                .max(-1.0).min(rrr_fx);
            trade_res * position_size - (fee_e + fee_x)
        } else {
            let direction = if side == 1 { exit_p - entry_p } else { entry_p - exit_p };
            if cfg.clamp_results && sl_perc > 0.0 {
                // Crypto-path R-clamp. 1R per-trade = qty * entry_p * sl_perc/100.
                // Mirrors the forex closure's trade_res clip so gap-prone
                // instruments cannot exceed [-1R, +RRR*R] regardless of how
                // far past SL/TP the bar opens.
                let rrr_crypto = tp_perc / sl_perc;
                let r_size_per_unit = entry_p * (sl_perc / 100.0);
                let r_value = direction / r_size_per_unit;
                let clipped = r_value.max(-1.0).min(rrr_crypto);
                clipped * qty_v * r_size_per_unit - (fee_e + fee_x + funding_v)
            } else {
                qty_v * direction - (fee_e + fee_x + funding_v)
            }
        }
    };

    let funding_mask: Vec<bool> = if cfg.use_forex {
        vec![false; bars.len()]
    } else {
        bars.iter().map(|b| {
            let (h, m) = utc_hour_minute(b.time_unix);
            m == 0 && (h == 0 || h == 8 || h == 16)
        }).collect()
    };

    let in_session: Vec<bool> = bars.iter().map(|b| {
        if !cfg.use_sessions { return true; }
        let (h, _m) = ny_hour_minute(b.time_unix);
        h >= cfg.session_start_hour && h < cfg.session_end_hour
    }).collect();
    let session_end_bar: Vec<bool> = {
        let mut out = vec![false; bars.len()];
        if cfg.use_sessions {
            for i in 0..bars.len() {
                let next_in = if i + 1 < bars.len() { in_session[i + 1] } else { false };
                if in_session[i] && !next_in { out[i] = true; }
            }
        }
        out
    };

    let mut trades: Vec<Trade> = Vec::new();
    let mut equity_list: Vec<f64> = vec![cfg.account_size];
    let mut funding_acc = 0.0f64;
    let mut open_pos: i8 = 0;
    let mut ent_bar: i32 = -1;
    let mut entry_price = 0.0f64;
    let mut qty = 0.0f64;
    let mut fee_entry = 0.0f64;

    for idx in 0..n {
        // Match Python's `for idx in session_idxs:` behaviour, out-of-session
        // bars are skipped entirely so neither funding, SL/TP, nor entries fire.
        if cfg.use_sessions && !in_session[idx] { continue; }

        if open_pos != 0 && funding_mask[idx] {
            let fee_f = qty * bars[idx].open * funding_rate;
            funding_acc += fee_f;
            let last = equity_list.len() - 1;
            equity_list[last] -= fee_f;
        }
        let mut code = sig[idx];

        // 200-bar warmup mirrors Python's `if use_regime and idx < 200: continue`
        // in `_backtest_numba_core`. Active only when the regime engine is in
        // use; otherwise this is a no-op. See `Config::use_regime_seg`.
        if cfg.use_regime_seg && idx < 200 { continue; }

        // Session-end handling (roadmap item 08, match Python
        // `_backtest_numba_core` exactly). On the last in-session bar of the
        // NY day: force-close any open position (exit at open) and NEVER
        // open or flip. Python sets `code = 2/4` UNCONDITIONALLY when a
        // position is open (overriding any flip signal on that bar) and
        // `code = 0` when flat (its `if use_sessions and code in (1,3) and
        // end_bar_flag: code = 0`, which blocks a new entry). The prior Rust
        // guards (`code != 3` / `code != 1`) let an opposite-flip entry slip
        // through, and there was no flat-entry block at all, so Rust opened
        // positions on the closing bar that Python suppresses, inflating the
        // trade count under the forex+session combo.
        let end_bar_flag = cfg.use_sessions && session_end_bar[idx];
        if end_bar_flag {
            code = if open_pos == 1 { 2 } else if open_pos == -1 { 4 } else { 0 };
        } else if cfg.max_hold_bars > 0 && open_pos != 0 && code != 2 && code != 4 {
            // hold-period force-close (only when not a session-end
            // bar, session-end takes precedence, matching Python's elif).
            let held = (idx as i64) - (ent_bar as i64);
            if held >= cfg.max_hold_bars as i64 {
                code = if open_pos == 1 { 2 } else { 4 };
            }
        }
        let price_open = bars[idx].open;

        // SL/TP check, additive in forex (sl_perc/tp_perc are pre-converted
        // to pip-distance fractions when use_forex), multiplicative in crypto.
        // Skipped on session-end bars (Python's `not end_bar_flag`): the
        // force-close above already exits the position there.
        if open_pos != 0 && code != 1 && code != 3 && !end_bar_flag {
            let (sl_pr, tp_pr) = if cfg.use_forex {
                let sl = if open_pos == 1 { entry_price - sl_perc } else { entry_price + sl_perc };
                let tp = if open_pos == 1 { entry_price + tp_perc } else { entry_price - tp_perc };
                (sl, tp)
            } else {
                let sl = if open_pos == 1 { entry_price * (1.0 - sl_perc/100.0) }
                         else { entry_price * (1.0 + sl_perc/100.0) };
                let tp = if open_pos == 1 { entry_price * (1.0 + tp_perc/100.0) }
                         else { entry_price * (1.0 - tp_perc/100.0) };
                (sl, tp)
            };
            let hit_sl = if open_pos == 1 { bars[idx].low <= sl_pr } else { bars[idx].high >= sl_pr };
            let mut hit_tp = if open_pos == 1 { bars[idx].high >= tp_pr } else { bars[idx].low <= tp_pr };
            if hit_sl && hit_tp { hit_tp = false; }
            let is_sl_hit = if use_sl && hit_sl { Some(true) }
                            else if cfg.use_tp && hit_tp { Some(false) }
                            else { None };
            if let Some(sl_hit) = is_sl_hit {
                let raw_exit = if sl_hit { sl_pr } else { tp_pr };
                let exit_price = if open_pos == 1 { raw_exit * (1.0 - slip) }
                                 else { raw_exit * (1.0 + slip) };
                let fee_exit = qty * exit_price * fee_rate;
                // For intrabar SL/TP hits in forex mode Python hard-codes
                // pnl to -1 (SL) or +RRR (TP) in R-units rather than running
                // the trade_res formula on the slippage-adjusted exit price.
                // Match that exactly, otherwise Rust's formula-based exit
                // accumulates a small slippage-floor bias across many trades.
                let pnl = if cfg.use_forex {
                    let r_unit = if sl_hit { -1.0 } else { rrr_fx };
                    r_unit * position_size - (fee_entry + fee_exit)
                } else {
                    pnl_for(open_pos, entry_price, exit_price, qty, fee_entry, fee_exit, funding_acc)
                };
                // cost decomposition (capture funding before reset).
                let (fee_v, slip_v, fund_v, gross_v) = decompose_costs(
                    open_pos, entry_price, exit_price, qty,
                    fee_entry, fee_exit, funding_acc, slip,
                    cfg.use_forex, pnl);
                funding_acc = 0.0;
                let tgid = trades.len() as u64;
                trades.push(Trade { side: open_pos, entry_idx: ent_bar, exit_idx: idx as i32,
                    entry_price, exit_price, qty, pnl, leg_id: 0, trade_group_id: tgid,
                    fee: fee_v, slippage: slip_v, funding: fund_v,
                    gross_pnl: gross_v, net_pnl: pnl });
                let last_eq = *equity_list.last().unwrap();
                equity_list.push(last_eq + pnl);
                open_pos = 0;
                continue;
            }
        }

        if code == 1 {
            if open_pos == -1 {
                let exit_price = price_open * (1.0 + slip);
                let fee_exit = qty * exit_price * fee_rate;
                let pnl = pnl_for(-1, entry_price, exit_price, qty, fee_entry, fee_exit, funding_acc);
                // cost decomposition.
                let (fee_v, slip_v, fund_v, gross_v) = decompose_costs(
                    -1, entry_price, exit_price, qty,
                    fee_entry, fee_exit, funding_acc, slip,
                    cfg.use_forex, pnl);
                funding_acc = 0.0;
                let tgid = trades.len() as u64;
                trades.push(Trade { side: -1, entry_idx: ent_bar, exit_idx: idx as i32,
                    entry_price, exit_price, qty, pnl, leg_id: 0, trade_group_id: tgid,
                    fee: fee_v, slippage: slip_v, funding: fund_v,
                    gross_pnl: gross_v, net_pnl: pnl });
                let last_eq = *equity_list.last().unwrap();
                equity_list.push(last_eq + pnl);
                open_pos = 0;
            }
            if open_pos == 0 {
                fee_entry = position_size * fee_rate;
                entry_price = price_open * (1.0 + slip);
                qty = position_size / entry_price;
                open_pos = 1; ent_bar = idx as i32;
            }
        } else if code == 3 {
            if open_pos == 1 {
                let exit_price = price_open * (1.0 - slip);
                let fee_exit = qty * exit_price * fee_rate;
                let pnl = pnl_for(1, entry_price, exit_price, qty, fee_entry, fee_exit, funding_acc);
                // cost decomposition.
                let (fee_v, slip_v, fund_v, gross_v) = decompose_costs(
                    1, entry_price, exit_price, qty,
                    fee_entry, fee_exit, funding_acc, slip,
                    cfg.use_forex, pnl);
                funding_acc = 0.0;
                let tgid = trades.len() as u64;
                trades.push(Trade { side: 1, entry_idx: ent_bar, exit_idx: idx as i32,
                    entry_price, exit_price, qty, pnl, leg_id: 0, trade_group_id: tgid,
                    fee: fee_v, slippage: slip_v, funding: fund_v,
                    gross_pnl: gross_v, net_pnl: pnl });
                let last_eq = *equity_list.last().unwrap();
                equity_list.push(last_eq + pnl);
                open_pos = 0;
            }
            if open_pos == 0 {
                fee_entry = position_size * fee_rate;
                entry_price = price_open * (1.0 - slip);
                qty = position_size / entry_price;
                open_pos = -1; ent_bar = idx as i32;
            }
        } else if code == 2 && open_pos == 1 {
            let exit_price = price_open * (1.0 - slip);
            let fee_exit = qty * exit_price * fee_rate;
            let pnl = pnl_for(1, entry_price, exit_price, qty, fee_entry, fee_exit, funding_acc);
            // cost decomposition.
            let (fee_v, slip_v, fund_v, gross_v) = decompose_costs(
                1, entry_price, exit_price, qty,
                fee_entry, fee_exit, funding_acc, slip,
                cfg.use_forex, pnl);
            funding_acc = 0.0;
            let tgid = trades.len() as u64;
            trades.push(Trade { side: 1, entry_idx: ent_bar, exit_idx: idx as i32,
                entry_price, exit_price, qty, pnl, leg_id: 0, trade_group_id: tgid,
                fee: fee_v, slippage: slip_v, funding: fund_v,
                gross_pnl: gross_v, net_pnl: pnl });
            let last_eq = *equity_list.last().unwrap();
            equity_list.push(last_eq + pnl);
            open_pos = 0;
        } else if code == 4 && open_pos == -1 {
            let exit_price = price_open * (1.0 + slip);
            let fee_exit = qty * exit_price * fee_rate;
            let pnl = pnl_for(-1, entry_price, exit_price, qty, fee_entry, fee_exit, funding_acc);
            // cost decomposition.
            let (fee_v, slip_v, fund_v, gross_v) = decompose_costs(
                -1, entry_price, exit_price, qty,
                fee_entry, fee_exit, funding_acc, slip,
                cfg.use_forex, pnl);
            funding_acc = 0.0;
            let tgid = trades.len() as u64;
            trades.push(Trade { side: -1, entry_idx: ent_bar, exit_idx: idx as i32,
                entry_price, exit_price, qty, pnl, leg_id: 0, trade_group_id: tgid,
                fee: fee_v, slippage: slip_v, funding: fund_v,
                gross_pnl: gross_v, net_pnl: pnl });
            let last_eq = *equity_list.last().unwrap();
            equity_list.push(last_eq + pnl);
            open_pos = 0;
        }
    }

    // Force-close open trade at end of data, restored to match v1 framework
    // and Python wrapper semantics (which both force-close on the last bar's
    // open ± slip). The earlier no-force-close behavior gave better live/batch
    // parity but broke 3-way parity with v1 Rust + Python framework on USDJPY
    // (3 specs/15k) and BTC (1228 specs/15k). Restoring v1 semantics keeps the
    // three frameworks fungible at the cost of the synthetic last-bar exit.
    if open_pos != 0 {
        let price_last = bars[n - 1].open;
        let exit_price = if open_pos == 1 { price_last * (1.0 - slip) }
                         else { price_last * (1.0 + slip) };
        let fee_exit = qty * exit_price * fee_rate;
        let pnl = pnl_for(open_pos, entry_price, exit_price, qty, fee_entry, fee_exit, funding_acc);
        let (fee_v, slip_v, fund_v, gross_v) = decompose_costs(
            open_pos, entry_price, exit_price, qty,
            fee_entry, fee_exit, funding_acc, slip,
            cfg.use_forex, pnl);
        let tgid = trades.len() as u64;
        trades.push(Trade { side: open_pos, entry_idx: ent_bar, exit_idx: (n-1) as i32,
            entry_price, exit_price, qty, pnl, leg_id: 0, trade_group_id: tgid,
            fee: fee_v, slippage: slip_v, funding: fund_v,
            gross_pnl: gross_v, net_pnl: pnl });
        let last_eq = *equity_list.last().unwrap();
        equity_list.push(last_eq + pnl);
    }

    // Equity / returns scaling: forex normalises to position_size_fx (=1.0)
    // → eq_frac becomes cumulative R-units starting at 0; crypto normalises
    // to ACCOUNT_SIZE. Matches Python's `if use_forex` branch in numba_core.
    let (eq_frac, rets): (Vec<f64>, Vec<f64>) = if cfg.use_forex {
        let pnls: Vec<f64> = trades.iter().map(|t| t.pnl / position_size).collect();
        let mut eq = vec![0.0f64];
        let mut cum = 0.0;
        for &r in &pnls { cum += r; eq.push(cum); }
        (eq, pnls)
    } else {
        let eq: Vec<f64> = equity_list.iter().map(|e| e / cfg.account_size).collect();
        let r: Vec<f64> = trades.iter().map(|t| t.pnl / cfg.account_size).collect();
        (eq, r)
    };
    let mut metrics = compute_metrics_for(&rets, &eq_frac, cfg.use_forex);
    // Standard (bar-based) Sharpe override. Default sharpe_bar=false leaves the
    // per-trade statistic untouched (bit-identical to prior releases).
    if cfg.sharpe_bar {
        metrics.sharpe = bar_based_sharpe(bars, &trades, cfg);
    }
    (trades, metrics, eq_frac, rets)
}

fn run_backtest(bars: &[Bar], sig: &[i8], cfg: &Config) -> (Vec<Trade>, Metrics, Vec<f64>, Vec<f64>) {
    backtest_core(bars, sig, cfg, cfg.use_sl)
}

#[derive(Clone, Debug)]
pub struct FrozenBacktestResult {
    pub trades: Vec<Trade>,
    pub metrics: Metrics,
    pub equity: Vec<f64>,
    pub returns: Vec<f64>,
}

/// Run one frozen strategy specification on the supplied bars.
///
/// This entrypoint does not optimise parameters. The caller provides the
/// strategy function and fixed lookback, so bar-permutation runs regenerate
/// signals from every permuted OHLCV path instead of replaying old signals.
pub fn run_frozen_backtest(
    bars: &[Bar],
    cfg: &Config,
    lookback: usize,
    signal_fn: RawSignalsFn,
) -> Result<FrozenBacktestResult, String> {
    if bars.len() < 2 {
        return Err("a frozen backtest requires at least 2 bars".into());
    }
    let raw = signal_fn(bars, lookback);
    if raw.len() != bars.len() {
        return Err(format!(
            "strategy returned {} signals for {} bars",
            raw.len(),
            bars.len()
        ));
    }
    if raw.iter().any(|signal| !matches!(*signal, -1..=1)) {
        return Err("raw strategy signals must be -1, 0 or 1".into());
    }
    let signals = parse_signals_for(&raw, bars, cfg);
    let (trades, metrics, equity, returns) = run_backtest(bars, &signals, cfg);
    Ok(FrozenBacktestResult { trades, metrics, equity, returns })
}

/// Benchmark-only access to the existing signal-driven core with SL/TP
/// exits disabled. The `benchmark-internal` feature keeps this adapter out
/// of the default public API while allowing external engines to consume the
/// same precomputed event codes and execution contract.
#[cfg(feature = "benchmark-internal")]
#[doc(hidden)]
pub fn benchmark_backtest(
    bars: &[Bar],
    sig: &[i8],
    cfg: &Config,
) -> (Vec<Trade>, Metrics, Vec<f64>, Vec<f64>) {
    let mut cfg = cfg.clone();
    cfg.use_tp = false;
    backtest_core(bars, sig, &cfg, false)
}

/// cost decomposition for a single trade leg. Returns
/// `(fee, slippage, funding, gross_pnl)` satisfying
/// `gross_pnl - fee - slippage - funding == pnl` to fp tolerance.
/// Crypto recovers raw prices via the inverse slippage adjust:
///   `raw_entry = entry_price / (1 + slip)`  for long entry
///   `raw_exit  = exit_price  / (1 - slip)`  for long exit
///   (mirror for short)
/// Forex returns `slippage = funding = 0` and `gross_pnl = pnl + fee`
/// since R-unit pnl already folds slippage and funding into trade_res.
fn decompose_costs(
    side_val: i8,
    entry_price: f64,
    exit_price: f64,
    qty: f64,
    fee_entry: f64,
    fee_exit: f64,
    funding_acc: f64,
    slip: f64,
    use_forex: bool,
    pnl: f64,
) -> (f64, f64, f64, f64) {
    let fee_total = fee_entry + fee_exit;
    if use_forex {
        return (fee_total, 0.0, 0.0, pnl + fee_total);
    }
    let (raw_entry, raw_exit) = if side_val == 1 {
        (entry_price / (1.0 + slip), exit_price / (1.0 - slip))
    } else {
        (entry_price / (1.0 - slip), exit_price / (1.0 + slip))
    };
    let slippage = qty * slip * (raw_entry + raw_exit);
    let funding = funding_acc;
    let gross_pnl = pnl + fee_total + slippage + funding;
    (fee_total, slippage, funding, gross_pnl)
}

pub fn compute_metrics_for(rets: &[f64], eq_frac: &[f64], use_forex: bool) -> Metrics {
    let tc = rets.len();
    if tc == 0 {
        let mut m = Metrics::default();
        m.pf = f64::INFINITY;
        return m;
    }
    let wr = rets.iter().filter(|&&r| r > 0.0).count() as f64 / tc as f64;
    let roi = if use_forex { *eq_frac.last().unwrap() } else { eq_frac.last().unwrap() - 1.0 };
    let wins_sum: f64 = rets.iter().filter(|&&r| r > 0.0).sum();
    let losses_sum: f64 = rets.iter().filter(|&&r| r <= 0.0).map(|r| -r).sum();
    let pf = if losses_sum > 0.0 { wins_sum / losses_sum } else { f64::INFINITY };
    let wins_count = rets.iter().filter(|&&r| r > 0.0).count();
    let losses_count = rets.iter().filter(|&&r| r <= 0.0).count();
    let mw = if wins_count > 0 { wins_sum / wins_count as f64 } else { 0.0 };
    let ml = if losses_count > 0 { losses_sum / losses_count as f64 } else { 0.0 };
    let exp = mw * wr - ml * (1.0 - wr);
    let mean: f64 = rets.iter().sum::<f64>() / tc as f64;
    let variance: f64 = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / tc as f64;
    let std = variance.sqrt();
    let shp = if tc > 1 && std > 0.0 { mean / std * (tc as f64).sqrt() } else { 0.0 };
    let mut hw = vec![0.0f64; eq_frac.len()];
    hw[0] = eq_frac[0];
    for i in 1..eq_frac.len() { hw[i] = hw[i-1].max(eq_frac[i]); }
    let dd = if use_forex {
        (0..eq_frac.len()).map(|i| hw[i] - eq_frac[i]).fold(0.0f64, f64::max)
    } else {
        (0..eq_frac.len()).map(|i| if hw[i] > 0.0 { (hw[i]-eq_frac[i])/hw[i] } else { 0.0 }).fold(0.0f64, f64::max)
    };
    // Daniel Gatto's project-specific Consistency example weights 5
    // chronological return segments and blends them with total ROI.
    let w = [0.0117, 0.0317, 0.0861, 0.2341, 0.6364];
    let segments = split_into_5(rets);
    let seg_sums: Vec<f64> = segments.iter().map(|s| s.iter().sum::<f64>()).collect();
    let weighted: f64 = w.iter().zip(seg_sums.iter()).map(|(wi, si)| wi * si).sum();
    let consistency = 0.6 * weighted + 0.4 * roi;
    Metrics { trades: tc, roi, pf, win_rate: wr, exp, sharpe: shp, max_drawdown: dd, consistency, rrr: None }
}

/// Standard calendar-time Sharpe on the per-bar mark-to-market equity curve.
/// 1:1 with Python `_bar_based_sharpe` (parity): equity at every bar is
/// account + realised PnL of closed trades + unrealised MtM of the open
/// position; per-bar returns are compounding (crypto) or additive R (forex),
/// annualised by periods-per-Julian-year from the median bar spacing.
fn bar_based_sharpe(bars: &[Bar], trades: &[Trade], cfg: &Config) -> f64 {
    let n = bars.len();
    if n < 3 || trades.is_empty() { return 0.0; }
    let mut realized = vec![0.0f64; n];
    let mut unreal = vec![0.0f64; n];
    let (sl_dist, rrr) = if cfg.use_forex {
        let sd = SL_PERCENTAGE * cfg.pip_size;
        let r = if SL_PERCENTAGE != 0.0 { cfg.tp_percentage / SL_PERCENTAGE } else { 1.0 };
        (sd, r)
    } else { (0.0, 1.0) };
    for t in trades {
        if t.exit_idx < t.entry_idx { continue; }
        let e = t.entry_idx as usize;
        let x = t.exit_idx as usize;
        if x < n { for k in x..n { realized[k] += t.pnl; } }
        let d = if t.side == 1 { 1.0 } else { -1.0 };
        for i in e..x.min(n) {
            unreal[i] = if cfg.use_forex {
                let r = if sl_dist != 0.0 { d * (bars[i].close - t.entry_price) / sl_dist } else { 0.0 };
                r.max(-1.0).min(rrr) * cfg.position_size
            } else {
                d * t.qty * (bars[i].close - t.entry_price)
            };
        }
    }
    let mut ret: Vec<f64> = Vec::with_capacity(n);
    if cfg.use_forex {
        for i in 1..n { ret.push((realized[i] + unreal[i]) - (realized[i-1] + unreal[i-1])); }
    } else {
        let a = cfg.account_size;
        for i in 1..n {
            let prev = a + realized[i-1] + unreal[i-1];
            ret.push(((a + realized[i] + unreal[i]) - prev) / prev);
        }
    }
    let ret: Vec<f64> = ret.into_iter().filter(|r| r.is_finite()).collect();
    if ret.len() < 2 { return 0.0; }
    let mean = ret.iter().sum::<f64>() / ret.len() as f64;
    let var = ret.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ret.len() as f64;
    let std = var.sqrt();
    if std <= 0.0 { return 0.0; }
    let mut diffs: Vec<i64> = (1..n).map(|i| bars[i].time_unix - bars[i-1].time_unix).collect();
    diffs.sort_unstable();
    let m = diffs.len();
    let sec = if m == 0 { 0.0 }
        else if m % 2 == 1 { diffs[m/2] as f64 }
        else { (diffs[m/2 - 1] + diffs[m/2]) as f64 / 2.0 };
    let ppy = if sec > 0.0 { 31_557_600.0 / sec } else { 1.0 };
    mean / std * ppy.sqrt()
}

fn split_into_5(arr: &[f64]) -> Vec<Vec<f64>> {
    let n = arr.len();
    let mut result = Vec::with_capacity(5);
    let mut start = 0;
    for k in 0..5usize {
        let end = start + (n + k) / 5;
        result.push(arr[start..end].to_vec());
        start = end;
    }
    result
}

// ============================================================================
// 6. OPTIMISER
// ============================================================================
fn lookback_range() -> Vec<usize> {
    let lo = (DEFAULT_LB as f64 * 0.25) as usize;
    let hi = (DEFAULT_LB as f64 * 1.5) as usize + 1;
    (lo..hi).collect()
}

fn optimiser(bars: &[Bar], cfg: &mut Config, sig_fn: RawSignalsFn) -> (Option<usize>, Metrics) {
    let all_lbs = lookback_range();
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let mut eval_cache: HashMap<usize, Option<(f64, usize, Metrics)>> = HashMap::new();

    let evaluate = |lb: usize, cfg: &mut Config, cache: &mut HashMap<usize, Option<(f64, usize, Metrics)>>| -> Option<(f64, usize, Metrics)> {
        if let Some(cached) = cache.get(&lb) { return cached.clone(); }
        let raw = sig_fn(bars, lb);
        let sig = parse_signals_for(&raw, bars, cfg);
        let met = if !OPTIMIZE_RRR {
            let (_, m, _, _) = run_backtest(bars, &sig, cfg);
            m
        } else {
            let old_tp = cfg.tp_percentage;
            let old_use = cfg.use_tp;
            cfg.tp_percentage = 5.0 * SL_PERCENTAGE;
            cfg.use_tp = true;
            let (probe_trades, _, _, _) = run_backtest(bars, &sig, cfg);
            let mut peak_rs: Vec<f64> = Vec::new();
            let mut close_rs_vec: Vec<f64> = Vec::new();
            // Forex-mode mirror: Python's SL_PERCENTAGE global has been
            // multiplied by PIP_SIZE at module-load time, so its forex value
            // is sl_perc_const * pip_size. Apply the same scaling here so the
            // RRR-probe `risk` denominator matches across engines.
            let sl_for_risk = if cfg.use_forex { SL_PERCENTAGE * cfg.pip_size } else { SL_PERCENTAGE };
            for t in &probe_trades {
                let e = t.entry_idx as usize;
                let x = t.exit_idx as usize;
                if e >= close.len() || x >= close.len() { continue; }
                let ep = close[e];
                let risk = ep * sl_for_risk / 100.0;
                if risk == 0.0 { continue; }
                // RRR-probe side branching. Pre-v0.2.5 the engine used
                // `side == 'long'` (str compared to int8, always false), so
                // ALL trades took the short branch. Default now is the
                // corrected `side == 1` test; cfg.legacy_side_bug = true
                // reverts to the buggy code path for bit-equality with prior
                // research.
                let is_long = if cfg.legacy_side_bug { false } else { t.side == 1 };
                let (peak_r, close_r) = if is_long {
                    let peak = bars[e..=x].iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
                    (((peak - ep) / risk).min(3.0), (close[x] - ep) / risk)
                } else {
                    let trough = bars[e..=x].iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
                    (((ep - trough) / risk).min(3.0), (ep - close[x]) / risk)
                };
                peak_rs.push(peak_r);
                close_rs_vec.push(close_r);
            }
            let mut best_rrr = 1usize;
            let mut best_sum = f64::NEG_INFINITY;
            for r_target in 1..=3usize {
                let sum: f64 = peak_rs.iter().zip(close_rs_vec.iter())
                    .map(|(&p, &c)| if p >= r_target as f64 { r_target as f64 } else { c }).sum();
                if sum > best_sum { best_sum = sum; best_rrr = r_target; }
            }
            cfg.tp_percentage = best_rrr as f64 * SL_PERCENTAGE;
            let (_, mut m, _, _) = run_backtest(bars, &sig, cfg);
            m.rrr = Some(best_rrr);
            cfg.tp_percentage = old_tp;
            cfg.use_tp = old_use;
            m
        };
        if met.trades < MIN_TRADES { cache.insert(lb, None); return None; }
        if let Some(dd_c) = cfg.dd_constraint() {
            if met.max_drawdown > dd_c { cache.insert(lb, None); return None; }
        }
        let val = if OPT_METRIC == "MaxDrawdown" { -met.get(OPT_METRIC) } else { met.get(OPT_METRIC) };
        let result = Some((val, lb, met));
        cache.insert(lb, result.clone());
        result
    };

    let coarse_lbs: Vec<usize> = all_lbs.iter().step_by(2).copied().collect();
    let mut coarse_results: Vec<(f64, usize, Metrics)> = Vec::new();
    for &lb in &coarse_lbs {
        if let Some(r) = evaluate(lb, cfg, &mut eval_cache) { coarse_results.push(r); }
    }
    if coarse_results.is_empty() {
        println!("No lookback meets drawdown constraint, using raw LB {}", DEFAULT_LB);
        let raw = sig_fn(bars, DEFAULT_LB);
        let sig = parse_signals_for(&raw, bars, cfg);
        let (_, m, _, _) = run_backtest(bars, &sig, cfg);
        return (Some(DEFAULT_LB), m);
    }
    coarse_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let (_, best_lb, _) = coarse_results[0].clone();
    let idx_in_all = all_lbs.iter().position(|&l| l == best_lb).unwrap();
    let mut candidates: Vec<(f64, usize, Metrics)> = vec![coarse_results[0].clone()];
    if idx_in_all > 0 {
        if let Some(r) = evaluate(all_lbs[idx_in_all - 1], cfg, &mut eval_cache) { candidates.push(r); }
    }
    if idx_in_all + 1 < all_lbs.len() {
        if let Some(r) = evaluate(all_lbs[idx_in_all + 1], cfg, &mut eval_cache) { candidates.push(r); }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut selected = candidates[0].clone();
    if SMART_OPTIMIZATION {
        let all_lb_set: HashSet<usize> = all_lbs.iter().copied().collect();
        for cand in &candidates {
            let (_, lb_cand, ref met_cand) = *cand;
            let pf_cand = met_cand.pf;
            let mut ok = true;
            for &delta in &[-1i64, 1i64] {
                let neigh = (lb_cand as i64 + delta) as usize;
                if all_lb_set.contains(&neigh) {
                    if let Some(neigh_res) = evaluate(neigh, cfg, &mut eval_cache) {
                        if pf_cand > 1.10 * neigh_res.2.pf { ok = false; break; }
                    }
                }
            }
            if ok {
                if lb_cand != candidates[0].1 {
                    println!("Smart Optimization: switched from LB {} to LB {} because PF spike exceeded 10% vs neighbors.", candidates[0].1, lb_cand);
                }
                selected = cand.clone();
                break;
            }
        }
    }
    (Some(selected.1), selected.2)
}

// ============================================================================
// 7. MONTE CARLO
// ============================================================================
fn print_monte_carlo(arr: &[f64], cfg: &Config) {
    if arr.is_empty() {
        println!(" Monte Carlo skipped: no return series provided.");
        return;
    }
    let mut result = run_trade_monte_carlo(
        arr,
        cfg.mc_mode,
        cfg.mc_runs,
        cfg.mc_seed,
        cfg.use_forex,
    )
    .unwrap_or_else(|error| panic!("Monte Carlo failed: {error}"));
    println!(
        "\n Monte Carlo ({}, seed={}, runs={})",
        result.mode.as_str(), result.seed, result.completed_runs
    );
    println!(" Sharpe: {}", result.sharpe_convention);
    println!(" Drawdown: {}", result.drawdown_convention);
    for metric in &result.metrics {
        println!(
            "  {:>12}: {:7.3}th percentile (ties={})",
            metric.name, metric.percentile_midrank, metric.ties
        );
    }
    result
        .equity_finals
        .sort_by(|left, right| left.partial_cmp(right).unwrap());
    println!("\n Equity Curve Final Value Percentiles ");
    for percentile in [5usize, 25, 50, 75, 95] {
        let index = ((percentile as f64 / 100.0 * result.equity_finals.len() as f64) as usize)
            .min(result.equity_finals.len() - 1);
        println!("  {:>2}th pct: {:9.4}", percentile, result.equity_finals[index]);
    }
    println!("\nSimulations ending with LOSS:           {:5.1}%", result.loss_percent);
    println!(
        "Simulations max-DD > 80 %:              {:5.1}%\n",
        result.drawdown_over_80_percent
    );
}

// ============================================================================
// 8. PRINTER
// ============================================================================
fn fmt_money(val: f64) -> String {
    let s = format!("{:.2}", val.abs());
    let parts: Vec<&str> = s.split('.').collect();
    let int_part = parts[0];
    let dec_part = parts[1];
    let chars: Vec<char> = int_part.chars().collect();
    let mut result = String::new();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 { result.push(','); }
        result.push(*c);
    }
    if val < 0.0 { format!("-{}.{}", result, dec_part) } else { format!("{}.{}", result, dec_part) }
}

fn prettyprint(tag: &str, m: &Metrics, lb: Option<usize>) {
    let lb_note = if let Some(l) = lb { format!("(LB {}) ", l) } else { String::new() };
    let rrr_note = if let Some(r) = m.rrr { format!("  RRR:{}", r) } else { String::new() };
    if DISPLAY_FOREX.load(Ordering::Relaxed) {
        // Forex: R-unit display, no dollar scaling. Matches Python's
        // FOREX_MODE branch in prettyprint.
        println!("{:>8} {}| Trades:{:4}  ROI:{:7.2}R  PF:{:6.2}  Shp:{:6.2}  Win:{:6.2}%  Exp:{:7.2}R  MaxDD:{:7.2}R{}",
            tag, lb_note, m.trades, m.roi, m.pf, m.sharpe,
            m.win_rate * 100.0, m.exp, m.max_drawdown, rrr_note);
    } else {
        println!("{:>8} {}| Trades:{:4}  ROI:${}  PF:{:6.2}  Shp:{:6.2}  Win:{:6.2}%  Exp:${}  MaxDD:${}{}",
            tag, lb_note, m.trades, fmt_money(m.roi * ACCOUNT_SIZE), m.pf, m.sharpe,
            m.win_rate * 100.0, fmt_money(m.exp * ACCOUNT_SIZE), fmt_money(m.max_drawdown * ACCOUNT_SIZE), rrr_note);
    }
}

// ============================================================================
// 9. EXPORT TRADES
// ============================================================================
fn export_trades(trades: &[Trade], bars: &[Bar], strat: &str, window: &str, sample: &str,
    path: &str, write_header: bool) {
    let _ledger = LedgerGuard::acquire(path);
    let file = if write_header {
        File::create(path).expect("Cannot create export file")
    } else {
        std::fs::OpenOptions::new().append(true).open(path).expect("Cannot open export file")
    };
    let mut writer = BufWriter::new(file);
    if write_header {
        writeln!(writer, "strategy,window,sample,side,entry_time,open_entry,high_entry,low_entry,close_entry,exit_time,open_exit,high_exit,low_exit,close_exit,pnl")
            .expect("Cannot write export header");
    }
    for t in trades {
        let ei = t.entry_idx as usize; let xi = t.exit_idx as usize;
        let side_str = if t.side == 1 { "long" } else { "short" };
        writeln!(writer, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            strat, window, sample, side_str,
            bars[ei].time_unix, bars[ei].open, bars[ei].high, bars[ei].low, bars[ei].close,
            bars[xi].time_unix, bars[xi].open, bars[xi].high, bars[xi].low, bars[xi].close,
            t.pnl).expect("Cannot write trade ledger");
    }
    writer.flush().expect("Cannot flush trade ledger");
}

// ============================================================================
// UTILITY
// ============================================================================
fn drift_entries(sig: &[i8]) -> Vec<i8> {
    let mut out = vec![0i8; sig.len()];
    for (i, &code) in sig.iter().enumerate() {
        if code == 1 || code == 3 { if i + 1 < sig.len() { out[i + 1] = code; } }
        else if code == 2 || code == 4 { out[i] = code; }
    }
    out
}

#[derive(Clone)]
struct RobustnessOpts { fee_mult: f64, slip_mult: f64, drift_on: bool, var_on: bool, news_on: bool }

fn opts_from_flags(flags: &[&str]) -> RobustnessOpts {
    let tokens: Vec<String> = flags.iter().map(|f| f.trim().to_lowercase().replace(' ', "_")).collect();
    RobustnessOpts {
        fee_mult: if tokens.iter().any(|t| t == "fee_shock") { 2.0 } else { 1.0 },
        slip_mult: if tokens.iter().any(|t| t == "slippage_shock") { 3.0 } else { 1.0 },
        drift_on: tokens.iter().any(|t| t == "entry_drift"),
        var_on: tokens.iter().any(|t| t == "indicator_variance"),
        news_on: tokens.iter().any(|t| t == "news_candles_injection"),
    }
}

fn label_from_flags(flags: &[&str]) -> String {
    let parts: Vec<&str> = flags.iter().map(|f| match f.trim().to_lowercase().replace(' ', "_").as_str() {
        "fee_shock" => "FEE", "slippage_shock" => "SLI", "entry_drift" => "ENT",
        "indicator_variance" => "IND", "news_candles_injection" => "NEWS", _ => "???",
    }).collect();
    if parts.is_empty() { "NONE".to_string() } else { parts.join("+") }
}

/// Return a perturbed copy of `bars` where every 500..1000 bars a burst of
/// 1..2 candles gets oversized wicks (~2..5× the average true range over
/// the previous 100 bars). Open/close are unchanged; only high/low stretch.
/// Mirrors `inject_news_candles` in `backtester.py`.
fn inject_news_candles(bars: &[Bar], seed: u64) -> Vec<Bar> {
    let mut out = bars.to_vec();
    if out.is_empty() { return out; }
    let mut rng = StdRng::seed_from_u64(seed);
    let n = out.len();
    let mut i: usize = 0;
    loop {
        i += rng.random_range(500..=1000);
        if i >= n { break; }
        let burst = rng.random_range(1..=2);
        for j in 0..burst {
            let idx = i + j;
            if idx >= n { break; }
            let w_start = idx.saturating_sub(100);
            let mut total = 0.0; let mut count = 0usize;
            for k in w_start..idx {
                total += (out[k].high - out[k].low).abs();
                count += 1;
            }
            let avg_range = if count > 0 { total / count as f64 } else {
                let mut t2 = 0.0; let mut c2 = 0usize;
                for b in &out { t2 += (b.high - b.low).abs(); c2 += 1; }
                if c2 > 0 { t2 / c2 as f64 } else { 0.0 }
            };
            if avg_range == 0.0 || !avg_range.is_finite() { continue; }
            let extent = avg_range * (2.0 + rng.random::<f64>() * 3.0);
            let direction = rng.random_range(0..3);     // 0=up, 1=down, 2=both
            let op = out[idx].open; let cp = out[idx].close;
            let hi = out[idx].high; let lo = out[idx].low;
            let top = op.max(cp).max(hi); let bot = op.min(cp).min(lo);
            match direction {
                0 => { out[idx].high = top + extent; }
                1 => { out[idx].low  = bot - extent; }
                _ => { out[idx].high = top + extent; out[idx].low = bot - extent; }
            }
        }
    }
    out
}

// ============================================================================
// 10. CLASSIC SINGLE RUN
// ============================================================================
/// Result struct returned by [`classic_single_run`]. Public so external
/// drivers (e.g. parallel batch runners) can read the optimised IS / OOS
/// metrics without parsing stdout.
pub struct ClassicResult {
    pub met_is_raw: Metrics, pub eq_is_raw: Vec<f64>,
    pub met_is_opt: Option<Metrics>, pub met_oos_opt: Option<Metrics>,
    pub best_lb: Option<usize>, pub best_rrr: Option<usize>,
}

/// Run one IS / OOS / smart-optimised pass on a pre-loaded bar slice and
/// return its metrics. This is the building block of [`run_cfg`] (which
/// adds robustness + WFO + printing on top); external parallel drivers
/// can call this directly to avoid stdout interleaving across workers.
pub fn classic_single_run(all_bars: &[Bar], cfg: &mut Config, strategy: &str, sig_fn: RawSignalsFn) -> ClassicResult {
    let _ledger = LedgerGuard::acquire(&cfg.export_path);
    let export_path_owned = cfg.export_path.clone();
    let export_path = export_path_owned.as_str();
    if let Err(err) = std::fs::remove_file(export_path) {
        if err.kind() != std::io::ErrorKind::NotFound {
            panic!("Cannot reset trade ledger {}: {}", export_path, err);
        }
    }
    let n = all_bars.len();
    let oos_candles = cfg.oos_candles;
    // Mimic Python iloc negative index wrapping
    let oos_start = python_iloc_idx(n as isize - oos_candles as isize, n);
    let is_start = python_iloc_idx(n as isize - oos_candles as isize - BACKTEST_CANDLES as isize, n);
    let is_bars = &all_bars[is_start..oos_start];
    let oos_bars = &all_bars[oos_start..n];
    let _is_close: Vec<f64> = is_bars.iter().map(|b| b.close).collect();
    let _oos_close: Vec<f64> = oos_bars.iter().map(|b| b.close).collect();

    // RAW baseline
    let raw_is = sig_fn(is_bars, DEFAULT_LB);
    let sig_is = parse_signals_for(&raw_is, is_bars, cfg);
    let (_, met_is_raw, eq_is_raw, rets_is_raw) = run_backtest(is_bars, &sig_is, cfg);
    prettyprint("IS-raw", &met_is_raw, None);

    let raw_oos = sig_fn(oos_bars, DEFAULT_LB);
    let sig_oos = parse_signals_for(&raw_oos, oos_bars, cfg);
    let (_, met_oos_raw, _, _) = run_backtest(oos_bars, &sig_oos, cfg);
    prettyprint("OOS-raw", &met_oos_raw, None);

    println!("\n Replication BEFORE optimisation ");
    for mm in &METRICS_LIST {
        let is_val = met_is_raw.get(mm);
        let oos_val = met_oos_raw.get(mm);
        let r = if is_val != 0.0 { oos_val / is_val } else { f64::NAN };
        println!("  {:>12}: {:6.3}", mm, r);
    }

    // Optimise
    let (best_lb, met_is_opt) = optimiser(is_bars, cfg, sig_fn);

    // emit the baseline IS objective surface (opt-in, default off).
    if cfg.emit_opt_surface {
        let header = !crate::opt_surface::surface_path(cfg).exists();
        crate::opt_surface::emit_surface_classic(is_bars, "baseline", cfg, sig_fn, header);
    }

    if let Some(lb) = best_lb {
        let best_rrr = if OPTIMIZE_RRR { met_is_opt.rrr } else { None };
        let rrr_note = if let Some(r) = best_rrr { format!("  |  Best RRR = {}", r) } else { String::new() };
        println!("\nBest {} look-back = {}{}\n", OPT_METRIC, lb, rrr_note);
        prettyprint("IS-opt", &met_is_opt, Some(lb));

        let old_tp = cfg.tp_percentage; let old_use = cfg.use_tp;
        if let Some(r) = best_rrr { cfg.tp_percentage = r as f64 * SL_PERCENTAGE; cfg.use_tp = true; }

        let raw_is_opt = sig_fn(is_bars, lb);
        let sig_is_opt = parse_signals_for(&raw_is_opt, is_bars, cfg);
        let (tr_is_opt, met_is_opt2, _, rets_is_opt) = run_backtest(is_bars, &sig_is_opt, cfg);

        let raw_oos_opt = sig_fn(oos_bars, lb);
        let sig_oos_opt = parse_signals_for(&raw_oos_opt, oos_bars, cfg);
        let (tr_oos_opt, mut met_oos_opt, _, _) = run_backtest(oos_bars, &sig_oos_opt, cfg);
        if let Some(r) = best_rrr { met_oos_opt.rrr = Some(r); }

        export_trades(&tr_is_opt, is_bars, strategy, &format!("LB{}", lb), "IS-opt", export_path, true);
        prettyprint("OOS-opt", &met_oos_opt, Some(lb));
        export_trades(&tr_oos_opt, oos_bars, strategy, &format!("LB{}", lb), "OOS-opt", export_path, false);

        cfg.tp_percentage = old_tp; cfg.use_tp = old_use;

        println!("\n Replication OOS-opt / IS-opt ");
        for mm in &METRICS_LIST {
            let is_val = met_is_opt2.get(mm);
            let oos_val = met_oos_opt.get(mm);
            let r = if is_val != 0.0 { oos_val / is_val } else { f64::NAN };
            println!("  {:>12}: {:6.3}", mm, r);
        }

        if cfg.use_monte_carlo { print_monte_carlo(&rets_is_opt, cfg); }

        return ClassicResult {
            met_is_raw, eq_is_raw,
            met_is_opt: Some(met_is_opt2), met_oos_opt: Some(met_oos_opt),
            best_lb: Some(lb), best_rrr,
        };
    }

    if cfg.use_monte_carlo { print_monte_carlo(&rets_is_raw, cfg); }
    ClassicResult { met_is_raw, eq_is_raw, met_is_opt: None, met_oos_opt: None, best_lb: None, best_rrr: None }
}

// ============================================================================
// 11. WALK-FORWARD
// ============================================================================
fn run_wfo_window(is_bars: &[Bar], oos_bars: &[Bar], lb: usize, window_tag: &str,
    cfg: &Config, strategy: &str, sig_fn: RawSignalsFn,
    rb_scenarios: &[(String, RobustnessOpts)], export_is: bool,
) -> (Vec<f64>, Vec<f64>) {
    let export_path = cfg.export_path.as_str();

    let raw_is = sig_fn(is_bars, lb);
    let sig_is = parse_signals_for(&raw_is, is_bars, cfg);
    let (tr_is, met_is, eq_is, _) = run_backtest(is_bars, &sig_is, cfg);

    let raw_oos = sig_fn(oos_bars, lb);
    let sig_oos = parse_signals_for(&raw_oos, oos_bars, cfg);
    let (tr_oos, met_oos, _, rets_oos) = run_backtest(oos_bars, &sig_oos, cfg);

    prettyprint(&format!("{} IS", window_tag), &met_is, Some(lb));
    prettyprint(&format!("{} OOS", window_tag), &met_oos, Some(lb));

    let header_needed = !Path::new(export_path).exists();
    let wfo_tag = format!("{}-WFO", strategy);
    if export_is {
        export_trades(&tr_is, is_bars, &wfo_tag, window_tag, "IS", export_path, header_needed);
    }
    let header_needed2 = !Path::new(export_path).exists();
    export_trades(&tr_oos, oos_bars, &wfo_tag, window_tag, "OOS", export_path, if export_is { false } else { header_needed2 });

    // Robustness overlays
    for (label, opts) in rb_scenarios {
        if opts.fee_mult == 1.0 && opts.slip_mult == 1.0 && !opts.drift_on && !opts.var_on && !opts.news_on { continue; }
        let mut cfg_rb = cfg.clone();
        cfg_rb.fee_pct *= opts.fee_mult;
        cfg_rb.slippage_pct *= opts.slip_mult;
        let lb_rb = if opts.var_on {
            let offset: i32 = { let mut r = StdRng::seed_from_u64(IND_VARIANCE_SEED); if r.random::<bool>() { 1 } else { -1 } };
            (lb as i32 + offset).max(1) as usize
        } else { lb };

        let is_owned: Vec<Bar>;
        let oos_owned: Vec<Bar>;
        let is_work: &[Bar] = if opts.news_on {
            is_owned = inject_news_candles(is_bars, NEWS_INJECTION_SEED);
            &is_owned
        } else { is_bars };
        let oos_work: &[Bar] = if opts.news_on {
            oos_owned = inject_news_candles(oos_bars, NEWS_INJECTION_SEED.wrapping_add(1));
            &oos_owned
        } else { oos_bars };

        let raw_is_rb = sig_fn(is_work, lb_rb);
        let mut sig_is_rb = parse_signals_for(&raw_is_rb, is_work, &cfg_rb);
        if opts.drift_on { sig_is_rb = drift_entries(&sig_is_rb); }
        let (_, met_is_rb, _, _) = run_backtest(is_work, &sig_is_rb, &cfg_rb);

        let raw_oos_rb = sig_fn(oos_work, lb_rb);
        let mut sig_oos_rb = parse_signals_for(&raw_oos_rb, oos_work, &cfg_rb);
        if opts.drift_on { sig_oos_rb = drift_entries(&sig_oos_rb); }
        let (_, met_oos_rb, _, _) = run_backtest(oos_work, &sig_oos_rb, &cfg_rb);

        prettyprint(&format!("{} IS+{}", window_tag, label), &met_is_rb, Some(lb_rb));
        prettyprint(&format!("{} OOS+{}", window_tag, label), &met_oos_rb, Some(lb_rb));
    }

    (rets_oos, eq_is)
}

fn walk_forward(all_bars: &[Bar], eq_is_baseline: &[f64], cfg: &mut Config, strategy: &str, sig_fn: RawSignalsFn) {
    let scenarios = robustness_scenarios();
    let items: Vec<_> = scenarios.iter().take(MAX_ROBUSTNESS_SCENARIOS).collect();
    let mut rb_scenarios_parsed: Vec<(String, RobustnessOpts)> = Vec::new();
    for (_name, flags) in &items {
        let opts = opts_from_flags(flags);
        if opts.fee_mult != 1.0 || opts.slip_mult != 1.0 || opts.drift_on || opts.var_on {
            rb_scenarios_parsed.push((label_from_flags(flags), opts));
        }
    }

    let n = all_bars.len();
    let ni = n as i64;
    let oos_candles = cfg.oos_candles as i64;
    // Python: start_total = n - OOS_CANDLES  (can be negative, e.g. 48094-90000 = -41906)
    let start_total: i64 = ni - oos_candles;
    let mut cur_start: i64 = start_total;
    let mut window_no = 1usize;
    let mut all_oos_rets: Vec<f64> = Vec::new();
    let mut eq_is_first: Option<Vec<f64>> = None;

    while cur_start < ni {
        let cur_end: i64 = if WFO_TRIGGER_MODE == "candles" {
            (cur_start + WFO_TRIGGER_VAL as i64).min(ni)
        } else {
            let cs_idx = python_iloc_idx(cur_start as isize, n);
            let is_win_start = cs_idx.saturating_sub(BACKTEST_CANDLES);
            let is_bars_roll = &all_bars[is_win_start..cs_idx];
            let (lb_roll, _) = optimiser(is_bars_roll, cfg, sig_fn);
            if lb_roll.is_none() { break; }
            let lb = lb_roll.unwrap();
            let oos_remaining = &all_bars[cs_idx..n];
            let raw_tmp = sig_fn(oos_remaining, lb);
            let sig_tmp = parse_signals_for(&raw_tmp, oos_remaining, cfg);
            let (tr_tmp, _, _, _) = run_backtest(oos_remaining, &sig_tmp, cfg);
            if tr_tmp.is_empty() { ni }
            else { (cur_start + tr_tmp[WFO_TRIGGER_VAL.min(tr_tmp.len()) - 1].exit_idx as i64 + 1).min(ni) }
        };

        // Python: is_win_start = cur_start - BACKTEST_CANDLES
        // Python: is_df_roll = df.iloc[is_win_start:cur_start]
        // Python: dfo = df.iloc[cur_start:cur_end]
        let is_raw_start = cur_start - BACKTEST_CANDLES as i64;
        let (is_s, is_e) = python_iloc_slice(is_raw_start, cur_start, n);
        let (oos_s, oos_e) = python_iloc_slice(cur_start, cur_end, n);
        let is_bars_roll = &all_bars[is_s..is_e];
        let (lb_roll, _) = optimiser(is_bars_roll, cfg, sig_fn);
        if lb_roll.is_none() { break; }
        let lb = lb_roll.unwrap();
        // emit the per-window IS objective surface (opt-in, default off).
        if cfg.emit_opt_surface {
            let header = !crate::opt_surface::surface_path(cfg).exists();
            crate::opt_surface::emit_surface_classic(
                is_bars_roll, &format!("{:02}", window_no), cfg, sig_fn, header);
        }
        let oos_slice = &all_bars[oos_s..oos_e];

        let (rets_oos, eq_is_window) = run_wfo_window(
            is_bars_roll, oos_slice, lb, &format!("W{:02}", window_no),
            cfg, strategy, sig_fn, &rb_scenarios_parsed, window_no == 1);

        if eq_is_first.is_none() { eq_is_first = Some(eq_is_window); }
        all_oos_rets.extend_from_slice(&rets_oos);
        cur_start = cur_end;
        window_no += 1;
    }

    let eq_seed = eq_is_first.as_deref().unwrap_or(eq_is_baseline);
    let seed_last = *eq_seed.last().unwrap_or(&1.0);
    let cum_oos: f64 = all_oos_rets.iter().sum();
    println!("\n WFO Summary ");
    println!("  Total OOS return segments: {}", all_oos_rets.len());
    println!("  Total OOS ROI: ${:.2}", cum_oos * ACCOUNT_SIZE);
    println!("  Final equity: ${:.2}", (seed_last + cum_oos) * ACCOUNT_SIZE);
}

// ============================================================================
// (benchmark v2): read-only WFO-harvest surface + default signal fn.
// `walk_forward_collect` mirrors `walk_forward` (above) but RETURNS the OOS
// harvest instead of printing the summary, with an EMPTY scenario set (no
// robustness overlays). The harvested baseline rets_oos is overlay-independent,
// so this is harvest-equivalent. No existing caller of `walk_forward` changes.
// ============================================================================

/// Out-of-sample harvest of one rolling walk-forward run (benchmark path).
/// `all_oos_rets` = concatenated OOS per-trade stream; `eq_wfo_oos_fraction` =
/// OOS-only equity fraction (1.0 crypto / 0.0 forex, NOT seed-prefixed, so MDD
/// agrees with the Python runner); `agg` = aggregated Metrics; `per_window_oos`
/// = per-window OOS Metrics (dispersion / window count).
pub struct WfoOut {
    pub all_oos_rets: Vec<f64>,
    pub eq_wfo_oos_fraction: Vec<f64>,
    pub agg: Metrics,
    pub per_window_oos: Vec<Metrics>,
}

/// Like `walk_forward` but RETURNS the OOS harvest (no summary print) and runs
/// the WFO loop with an EMPTY scenario set. Read-only; no engine behaviour change.
pub fn walk_forward_collect(
    all_bars: &[Bar], eq_is_baseline: &[f64], cfg: &mut Config,
    strategy: &str, sig_fn: RawSignalsFn,
) -> WfoOut {
    let _ledger = LedgerGuard::acquire(&cfg.export_path);
    let _ = eq_is_baseline; // OOS-only fraction starts at base0; seed not needed
    let rb_scenarios_parsed: Vec<(String, RobustnessOpts)> = Vec::new(); // EMPTY

    let n = all_bars.len();
    let ni = n as i64;
    let oos_candles = cfg.oos_candles as i64;
    let start_total: i64 = ni - oos_candles;
    let mut cur_start: i64 = start_total;
    let mut window_no = 1usize;
    let mut all_oos_rets: Vec<f64> = Vec::new();
    let mut per_window_oos: Vec<Metrics> = Vec::new();
    let base0 = if cfg.use_forex { 0.0 } else { 1.0 };

    while cur_start < ni {
        let cur_end: i64 = if WFO_TRIGGER_MODE == "candles" {
            (cur_start + WFO_TRIGGER_VAL as i64).min(ni)
        } else {
            let cs_idx = python_iloc_idx(cur_start as isize, n);
            let is_win_start = cs_idx.saturating_sub(BACKTEST_CANDLES);
            let is_bars_roll = &all_bars[is_win_start..cs_idx];
            let (lb_roll, _) = optimiser(is_bars_roll, cfg, sig_fn);
            if lb_roll.is_none() { break; }
            let lb = lb_roll.unwrap();
            let oos_remaining = &all_bars[cs_idx..n];
            let raw_tmp = sig_fn(oos_remaining, lb);
            let sig_tmp = parse_signals_for(&raw_tmp, oos_remaining, cfg);
            let (tr_tmp, _, _, _) = run_backtest(oos_remaining, &sig_tmp, cfg);
            if tr_tmp.is_empty() { ni }
            else { (cur_start + tr_tmp[WFO_TRIGGER_VAL.min(tr_tmp.len()) - 1].exit_idx as i64 + 1).min(ni) }
        };

        let is_raw_start = cur_start - BACKTEST_CANDLES as i64;
        let (is_s, is_e) = python_iloc_slice(is_raw_start, cur_start, n);
        let (oos_s, oos_e) = python_iloc_slice(cur_start, cur_end, n);
        let is_bars_roll = &all_bars[is_s..is_e];
        let (lb_roll, _) = optimiser(is_bars_roll, cfg, sig_fn);
        if lb_roll.is_none() { break; }
        let lb = lb_roll.unwrap();
        let oos_slice = &all_bars[oos_s..oos_e];

        let (rets_oos, _eq_is_window) = run_wfo_window(
            is_bars_roll, oos_slice, lb, &format!("W{:02}", window_no),
            cfg, strategy, sig_fn, &rb_scenarios_parsed, window_no == 1);

        let mut eq_w = vec![base0];
        let mut acc_w = base0;
        for r in &rets_oos { acc_w += r; eq_w.push(acc_w); }
        per_window_oos.push(compute_metrics_for(&rets_oos, &eq_w, cfg.use_forex));

        all_oos_rets.extend_from_slice(&rets_oos);
        cur_start = cur_end;
        window_no += 1;
    }

    let mut eq = vec![base0];
    let mut acc = base0;
    for r in &all_oos_rets { acc += r; eq.push(acc); }
    let agg = compute_metrics_for(&all_oos_rets, &eq, cfg.use_forex);

    WfoOut { all_oos_rets, eq_wfo_oos_fraction: eq, agg, per_window_oos }
}

/// The engine's built-in default raw-signal function (EMA(FAST_EMA_SPAN) x
/// EMA(lb)), the signal the parity harness validates (mirror of
/// src/main.rs::ema_crossover). Exposed read-only so the benchmark runner
/// exercises the real default path on `engine_ema`, not a hand copy.
pub fn default_ema_signal(bars: &[Bar], lb: usize) -> Vec<i8> {
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ema_fast = compute_ema(&close, FAST_EMA_SPAN);
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

// ============================================================================
// ROBUSTNESS TESTS
// ============================================================================
fn run_robustness_tests(all_bars: &[Bar], best_lb: Option<usize>, best_rrr: Option<usize>, cfg: &Config, sig_fn: RawSignalsFn) {
    let scenarios = robustness_scenarios();
    for (name, flags) in scenarios.iter().take(MAX_ROBUSTNESS_SCENARIOS) {
        let opts = opts_from_flags(flags);
        if opts.fee_mult == 1.0 && opts.slip_mult == 1.0 && !opts.drift_on && !opts.var_on && !opts.news_on { continue; }
        let label = label_from_flags(flags);
        println!("\n Robustness Test: {} ({}) ", label, name);
        let mut cfg_rb = cfg.clone();
        cfg_rb.fee_pct *= opts.fee_mult;
        cfg_rb.slippage_pct *= opts.slip_mult;
        let lb = best_lb.unwrap_or(DEFAULT_LB);
        let lb_use = if opts.var_on {
            let offset: i32 = { let mut r = StdRng::seed_from_u64(IND_VARIANCE_SEED); if r.random::<bool>() { 1 } else { -1 } };
            (lb as i32 + offset).max(1) as usize
        } else { lb };
        if let Some(r) = best_rrr { cfg_rb.tp_percentage = r as f64 * SL_PERCENTAGE; cfg_rb.use_tp = true; }

        let n = all_bars.len();
        let oos_candles = cfg.oos_candles;
        let oos_start = python_iloc_idx(n as isize - oos_candles as isize, n);
        let is_start = python_iloc_idx(n as isize - oos_candles as isize - BACKTEST_CANDLES as isize, n);
        let is_bars_view = &all_bars[is_start..oos_start];
        let oos_bars_view = &all_bars[oos_start..n];

        let is_owned: Vec<Bar>;
        let oos_owned: Vec<Bar>;
        let is_bars: &[Bar] = if opts.news_on {
            is_owned = inject_news_candles(is_bars_view, NEWS_INJECTION_SEED);
            &is_owned
        } else { is_bars_view };
        let oos_bars: &[Bar] = if opts.news_on {
            oos_owned = inject_news_candles(oos_bars_view, NEWS_INJECTION_SEED.wrapping_add(1));
            &oos_owned
        } else { oos_bars_view };

        let raw_is = sig_fn(is_bars, lb_use);
        let mut sig_is = parse_signals_for(&raw_is, is_bars, &cfg_rb);
        if opts.drift_on { sig_is = drift_entries(&sig_is); }
        let (_, met_is, _, _) = run_backtest(is_bars, &sig_is, &cfg_rb);

        let raw_oos = sig_fn(oos_bars, lb_use);
        let mut sig_oos = parse_signals_for(&raw_oos, oos_bars, &cfg_rb);
        if opts.drift_on { sig_oos = drift_entries(&sig_oos); }
        let (_, met_oos, _, _) = run_backtest(oos_bars, &sig_oos, &cfg_rb);

        // No lookback note: the reference prints these robustness stages
        // without one (`prettyprint(f"{label} IS", ...)` in __init__.py), and a
        // label that differs cannot be compared tag-to-tag.
        prettyprint(&format!("{} IS", label), &met_is, None);
        prettyprint(&format!("{} OOS1", label), &met_oos, None);
    }
}

// ============================================================================
// MAIN
// ============================================================================
/// Run the full backtester pipeline (IS/OOS baseline + optimiser + robustness
/// + walk-forward) with a user-supplied raw-signals function.
///
/// See `src/main.rs` for the reference EMA-crossover strategy and
/// `examples/atr_cross.rs` for an ATR-cross variant.
pub fn run(bars: &[Bar], strategy: &str, sig_fn: RawSignalsFn) {
    let total_start = Instant::now();
    let bars = age_dataset(bars.to_vec(), AGE_DATASET);
    let mut cfg = Config::new();
    let _ledger = LedgerGuard::acquire(&cfg.export_path);
    DISPLAY_FOREX.store(cfg.use_forex, Ordering::Relaxed);
    let base = classic_single_run(&bars, &mut cfg, strategy, sig_fn);

    println!(" Baseline Optimized Metrics ");
    if let Some(ref met) = base.met_is_opt {
        prettyprint("Baseline IS", met, None);
        if let Some(ref met_oos) = base.met_oos_opt {
            prettyprint("Baseline OOS", met_oos, None);
        }
    }

    run_robustness_tests(&bars, base.best_lb, base.best_rrr, &cfg, sig_fn);

    if USE_WFO {
        println!("\n Running Walk-Forward Windows ");
        walk_forward(&bars, &base.eq_is_raw, &mut cfg, strategy, sig_fn);
    }

    println!("\nTotal runtime: {:.2}s", total_start.elapsed().as_secs_f64());
}

/// Same as [`run`] but accepts a pre-built [`Config`]. Use this to enable
/// forex / session / oos2 modes from a single binary without forking
/// `main.rs`. Mirrors `run_with_regime_cfg` but skips regime segmentation.
pub fn run_cfg(bars: &[Bar], strategy: &str, sig_fn: RawSignalsFn, mut cfg: Config) {
    let _ledger = LedgerGuard::acquire(&cfg.export_path);
    let total_start = Instant::now();
    let bars = age_dataset(bars.to_vec(), AGE_DATASET);
    DISPLAY_FOREX.store(cfg.use_forex, Ordering::Relaxed);
    let base = classic_single_run(&bars, &mut cfg, strategy, sig_fn);

    println!(" Baseline Optimized Metrics ");
    if let Some(ref met) = base.met_is_opt {
        prettyprint("Baseline IS", met, None);
        if let Some(ref met_oos) = base.met_oos_opt {
            prettyprint("Baseline OOS", met_oos, None);
        }
    }

    run_robustness_tests(&bars, base.best_lb, base.best_rrr, &cfg, sig_fn);

    if USE_WFO {
        println!("\n Running Walk-Forward Windows ");
        walk_forward(&bars, &base.eq_is_raw, &mut cfg, strategy, sig_fn);
    }

    println!("\nTotal runtime: {:.2}s", total_start.elapsed().as_secs_f64());
}

/// Convenience: resolve the CSV path from the first CLI arg (or a default),
/// load it, and call [`run`]. Panics with a helpful message if the CSV is missing.
pub fn run_with_csv(default_csv: &str, strategy: &str, sig_fn: RawSignalsFn) {
    let csv_path = std::env::args().nth(1).unwrap_or_else(|| default_csv.to_string());
    if !Path::new(&csv_path).exists() {
        panic!("CSV file not found: {}\n\nPut an OHLC CSV at that path, or pass one as the first CLI arg.\nExpected columns: time (unix seconds),open,high,low,close.", csv_path);
    }
    println!("Loading data from: {}", csv_path);
    let load_start = Instant::now();
    let bars = load_ohlc(&csv_path);
    println!("Loaded {} bars in {:.2}s", bars.len(), load_start.elapsed().as_secs_f64());
    run(&bars, strategy, sig_fn);
}

// ============================================================================
// 12. REGIME SEGMENTATION ENGINE (v0.2.1)
// Mirrors backtester.py's get_regimes / optimize_regimes_sequential /
// create_regime_signals / walk_forward (regime path). User-pluggable via
// RegimeConfig: pass any (labels, detector) pair with 2..=5 labels.
// ============================================================================

/// Default 3-regime EMA-200 / 8-bar consistency detector. Matches Python's
/// `get_regimes`: bar `i` is Uptrend (0) if close[i-1..i-8] all > EMA200[i-1..i-8],
/// Downtrend (1) if all <, else Ranging (2).
pub fn default_regime_detector(bars: &[Bar]) -> Vec<u8> {
    let n = bars.len();
    let mut out = vec![2u8; n];                // default = Ranging
    if n < 9 { return out; }
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ema200 = compute_ema(&close, 200);
    const LEN: usize = 8;
    for i in LEN..n {
        let mut all_up = true; let mut all_dn = true;
        for k in 0..LEN {
            let idx = i - k - 1;               // shift(1): yesterday and back
            // Explicit NaN handling: if either side is NaN (e.g. EMA200 not
            // warmed up), the bar can be neither strictly > nor strictly <
            // EMA200, so it falls through to the default Ranging label. This
            // mirrors Python's NaN comparison semantics (NaN > x and NaN < x
            // are both False) and is now intentional rather than implicit.
            let c = close[idx];
            let e = ema200[idx];
            if c.is_nan() || e.is_nan() {
                all_up = false;
                all_dn = false;
            } else {
                if !(c > e) { all_up = false; }
                if !(c < e) { all_dn = false; }
            }
            if !all_up && !all_dn { break; }
        }
        if all_up      { out[i] = 0; }
        else if all_dn { out[i] = 1; }
    }
    out
}

/// User-supplied regime detector + label set. Length must be in [2, 5].
#[derive(Clone)]
pub struct RegimeConfig {
    pub labels: Vec<String>,
    pub detector: RegimeDetectorFn,
}
impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            labels:  REGIME_LABELS.iter().map(|s| s.to_string()).collect(),
            detector: default_regime_detector,
        }
    }
}
impl RegimeConfig {
    pub fn new(labels: Vec<String>, detector: RegimeDetectorFn) -> Self {
        assert!(labels.len() >= 2 && labels.len() <= 5,
            "RegimeConfig.labels must have length 2..=5, got {}", labels.len());
        Self { labels, detector }
    }
}

/// Build raw +1/-1 signals where the slow EMA span rotates per bar based on
/// the regime label. Mirrors Python's `create_regime_signals`.
fn create_regime_signals_internal(
    close: &[f64], ema20: &[f64],
    best_lbs: &[Option<usize>], regimes: &[u8],
) -> Vec<i8> {
    let n = close.len();
    let mut raw = vec![0i8; n];
    let unique: HashSet<usize> = best_lbs.iter().filter_map(|x| *x).collect();
    let lb_emas: HashMap<usize, Vec<f64>> = unique.iter()
        .map(|&lb| (lb, compute_ema(close, lb))).collect();
    for i in 1..n {
        let r = regimes[i] as usize;
        if r >= best_lbs.len() { continue; }
        let lb = match best_lbs[r] { Some(l) => l, None => continue };
        let slow = match lb_emas.get(&lb) { Some(v) => v, None => continue };
        if ema20[i-1].is_nan() || slow[i-1].is_nan() { continue; }
        raw[i] = if ema20[i-1] > slow[i-1] { 1 } else { -1 };
    }
    raw
}

/// Per-regime LB optimiser. For each regime in turn, sweeps coarse LBs while
/// holding other regimes at their current best (or DEFAULT_LB initially), then
/// fine-tunes the winner by ±1. Mirrors Python's `optimize_regimes_sequential`.
/// Returns `(best_lbs, best_rrrs)` indexed by regime u8.
///
/// When `OPTIMIZE_RRR` is true each LB candidate is scored at its
/// regime-best RRR (probe TP=5×SL → restrict to in-regime trades →
/// pick RRR ∈ {1..5} maximising sum-of-R), exactly mirroring Python's
/// `optimize_regimes_sequential::_evaluate` (peak cap 5.0, range 1..=5,
/// in-regime trade filter on the entry index). Note: the regime-path
/// caps/ranges differ from the classic optimiser (3.0 / 1..=3), that's
/// the Python reference's own choice, faithfully preserved here.
fn optimize_regimes_sequential_rs(
    bars: &[Bar], regimes: &[u8], n_regimes: usize, cfg: &Config,
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut best_lbs: Vec<Option<usize>> = vec![Some(DEFAULT_LB); n_regimes];
    let mut best_rrrs: Vec<Option<usize>> = vec![None; n_regimes];
    // Mirrors Python's regime-path candidate list: range(*LOOKBACK_RANGE)
    // *minus FAST_EMA_SPAN*. The classic optimiser does not apply this
    // exclusion; it lives only in `optimize_regimes_sequential`.
    let all_lbs: Vec<usize> = lookback_range().into_iter()
        .filter(|&lb| lb != FAST_EMA_SPAN).collect();
    if all_lbs.is_empty() { return (vec![None; n_regimes], vec![None; n_regimes]); }
    let coarse_lbs: Vec<usize> = all_lbs.iter().step_by(2).copied().collect();

    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ema20 = compute_ema(&close, 20);

    // Returns (score, rrr_used, met).
    let evaluate = |best_lbs: &[Option<usize>], r: usize, lb: usize, cfg: &Config|
        -> Option<(f64, Option<usize>, Metrics)>
    {
        let mut cand = best_lbs.to_vec(); cand[r] = Some(lb);
        let raw = create_regime_signals_internal(&close, &ema20, &cand, regimes);
        let sig = parse_signals_for(&raw, bars, cfg);

        let (met, rrr_used);
        if !OPTIMIZE_RRR {
            let (_, m, _, _) = run_backtest(bars, &sig, cfg);
            met = m;
            rrr_used = None;
        } else {
            // Probe at TP = 5×SL, restrict R-collection to in-regime trades,
            // pick best RRR ∈ {1..5}, rerun backtest with that RRR.
            let mut cfg_probe = cfg.clone();
            cfg_probe.tp_percentage = 5.0 * SL_PERCENTAGE;
            cfg_probe.use_tp = true;
            let (probe_trades, _, _, _) = run_backtest(bars, &sig, &cfg_probe);

            let mut peak_rs: Vec<f64> = Vec::new();
            let mut close_rs_vec: Vec<f64> = Vec::new();
            // Forex pip-scaling for the SL_PERCENTAGE constant, Python's
            // module-level scaling pre-multiplies it by PIP_SIZE in forex mode.
            let sl_for_risk = if cfg.use_forex { SL_PERCENTAGE * cfg.pip_size } else { SL_PERCENTAGE };
            for t in &probe_trades {
                let e = t.entry_idx as usize;
                let x = t.exit_idx as usize;
                if e >= close.len() || x >= close.len() { continue; }
                if (regimes[e] as usize) != r { continue; }     // in-regime only
                // Python's regime optimiser uses the trade's slippage-adjusted
                // entry / exit prices (`entry`, `exit_p` from the tuple), not
                // raw close[idx]. Classic optimiser uses close[idx] in both
                // engines, that's a separate code path.
                let ep = t.entry_price;
                let xp = t.exit_price;
                let risk = ep * sl_for_risk / 100.0;
                if risk == 0.0 { continue; }
                let is_long = if cfg.legacy_side_bug { false } else { t.side == 1 };
                let (peak_r, close_r) = if is_long {
                    let peak = bars[e..=x].iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
                    (((peak - ep) / risk).min(5.0), (xp - ep) / risk)
                } else {
                    let trough = bars[e..=x].iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
                    (((ep - trough) / risk).min(5.0), (ep - xp) / risk)
                };
                peak_rs.push(peak_r);
                close_rs_vec.push(close_r);
            }

            let chosen_rrr = if peak_rs.is_empty() {
                None
            } else {
                let mut best_r = 1usize;
                let mut best_sum = f64::NEG_INFINITY;
                for r_target in 1..=5usize {
                    let sum: f64 = peak_rs.iter().zip(close_rs_vec.iter())
                        .map(|(&p, &c)| if p >= r_target as f64 { r_target as f64 } else { c })
                        .sum();
                    if sum > best_sum { best_sum = sum; best_r = r_target; }
                }
                Some(best_r)
            };

            let mut cfg_run = cfg.clone();
            if let Some(rv) = chosen_rrr {
                cfg_run.tp_percentage = rv as f64 * SL_PERCENTAGE;
                cfg_run.use_tp = true;
            }
            let (_, mut m, _, _) = run_backtest(bars, &sig, &cfg_run);
            if let Some(rv) = chosen_rrr { m.rrr = Some(rv); }
            met = m;
            rrr_used = chosen_rrr;
        }

        if met.trades < MIN_TRADES { return None; }
        if let Some(dd_c) = cfg.dd_constraint() {
            if met.max_drawdown > dd_c { return None; }
        }
        let val = if OPT_METRIC == "MaxDrawdown" { -met.get(OPT_METRIC) } else { met.get(OPT_METRIC) };
        Some((val, rrr_used, met))
    };

    for r in 0..n_regimes {
        if !regimes.iter().any(|&v| v as usize == r) { continue; }

        let mut coarse: Vec<(f64, usize, Option<usize>)> = Vec::new();
        for &lb in &coarse_lbs {
            if let Some((v, rr, _)) = evaluate(&best_lbs, r, lb, cfg) {
                coarse.push((v, lb, rr));
            }
        }
        if coarse.is_empty() { best_lbs[r] = None; best_rrrs[r] = None; continue; }
        coarse.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let (best_val, best_lb, best_rrr) = coarse[0];

        let mut cands: Vec<(f64, usize, Option<usize>)> = vec![(best_val, best_lb, best_rrr)];
        if let Some(idx) = all_lbs.iter().position(|&l| l == best_lb) {
            for delta in [-1i64, 1i64] {
                let n_idx = idx as i64 + delta;
                if n_idx >= 0 && (n_idx as usize) < all_lbs.len() {
                    let lb = all_lbs[n_idx as usize];
                    if let Some((v, rr, _)) = evaluate(&best_lbs, r, lb, cfg) {
                        cands.push((v, lb, rr));
                    }
                }
            }
        }
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        best_lbs[r]  = Some(cands[0].1);
        best_rrrs[r] = cands[0].2;
    }
    (best_lbs, best_rrrs)
}

fn fmt_lb_tag(best_lbs: &[Option<usize>], labels: &[String]) -> String {
    best_lbs.iter().enumerate()
        .filter_map(|(i, lb)| lb.map(|l| format!("{}:{}", labels[i], l)))
        .collect::<Vec<_>>()
        .join(",")
}

fn prettyprint_str(tag: &str, m: &Metrics, lb_tag: &str) {
    let lb_note = if lb_tag.is_empty() { String::new() } else { format!("(LB {}) ", lb_tag) };
    let rrr_note = if let Some(r) = m.rrr { format!("  RRR:{}", r) } else { String::new() };
    if DISPLAY_FOREX.load(Ordering::Relaxed) {
        println!("{:>8} {}| Trades:{:4}  ROI:{:7.2}R  PF:{:6.2}  Shp:{:6.2}  Win:{:6.2}%  Exp:{:7.2}R  MaxDD:{:7.2}R{}",
            tag, lb_note, m.trades, m.roi, m.pf, m.sharpe,
            m.win_rate * 100.0, m.exp, m.max_drawdown, rrr_note);
    } else {
        println!("{:>8} {}| Trades:{:4}  ROI:${}  PF:{:6.2}  Shp:{:6.2}  Win:{:6.2}%  Exp:${}  MaxDD:${}{}",
            tag, lb_note, m.trades, fmt_money(m.roi * ACCOUNT_SIZE), m.pf, m.sharpe,
            m.win_rate * 100.0, fmt_money(m.exp * ACCOUNT_SIZE), fmt_money(m.max_drawdown * ACCOUNT_SIZE), rrr_note);
    }
}

/// Walk-forward path with regime segmentation. Mirrors the v0.2.0 Python
/// fix exactly: the WFO walk is driven by `WFO_TRIGGER_VAL` (candles) /
/// trade count, NOT by regime change indices. Inside each window the active
/// LB rotates per bar based on the regime label; window boundaries are
/// independent of regime flips.
fn walk_forward_regime(
    all_bars: &[Bar], cfg: &mut Config,
    regime_cfg: &RegimeConfig, eq_is_baseline: &[f64],
) {
    let scenarios = robustness_scenarios();
    let items: Vec<_> = scenarios.iter().take(MAX_ROBUSTNESS_SCENARIOS).collect();
    let mut rb_scenarios_parsed: Vec<(String, RobustnessOpts)> = Vec::new();
    for (_name, flags) in &items {
        let opts = opts_from_flags(flags);
        if opts.fee_mult != 1.0 || opts.slip_mult != 1.0 || opts.drift_on || opts.var_on || opts.news_on {
            rb_scenarios_parsed.push((label_from_flags(flags), opts));
        }
    }

    let regimes_full = (regime_cfg.detector)(all_bars);
    assert_eq!(regimes_full.len(), all_bars.len(),
        "regime detector returned {} labels for {} bars", regimes_full.len(), all_bars.len());
    let n_regimes = regime_cfg.labels.len();

    let n = all_bars.len();
    let ni = n as i64;
    let oos_candles = cfg.oos_candles as i64;
    let start_total: i64 = ni - oos_candles;
    let mut cur_start: i64 = start_total;
    let mut window_no = 1usize;
    let mut all_oos_rets: Vec<f64> = Vec::new();
    let mut eq_is_first: Option<Vec<f64>> = None;

    while cur_start < ni {
        let cur_end: i64 = (cur_start + WFO_TRIGGER_VAL as i64).min(ni);
        let is_raw_start = cur_start - BACKTEST_CANDLES as i64;
        let (is_s, is_e) = python_iloc_slice(is_raw_start, cur_start, n);
        let (oos_s, oos_e) = python_iloc_slice(cur_start, cur_end, n);
        if is_e <= is_s || oos_e <= oos_s { break; }

        let is_bars   = &all_bars[is_s..is_e];
        let oos_bars  = &all_bars[oos_s..oos_e];
        let regimes_is  = &regimes_full[is_s..is_e];
        let regimes_oos = &regimes_full[oos_s..oos_e];

        // Mirrors Python: optimize_regimes_sequential() re-detects regimes
        // *locally* on the IS slice (because EMA_200 with adjust=False on a
        // slice differs from EMA_200 on the full series). The actual IS/OOS
        // run still uses the globally-detected regimes sliced.
        let regimes_is_local = (regime_cfg.detector)(is_bars);
        let (best_lbs, _best_rrrs) = optimize_regimes_sequential_rs(
            is_bars, &regimes_is_local, n_regimes, cfg);
        if best_lbs.iter().all(|x| x.is_none()) { break; }
        let lb_tag = fmt_lb_tag(&best_lbs, &regime_cfg.labels);

        // emit the regime IS surface over the labels the engine TRADES
        // (regimes_is), holding non-swept regimes at the FINAL optimised best_lbs.
        if cfg.emit_opt_surface {
            let header = !crate::opt_surface::surface_path(cfg).exists();
            crate::opt_surface::emit_surface_regime(
                is_bars, regimes_is, &best_lbs, &regime_cfg.labels,
                &format!("{:02}", window_no), cfg, header);
        }

        // IS run (for equity seed and reporting)
        let is_close: Vec<f64> = is_bars.iter().map(|b| b.close).collect();
        let is_ema20 = compute_ema(&is_close, 20);
        let raw_is = create_regime_signals_internal(&is_close, &is_ema20, &best_lbs, regimes_is);
        let sig_is = parse_signals_for(&raw_is, is_bars, cfg);
        let (_, met_is, eq_is, _) = run_backtest(is_bars, &sig_is, cfg);

        // OOS run
        let oos_close: Vec<f64> = oos_bars.iter().map(|b| b.close).collect();
        let oos_ema20 = compute_ema(&oos_close, 20);
        let raw_oos = create_regime_signals_internal(&oos_close, &oos_ema20, &best_lbs, regimes_oos);
        let sig_oos = parse_signals_for(&raw_oos, oos_bars, cfg);
        let (_, met_oos, _, rets_oos) = run_backtest(oos_bars, &sig_oos, cfg);

        prettyprint_str(&format!("W{:02} IS",  window_no), &met_is,  &lb_tag);
        prettyprint_str(&format!("W{:02} OOS", window_no), &met_oos, &lb_tag);

        // Robustness overlays (regime-aware: rotate LBs with ±1 jitter on var_on)
        for (label, opts) in &rb_scenarios_parsed {
            let mut cfg_rb = cfg.clone();
            cfg_rb.fee_pct *= opts.fee_mult;
            cfg_rb.slippage_pct *= opts.slip_mult;
            let lbs_rb: Vec<Option<usize>> = if opts.var_on {
                best_lbs.iter().map(|lb| lb.map(|v| {
                    let off: i32 = { let mut r = StdRng::seed_from_u64(IND_VARIANCE_SEED); if r.random::<bool>() { 1 } else { -1 } };
                    (v as i32 + off).max(1) as usize
                })).collect()
            } else { best_lbs.clone() };
            let lb_tag_rb = fmt_lb_tag(&lbs_rb, &regime_cfg.labels);

            // News injection (replaces bar series with a perturbed copy)
            let is_owned: Vec<Bar>;
            let oos_owned: Vec<Bar>;
            let is_w: &[Bar] = if opts.news_on {
                is_owned = inject_news_candles(is_bars, NEWS_INJECTION_SEED); &is_owned
            } else { is_bars };
            let oos_w: &[Bar] = if opts.news_on {
                oos_owned = inject_news_candles(oos_bars, NEWS_INJECTION_SEED.wrapping_add(1)); &oos_owned
            } else { oos_bars };

            let is_close_rb: Vec<f64> = is_w.iter().map(|b| b.close).collect();
            let is_ema20_rb = compute_ema(&is_close_rb, 20);
            let raw_is_rb = create_regime_signals_internal(&is_close_rb, &is_ema20_rb, &lbs_rb, regimes_is);
            let mut sig_is_rb = parse_signals_for(&raw_is_rb, is_w, &cfg_rb);
            if opts.drift_on { sig_is_rb = drift_entries(&sig_is_rb); }
            let (_, met_is_rb, _, _) = run_backtest(is_w, &sig_is_rb, &cfg_rb);

            let oos_close_rb: Vec<f64> = oos_w.iter().map(|b| b.close).collect();
            let oos_ema20_rb = compute_ema(&oos_close_rb, 20);
            // Renamed from raw_is_rb (was reusing the IS-named variable for the
            // OOS phase, confusing, behaviour unchanged).
            let raw_oos_rb = create_regime_signals_internal(&oos_close_rb, &oos_ema20_rb, &lbs_rb, regimes_oos);
            let mut sig_oos_rb = parse_signals_for(&raw_oos_rb, oos_w, &cfg_rb);
            if opts.drift_on { sig_oos_rb = drift_entries(&sig_oos_rb); }
            let (_, met_oos_rb, _, _) = run_backtest(oos_w, &sig_oos_rb, &cfg_rb);

            prettyprint_str(&format!("W{:02} IS+{}",  window_no, label), &met_is_rb,  &lb_tag_rb);
            prettyprint_str(&format!("W{:02} OOS+{}", window_no, label), &met_oos_rb, &lb_tag_rb);
        }

        if eq_is_first.is_none() { eq_is_first = Some(eq_is); }
        all_oos_rets.extend_from_slice(&rets_oos);
        cur_start = cur_end;
        window_no += 1;
    }

    let eq_seed = eq_is_first.as_deref().unwrap_or(eq_is_baseline);
    let seed_last = *eq_seed.last().unwrap_or(&1.0);
    let cum: f64 = all_oos_rets.iter().sum();
    println!("\n WFO+Regime Summary ");
    println!("  Total OOS return segments: {}", all_oos_rets.len());
    println!("  Total OOS ROI: ${:.2}", cum * ACCOUNT_SIZE);
    println!("  Final equity: ${:.2}", (seed_last + cum) * ACCOUNT_SIZE);
}

/// Run the full pipeline with regime segmentation. Use this entry point
/// instead of `run()` when you have a custom regime detector or label set.
/// The signal function (`sig_fn`) is still consulted for the IS/OOS
/// baseline + classic optimisation phases; the WFO+regime path uses the
/// regime-rotated EMA crossover internally.
pub fn run_with_regime(
    bars: &[Bar], strategy: &str, sig_fn: RawSignalsFn, regime_cfg: RegimeConfig,
) {
    let total_start = Instant::now();
    let bars = age_dataset(bars.to_vec(), AGE_DATASET);
    let mut cfg = Config::new();
    let _ledger = LedgerGuard::acquire(&cfg.export_path);
    cfg.use_regime_seg = true;       // matches Python's global USE_REGIME_SEG flag
    DISPLAY_FOREX.store(cfg.use_forex, Ordering::Relaxed);
    let base = classic_single_run(&bars, &mut cfg, strategy, sig_fn);

    println!(" Baseline Optimized Metrics ");
    if let Some(ref met) = base.met_is_opt { prettyprint("Baseline IS", met, None); }
    if let Some(ref met) = base.met_oos_opt { prettyprint("Baseline OOS", met, None); }
    run_robustness_tests(&bars, base.best_lb, base.best_rrr, &cfg, sig_fn);

    println!("\n Running Walk-Forward Windows (regime-rotated LB) ");
    walk_forward_regime(&bars, &mut cfg, &regime_cfg, &base.eq_is_raw);

    println!("\nTotal runtime: {:.2}s", total_start.elapsed().as_secs_f64());
}


/// Like `run_with_regime` but takes a pre-built `Config` so callers can
/// flip use_forex / use_sessions / use_oos2 / etc. before the engine runs.
pub fn run_with_regime_cfg(
    bars: &[Bar], strategy: &str, sig_fn: RawSignalsFn,
    regime_cfg: RegimeConfig, mut cfg: Config,
) {
    let _ledger = LedgerGuard::acquire(&cfg.export_path);
    let total_start = std::time::Instant::now();
    let bars = age_dataset(bars.to_vec(), AGE_DATASET);
    cfg.use_regime_seg = true;       // matches Python's global USE_REGIME_SEG flag
    DISPLAY_FOREX.store(cfg.use_forex, Ordering::Relaxed);
    let base = classic_single_run(&bars, &mut cfg, strategy, sig_fn);
    println!(" Baseline Optimized Metrics ");
    if let Some(ref met) = base.met_is_opt { prettyprint("Baseline IS", met, None); }
    if let Some(ref met) = base.met_oos_opt { prettyprint("Baseline OOS", met, None); }
    run_robustness_tests(&bars, base.best_lb, base.best_rrr, &cfg, sig_fn);
    println!("\n Running Walk-Forward Windows (regime-rotated LB) ");
    walk_forward_regime(&bars, &mut cfg, &regime_cfg, &base.eq_is_raw);
    println!("\nTotal runtime: {:.2}s", total_start.elapsed().as_secs_f64());
}

#[cfg(test)]
mod export_tests;
