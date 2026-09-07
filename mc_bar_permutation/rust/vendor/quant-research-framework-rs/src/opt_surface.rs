//! IS parameter-robustness isosurface: dense in-sample objective
//! grid emit. OPT-IN, parity-safe. Compiled unconditionally (pure std + f64,
//! zero new crates); only INVOKED when `cfg.emit_opt_surface` is true, so
//! default runs never touch this module and stay byte-identical.
//!
//! Re-implements the optimiser's per-cell evaluate math (src/lib.rs:942-1014
//! classic; :1738-1836 regime) verbatim, same RRR probe, same re-run at the
//! chosen RRR, same metric capture, but over the DENSE lookback range (and an
//! optional SL grid) instead of the sparse coarse+fine subset, keeping every
//! cell. Never calls `optimiser()`; never mutates its behaviour.
//!
//! Schema (identical to backtester/opt_surface.py):
//!   window_idx,regime,lb,rrr,sl_idx,sl,sharpe_mode,roi,pf,sharpe,mdd,n_trades,split
//! `split` is always "IS" (the surface is the IS objective landscape only ,
//! no OOS bars, no look-ahead). `sl_idx` is the integer SL-grid index used as
//! the cross-engine parity join key. `sharpe_mode` records the run's Sharpe
//! convention so a "Sharpe" surface never silently mixes trade/bar.
//!
//! 3-axis SL sweep is CRYPTO-ONLY: forex's module-load pip-scaling of the SL
//! makes a multiplicative SL grid ambiguous across engines, so we reject it.

use std::fs::{File, OpenOptions};
use std::io::Write;

use crate::{Bar, Config, Metrics, RawSignalsFn, parse_signals_for, run_backtest,
            lookback_range, compute_ema, create_regime_signals_internal,
            OPTIMIZE_RRR, SL_PERCENTAGE, FAST_EMA_SPAN};

/// Multiplicative SL grid for 3-axis mode, applied around the base
/// SL_PERCENTAGE. Mirrors `_SL_GRID_MULTIPLIERS` in backtester/opt_surface.py.
pub const SL_GRID_MULTIPLIERS: [f64; 5] = [0.5, 0.75, 1.0, 1.25, 1.5];

pub(crate) fn surface_path(cfg: &Config) -> std::path::PathBuf {
    let ledger = std::path::Path::new(&cfg.export_path);
    let name = ledger.file_name().expect("Trade ledger path needs a filename").to_string_lossy();
    let surface = if name == "trade_list.csv" { "opt_surface.csv".to_owned() }
                  else { format!("{}.opt_surface.csv", name) };
    ledger.with_file_name(surface)
}
const HEADER: &str = "window_idx,regime,lb,rrr,sl_idx,sl,sharpe_mode,roi,pf,sharpe,mdd,n_trades,split";

struct SurfaceRow {
    window_idx: String,
    regime: String,
    lb: usize,
    rrr: usize,      // 0 when OPTIMIZE_RRR off / no RRR chosen
    sl_idx: usize,   // integer SL-grid index (parity join key)
    sl: f64,         // crypto-equivalent SL percentage (pre-pip base * mult)
    roi: f64,
    pf: f64,
    sharpe: f64,
    mdd: f64,
    n_trades: usize,
}

/// f64 -> string. `{:?}` is the shortest round-trippable decimal; parity
/// compares values at rel 1e-3 + abs floor so exact digits aren't load-bearing.
fn fmtf(v: f64) -> String {
    if v.is_nan() { "nan".to_string() }
    else if v.is_infinite() { if v > 0.0 { "inf".to_string() } else { "-inf".to_string() } }
    else { format!("{:?}", v) }
}

fn sharpe_mode_str(cfg: &Config) -> &'static str {
    if cfg.sharpe_bar { "bar" } else { "trade" }
}

fn write_rows(path: &str, rows: &[SurfaceRow], smode: &str, write_header: bool) {
    let mut file = if write_header {
        let mut f = File::create(path).expect("Cannot create opt_surface file");
        writeln!(f, "{}", HEADER).unwrap();
        f
    } else {
        OpenOptions::new().append(true).open(path)
            .expect("Cannot open opt_surface file")
    };
    for r in rows {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},IS",
            r.window_idx, r.regime, r.lb, r.rrr, r.sl_idx, fmtf(r.sl), smode,
            fmtf(r.roi), fmtf(r.pf), fmtf(r.sharpe), fmtf(r.mdd), r.n_trades,
        ).unwrap();
    }
}

/// Classic per-cell metric for one (lb, sl) pair. `sl_override` substitutes the
/// swept SL into BOTH the RRR-probe denominator AND the actual stop-loss
/// (threaded through Config.sl_override into backtest_core). When `None` the
/// base const is used, reproducing the optimiser exactly.
///
/// MIN_TRADES / drawdown filters do NOT drop the cell (we want the full
/// landscape, including rejected basins); they only would have set the cell to
/// None in the optimiser's cache. The argmax cell still matches the optimiser.
fn eval_cell_classic(
    bars: &[Bar], lb: usize, sl_override: Option<f64>,
    cfg: &Config, sig_fn: RawSignalsFn,
) -> (Metrics, usize) {
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let sl_base = sl_override.unwrap_or(SL_PERCENTAGE);

    let raw = sig_fn(bars, lb);
    let sig = parse_signals_for(&raw, bars, cfg);

    if !OPTIMIZE_RRR {
        let mut c = cfg.clone();
        c.sl_override = sl_override;          // sweep the real stop
        let (_, m, _, _) = run_backtest(bars, &sig, &c);
        return (m, 0);
    }

    let mut c = cfg.clone();
    c.sl_override = sl_override;              // real stop + probe both use sl_base
    c.tp_percentage = 5.0 * sl_base;
    c.use_tp = true;
    let (probe_trades, _, _, _) = run_backtest(bars, &sig, &c);

    let mut peak_rs: Vec<f64> = Vec::new();
    let mut close_rs_vec: Vec<f64> = Vec::new();
    let sl_for_risk = if c.use_forex { sl_base * c.pip_size } else { sl_base };
    for t in &probe_trades {
        let e = t.entry_idx as usize;
        let x = t.exit_idx as usize;
        if e >= close.len() || x >= close.len() { continue; }
        let ep = close[e];
        let risk = ep * sl_for_risk / 100.0;
        if risk == 0.0 { continue; }
        let is_long = if c.legacy_side_bug { false } else { t.side == 1 };
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
            .map(|(&p, &cc)| if p >= r_target as f64 { r_target as f64 } else { cc }).sum();
        if sum > best_sum { best_sum = sum; best_rrr = r_target; }
    }
    c.tp_percentage = best_rrr as f64 * sl_base;
    let (_, mut m, _, _) = run_backtest(bars, &sig, &c);
    m.rrr = Some(best_rrr);
    (m, best_rrr)
}

/// Emit the dense classic (non-regime) IS surface for one WFO window.
/// 3-axis SL sweep is crypto-only; in forex it is silently downgraded to the
/// 2-axis single-SL surface (the caller-level guard also rejects it).
pub fn emit_surface_classic(
    is_bars: &[Bar], window_idx: &str, cfg: &Config, sig_fn: RawSignalsFn,
    write_header: bool,
) {
    let all_lbs = lookback_range();
    let do_sl = cfg.emit_opt_surface_sl && !cfg.use_forex;
    let sl_mults: Vec<f64> = if do_sl { SL_GRID_MULTIPLIERS.to_vec() } else { vec![1.0] };
    let smode = sharpe_mode_str(cfg);
    let mut rows: Vec<SurfaceRow> = Vec::with_capacity(all_lbs.len() * sl_mults.len());
    for (si, &mult) in sl_mults.iter().enumerate() {
        let sl_cell = SL_PERCENTAGE * mult;
        let sl_override = if do_sl { Some(sl_cell) } else { None };
        for &lb in &all_lbs {
            let (m, rrr) = eval_cell_classic(is_bars, lb, sl_override, cfg, sig_fn);
            rows.push(SurfaceRow {
                window_idx: window_idx.to_string(), regime: String::new(),
                lb, rrr, sl_idx: si, sl: sl_cell,
                roi: m.roi, pf: m.pf, sharpe: m.sharpe, mdd: m.max_drawdown,
                n_trades: m.trades,
            });
        }
    }
    write_rows(&surface_path(cfg).to_string_lossy(), &rows, smode, write_header);
}

/// Regime per-cell metric for one (regime r, lb, sl) triple, mirror of
/// optimize_regimes_sequential_rs::evaluate (src/lib.rs:1755-1836): RRR cap
/// 5.0 / range 1..=5, in-regime trade filter on entry index, slippage-adjusted
/// entry/exit trade prices. `best_lbs` holds the FINAL optimised LBs for the
/// non-swept regimes (matches what the engine trades).
fn eval_cell_regime(
    bars: &[Bar], regimes: &[u8], best_lbs: &[Option<usize>],
    r: usize, lb: usize, sl_override: Option<f64>, cfg: &Config,
) -> (Metrics, usize) {
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let ema20 = compute_ema(&close, 20);
    let sl_base = sl_override.unwrap_or(SL_PERCENTAGE);

    let mut cand = best_lbs.to_vec();
    cand[r] = Some(lb);
    let raw = create_regime_signals_internal(&close, &ema20, &cand, regimes);
    let sig = parse_signals_for(&raw, bars, cfg);

    if !OPTIMIZE_RRR {
        let mut c = cfg.clone();
        c.sl_override = sl_override;
        let (_, m, _, _) = run_backtest(bars, &sig, &c);
        return (m, 0);
    }

    let mut cfg_probe = cfg.clone();
    cfg_probe.sl_override = sl_override;
    cfg_probe.tp_percentage = 5.0 * sl_base;
    cfg_probe.use_tp = true;
    let (probe_trades, _, _, _) = run_backtest(bars, &sig, &cfg_probe);

    let mut peak_rs: Vec<f64> = Vec::new();
    let mut close_rs_vec: Vec<f64> = Vec::new();
    let sl_for_risk = if cfg.use_forex { sl_base * cfg.pip_size } else { sl_base };
    for t in &probe_trades {
        let e = t.entry_idx as usize;
        let x = t.exit_idx as usize;
        if e >= close.len() || x >= close.len() { continue; }
        if (regimes[e] as usize) != r { continue; }
        let ep = t.entry_price;       // slippage-adjusted, per regime optimiser
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
                .map(|(&p, &cc)| if p >= r_target as f64 { r_target as f64 } else { cc }).sum();
            if sum > best_sum { best_sum = sum; best_r = r_target; }
        }
        Some(best_r)
    };

    let mut cfg_run = cfg.clone();
    cfg_run.sl_override = sl_override;
    if let Some(rv) = chosen_rrr {
        cfg_run.tp_percentage = rv as f64 * sl_base;
        cfg_run.use_tp = true;
    }
    let (_, mut m, _, _) = run_backtest(bars, &sig, &cfg_run);
    if let Some(rv) = chosen_rrr { m.rrr = Some(rv); }
    (m, chosen_rrr.unwrap_or(0))
}

/// Emit the dense regime IS surface for one WFO window: one grid block per
/// (window, present-regime). `best_lbs` = the optimiser's FINAL per-regime
/// pick; `labels` supplies the regime label strings.
pub fn emit_surface_regime(
    is_bars: &[Bar], regimes: &[u8], best_lbs: &[Option<usize>],
    labels: &[String], window_idx: &str, cfg: &Config, write_header: bool,
) {
    let all_lbs: Vec<usize> = lookback_range().into_iter()
        .filter(|&lb| lb != FAST_EMA_SPAN).collect();
    let do_sl = cfg.emit_opt_surface_sl && !cfg.use_forex;
    let sl_mults: Vec<f64> = if do_sl { SL_GRID_MULTIPLIERS.to_vec() } else { vec![1.0] };
    let smode = sharpe_mode_str(cfg);
    let n_regimes = labels.len();
    let mut rows: Vec<SurfaceRow> = Vec::new();
    for r in 0..n_regimes {
        if !regimes.iter().any(|&v| v as usize == r) { continue; }
        let label = &labels[r];
        for (si, &mult) in sl_mults.iter().enumerate() {
            let sl_cell = SL_PERCENTAGE * mult;
            let sl_override = if do_sl { Some(sl_cell) } else { None };
            for &lb in &all_lbs {
                let (m, rrr) = eval_cell_regime(
                    is_bars, regimes, best_lbs, r, lb, sl_override, cfg);
                rows.push(SurfaceRow {
                    window_idx: window_idx.to_string(), regime: label.clone(),
                    lb, rrr, sl_idx: si, sl: sl_cell,
                    roi: m.roi, pf: m.pf, sharpe: m.sharpe, mdd: m.max_drawdown,
                    n_trades: m.trades,
                });
            }
        }
    }
    write_rows(&surface_path(cfg).to_string_lossy(), &rows, smode, write_header);
}
