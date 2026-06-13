"""Item #1 — IS parameter-robustness isosurface: dense in-sample objective grid
emit. OPT-IN, parity-safe.

Re-implements the optimiser's per-cell evaluate math (__init__.py:1697-1776
classic; :2291-2360 regime) verbatim — same RRR probe, same re-run at the chosen
RRR, same met[metric] — over the DENSE lookback range (and an optional SL grid)
instead of the sparse coarse+fine subset, keeping every cell. Never calls
optimiser(); never mutates its behaviour. Gated OFF by default
(EMIT_OPT_SURFACE) so importing this module and leaving the flag off leaves
every existing parity number byte-unchanged.

Schema (identical column order to src/opt_surface.rs):
    window_idx,regime,lb,rrr,sl_idx,sl,sharpe_mode,roi,pf,sharpe,mdd,n_trades,split
`split` is always "IS" (no OOS bars, no look-ahead). `sl_idx` is the integer
SL-grid index = cross-engine parity join key. `sharpe_mode` records the run's
Sharpe convention. The 3-axis SL sweep is CRYPTO-ONLY (forex's module-load
pip-scaling makes a multiplicative SL grid ambiguous across engines).

Default emit format is CSV (works on a stock install; matches Rust). Parquet is
opt-in via EMIT_OPT_SURFACE_FMT=parquet and degrades to CSV if pyarrow is absent.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

SL_GRID_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.5)

_COLS = ["window_idx", "regime", "lb", "rrr", "sl_idx", "sl", "sharpe_mode",
         "roi", "pf", "sharpe", "mdd", "n_trades", "split"]


def _resolve_format() -> str:
    """csv (default) | parquet. Parquet degrades to csv if pyarrow is missing."""
    fmt = os.environ.get("EMIT_OPT_SURFACE_FMT", "csv").lower()
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            sys.stderr.write("opt_surface: pyarrow not installed; writing CSV. "
                             "pip install quant-research-framework[surface]\n")
            return "csv"
        return "parquet"
    return "csv"


def _surface_path(export_path: str, fmt: str) -> str:
    base = os.path.join(os.path.dirname(export_path) or ".", "opt_surface")
    return f"{base}.{'csv' if fmt == 'csv' else 'parquet'}"


def _sharpe_mode(bt) -> str:
    return "bar" if getattr(bt, "SHARPE_MODE", "trade") == "bar" else "trade"


def _eval_cell_classic(bt, df, lb: int, sl_cell: float):
    """Mirror of _optimiser_impl._evaluate (__init__.py:1697-1776) for one
    (lb, sl) cell, with `sl_cell` substituted for SL_PERCENTAGE (sweeps the real
    stop). Python's SL_PERCENTAGE/TP_PERCENTAGE/USE_TP are module globals; we
    save/restore them in a try/finally so a mid-cell raise cannot leak a
    corrupted SL into the next real WFO window. Returns (met, rrr_chosen)."""
    dfi = bt.compute_indicators(df, lb)
    raw = bt.create_raw_signals(dfi, lb)
    bt._runtime_state['last_unfiltered_raw'] = raw.copy()
    sig = bt.parse_signals(raw, dfi['time'])

    g = bt.__dict__
    old_tp, old_tp_flag, old_sl = bt.TP_PERCENTAGE, bt.USE_TP, bt.SL_PERCENTAGE
    try:
        g['SL_PERCENTAGE'] = sl_cell             # sweep the real stop
        if not bt.OPTIMIZE_RRR:
            bt._runtime_state['last_unfiltered_raw'] = raw.copy()
            _, met, _, _, _ = bt.backtest(dfi, sig)
            return met, 0

        g['TP_PERCENTAGE'] = 5 * sl_cell
        g['USE_TP'] = True
        bt._runtime_state['last_unfiltered_raw'] = raw.copy()
        trades_probe, _, _, _, _ = bt.backtest(dfi, sig)

        peak_Rs, close_Rs = [], []
        for side, e, x, *_ in trades_probe:
            entry_price = dfi['close'].iloc[e]
            risk = entry_price * sl_cell / 100.0
            if risk == 0.0:
                continue
            is_long = (side == 'long') if bt.LEGACY_SIDE_BUG else (side == 1)
            if is_long:
                peak_R = (dfi['high'].iloc[e:x + 1].values.max() - entry_price) / risk
                close_R = (dfi['close'].iloc[x] - entry_price) / risk
            else:
                peak_R = (entry_price - dfi['low'].iloc[e:x + 1].values.min()) / risk
                close_R = (entry_price - dfi['close'].iloc[x]) / risk
            peak_Rs.append(min(peak_R, 3.0))
            close_Rs.append(close_R)

        if peak_Rs:
            arr_p, arr_c = np.array(peak_Rs), np.array(close_Rs)
            sums = {R: np.where(arr_p >= R, R, arr_c).sum() for R in range(1, 4)}
            best_rrr = int(max(sums, key=sums.get))
        else:
            best_rrr = 1
        g['TP_PERCENTAGE'] = best_rrr * sl_cell
        bt._runtime_state['last_unfiltered_raw'] = raw.copy()
        _, met, _, _, _ = bt.backtest(dfi, sig)
        met['RRR'] = best_rrr
        return met, best_rrr
    finally:
        g['TP_PERCENTAGE'], g['USE_TP'], g['SL_PERCENTAGE'] = old_tp, old_tp_flag, old_sl


def emit_surface_classic(bt, is_df, window_idx: str, write_header: bool) -> None:
    """Dense classic (non-regime) IS surface for one WFO window. Gated by the
    caller on bt.EMIT_OPT_SURFACE. 3-axis SL sweep is crypto-only."""
    lbs = list(range(*bt.LOOKBACK_RANGE))
    do_sl = bool(bt.EMIT_OPT_SURFACE_SL) and not bt.FOREX_MODE
    sl_mults = SL_GRID_MULTIPLIERS if do_sl else (1.0,)
    smode = _sharpe_mode(bt)
    rows = []
    for si, mult in enumerate(sl_mults):
        sl_cell = bt.SL_PERCENTAGE * mult
        for lb in lbs:
            met, rrr = _eval_cell_classic(bt, is_df, lb, sl_cell)
            rows.append([window_idx, "", lb, rrr, si, sl_cell, smode,
                         met['ROI'], met['PF'], met['Sharpe'],
                         met['MaxDrawdown'], met['Trades'], "IS"])
    _write(bt, rows, write_header)


def _eval_cell_regime(bt, dfi, regimes, ema20, slow_emas, best_lbs,
                      reg, lb: int, sl_cell: float):
    """Mirror of optimize_regimes_sequential._evaluate (__init__.py:2291-2360)
    for one (regime, lb, sl) cell. RRR cap 5.0 / range 1..6, in-regime trade
    filter, slippage-adjusted entry/exit prices. `best_lbs` holds the FINAL
    optimised LBs for non-swept regimes. try/finally restore."""
    temp_lbs = dict(best_lbs)
    temp_lbs[reg] = lb

    # Reuse the optimiser's exact signal construction: ema20 and slow EMAs are
    # already .shift(1) (== Rust's [i-1] indexing).
    raw = np.empty(len(dfi), dtype=np.int8)
    for i, r in enumerate(regimes):
        raw[i] = 1 if ema20[i] > slow_emas[temp_lbs[r]][i] else -1
    bt._runtime_state['last_unfiltered_raw'] = None
    sig = bt.parse_signals(raw, dfi['time'])

    g = bt.__dict__
    old_tp, old_tp_flag, old_sl = bt.TP_PERCENTAGE, bt.USE_TP, bt.SL_PERCENTAGE
    try:
        g['SL_PERCENTAGE'] = sl_cell
        if not bt.OPTIMIZE_RRR:
            _, met, _, _, _ = bt.backtest(dfi, sig)
            return met, 0

        g['TP_PERCENTAGE'] = 5 * sl_cell
        g['USE_TP'] = True
        bt._runtime_state['last_unfiltered_raw'] = None
        trades_p, _, _, _, _ = bt.backtest(dfi, sig)

        peak_Rs, close_Rs = [], []
        for side, ent, exi, entry, exit_p, *_ in trades_p:
            if regimes.iloc[ent] != reg:
                continue
            risk = entry * sl_cell / 100.0
            if risk == 0.0:
                continue
            is_long = (side == 'long') if bt.LEGACY_SIDE_BUG else (side == 1)
            if is_long:
                seg = dfi['high'].iloc[ent:exi + 1].values
                peak_Rs.append(min((seg.max() - entry) / risk, 5.0))
                close_Rs.append((exit_p - entry) / risk)
            else:
                seg = dfi['low'].iloc[ent:exi + 1].values
                peak_Rs.append(min((entry - seg.min()) / risk, 5.0))
                close_Rs.append((entry - exit_p) / risk)

        best_rrr_cand = 0
        if peak_Rs:
            arr_p, arr_c = np.array(peak_Rs), np.array(close_Rs)
            sums = {r: np.where(arr_p >= r, r, arr_c).sum() for r in range(1, 6)}
            best_rrr_cand = int(max(sums, key=sums.get))
            g['TP_PERCENTAGE'] = best_rrr_cand * sl_cell
            g['USE_TP'] = True
        bt._runtime_state['last_unfiltered_raw'] = None
        _, met, _, _, _ = bt.backtest(dfi, sig)
        if best_rrr_cand:
            met['RRR'] = best_rrr_cand
        return met, best_rrr_cand
    finally:
        g['TP_PERCENTAGE'], g['USE_TP'], g['SL_PERCENTAGE'] = old_tp, old_tp_flag, old_sl


def emit_surface_regime(bt, is_df, best_lbs, window_idx: str,
                        write_header: bool) -> None:
    """Dense regime IS surface for one WFO window — one grid block per
    (window, present-regime). `best_lbs` = the optimiser's FINAL per-regime
    pick (threaded from the caller). Rebuilds the same EMA scaffold the regime
    optimiser builds (__init__.py:2261-2280)."""
    dfi = is_df.copy()
    dfi['EMA_20'] = is_df['close'].ewm(span=20, adjust=False).mean()
    dfi['EMA_200'] = is_df['close'].ewm(span=200, adjust=False).mean()
    dfi['EMA_900'] = is_df['close'].ewm(span=900, adjust=False).mean()
    regimes = bt.detect_regimes(dfi)
    ema20 = dfi['EMA_20'].shift(1).values

    lbs_candidates = [lb for lb in range(*bt.LOOKBACK_RANGE) if lb != bt.FAST_EMA_SPAN]
    slow_emas = {
        v: is_df['close'].ewm(span=v, adjust=False).mean().shift(1).values
        for v in lbs_candidates
    }

    regs = list(bt.REGIME_LABELS)
    # best_lbs may arrive as a dict {label: lb}; non-swept regimes held at final.
    final_lbs = {r: best_lbs.get(r, bt.DEFAULT_LB) for r in regs}

    do_sl = bool(bt.EMIT_OPT_SURFACE_SL) and not bt.FOREX_MODE
    sl_mults = SL_GRID_MULTIPLIERS if do_sl else (1.0,)
    smode = _sharpe_mode(bt)

    rows = []
    for reg in regs:
        if not bool((regimes == reg).any()):
            continue
        for si, mult in enumerate(sl_mults):
            sl_cell = bt.SL_PERCENTAGE * mult
            for lb in lbs_candidates:
                met, rrr = _eval_cell_regime(
                    bt, dfi, regimes, ema20, slow_emas, final_lbs,
                    reg, lb, sl_cell)
                rows.append([window_idx, reg, lb, rrr, si, sl_cell, smode,
                             met['ROI'], met['PF'], met['Sharpe'],
                             met['MaxDrawdown'], met['Trades'], "IS"])
    _write(bt, rows, write_header)


def _write(bt, rows: List[list], write_header: bool) -> None:
    fmt = _resolve_format()
    path = _surface_path(bt.EXPORT_PATH, fmt)
    df = pd.DataFrame(rows, columns=_COLS)
    if fmt == "parquet":
        # Single growing file; parquet has no native append. Surfaces are small
        # (64-320 rows/window); read-concat-rewrite is O(windows^2) rows total
        # but tiny in absolute terms. CSV (the default) is O(n) native append.
        if (not write_header) and os.path.exists(path):
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df.to_parquet(path, index=False)
    else:
        # Normalise non-finite tokens to match Rust (nan / inf / -inf).
        out = df.copy()
        for c in ("roi", "pf", "sharpe", "mdd", "sl"):
            out[c] = out[c].map(_finite_token)
        out.to_csv(path, mode="w" if write_header else "a",
                   header=write_header, index=False)


def _finite_token(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if np.isnan(f):
        return "nan"
    if np.isposinf(f):
        return "inf"
    if np.isneginf(f):
        return "-inf"
    return f
