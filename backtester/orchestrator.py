#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-flow orchestration extracted from ``backtester/__init__.py``.

Mirrors the Rust port ``src/orchestrator.rs``. Pure move/re-export: every
function here behaves byte-for-byte as it did in its prior location.

Every module-level global (EXPORT_PATH, OOS_CANDLES, BACKTEST_CANDLES,
USE_WFO, USE_REGIME_SEG, REGIME_LABELS, ROBUSTNESS_SCENARIOS, FEE_PCT,
SLIPPAGE_PCT, TP_PERCENTAGE, USE_TP, ...), the ``signals_cache`` /
``_runtime_state`` scratch dicts, the ``blocked_*`` filter structures,
the ``with_config`` context manager, and every engine function that
STAYS in ``__init__`` (``compute_indicators``, ``create_raw_signals``,
``parse_signals``, ``backtest``, ``create_regime_signals``,
``detect_regimes``, ``filter_raw_signals``, ``evaluate_filters``,
``load_ohlc``, ``in_session``, ``_safe_*``) or that moved to a sibling
submodule (``prettyprint``, ``_metrics_from_trades`` in metrics;
``optimiser``, ``monte_carlo`` in objectives) is accessed LIVE via
``_bt.<name>`` so the ``Config.with_config()`` runtime-mutation contract
keeps working unchanged.

Globals the run flow temporarily rebinds (TP_PERCENTAGE, USE_TP,
FEE_PCT, SLIPPAGE_PCT, OOS_CANDLES) are written with ``_bt.X = ...`` so
the mutation lands on the ``backtester`` module namespace (which the
engine reads back through ``backtest()``), exactly as the old
``globals()['X'] = ...`` did when this code lived in ``__init__``.
"""

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

import backtester as _bt

__all__ = [
    "export_trades",
    "optimise_regime_full",
    "backtest_continuous_regime",
    "optimize_regimes_sequential",
    "_optimize_regimes_sequential_impl",
    "classic_single_run",
    "_classic_single_run_impl",
    "_run_wfo_window",
    "walk_forward",
    "_walk_forward_impl",
    "inject_news_candles",
    "apply_news_injection",
    "_apply_news_injection_impl",
    "drift_entries",
    "apply_combined_robustness",
    "_normalize_rb_flag",
    "_opts_from_flags",
    "_label_from_flags",
    "run_robustness_tests",
    "age_dataset",
    "main",
    "_main_impl",
]


# 9. EXPORT
def export_trades(trades, df, strat, window, sample, path, write_header):
    cols = [
        'strategy','window','sample','side',                  #  added 'side'
        'entry_time','open_entry','high_entry','low_entry','close_entry',
        'exit_time','open_exit','high_exit','low_exit','close_exit',
        'pnl'                                                 #  added 'pnl'
    ]
    t, o, h, l, c = (df[x].values for x in ('time','open','high','low','close'))
    rows = []
    for trade in trades:
        side, ei, xi, _entry_price, _exit_price, _qty, pnl = trade
        rows.append([
            strat,
            window,
            sample,
            side,                                            #  include side
            t[ei], o[ei], h[ei], l[ei], c[ei],
            t[xi], o[xi], h[xi], l[xi], c[xi],
            pnl                                              #  include P&L
        ])
    df_export = pd.DataFrame(rows, columns=cols)
    _bt._safe_append_or_write_trade_csv(df_export, path, write_header)


def optimise_regime_full(df_full, regimes, target_regime, current_lbs,
                         lb_range, metric, min_trades):
    """
    Two-stage regime optimiser:
      1) Coarse search in steps of 2 over lb_range for target_regime,
         holding other regimes at current_lbs.
      2) Pick best coarse lookback, then fine-tune by testing
         best_lb - 1 and best_lb + 1.
      3) Return the best (lb, rrr, metrics) tuple.
    """
    import math
    # Was: `global last_unfiltered_raw`. Moved to _runtime_state.

    # helper: run one candidate lookback and return (val, lb, rrr, met) or None
    def _evaluate(lb):
        # 1) set candidate look-backs
        cand_lbs = current_lbs.copy()
        cand_lbs[target_regime] = lb

        # 2) generate raw signals and parse
        base = _bt.compute_indicators(df_full.copy(), _bt.DEFAULT_LB) if 'EMA_20' not in df_full else df_full.copy()
        raw = _bt.create_regime_signals(base, cand_lbs, regimes)
        _bt._runtime_state['last_unfiltered_raw'] = raw.copy()
        sig = _bt.parse_signals(raw, df_full['time'])

        # 3) backtest (with or without RRR probe)
        if not _bt.OPTIMIZE_RRR:
            _, met, _, _, _ = _bt.backtest(base, sig)
            rrr_used = None
        else:
            # probe at 5R
            tp_old, flag_old = _bt.TP_PERCENTAGE, _bt.USE_TP
            _bt.TP_PERCENTAGE, _bt.USE_TP = 5 * _bt.SL_PERCENTAGE, True
            trades_p, _, _, _, _ = _bt.backtest(base, sig)

            # collect peak/close only for target_regime entries
            peak_Rs, close_Rs = [], []
            for side, ent, exi, *_ in trades_p:
                if regimes.iloc[ent] != target_regime:
                    continue
                entry = base['close'].iloc[ent]
                risk = entry * _bt.SL_PERCENTAGE / 100.0
                is_long = (side == 'long') if _bt.LEGACY_SIDE_BUG else (side == 1)
                if is_long:
                    peak = base['high'].iloc[ent:exi+1].max()
                    close = base['close'].iloc[exi]
                    peak_Rs.append(min((peak - entry) / risk, 5.0))
                    close_Rs.append((close - entry) / risk)
                else:
                    trough = base['low'].iloc[ent:exi+1].min()
                    close = base['close'].iloc[exi]
                    peak_Rs.append(min((entry - trough) / risk, 5.0))
                    close_Rs.append((entry - close) / risk)

            # choose best RRR if we have any trades
            best_rrr_cand = None
            if peak_Rs:
                sums = {
                    r: np.where(np.array(peak_Rs) >= r, r, close_Rs).sum()
                    for r in range(1, 6)
                }
                best_rrr_cand = max(sums, key=sums.get)

            # rerun full backtest with candidate RRR
            if best_rrr_cand is not None:
                _bt.TP_PERCENTAGE, _bt.USE_TP = best_rrr_cand * _bt.SL_PERCENTAGE, True
            _, met, _, _, _ = _bt.backtest(base, sig)
            rrr_used = best_rrr_cand
            if best_rrr_cand is not None:
                met['RRR'] = best_rrr_cand

            # restore TP settings
            _bt.TP_PERCENTAGE, _bt.USE_TP = tp_old, flag_old

        # 4) filter by minimum trades
        if met['Trades'] < min_trades:
            return None

        # 5) Drawdown constraint
        if _bt.dd_constraint is not None and met['MaxDrawdown'] > _bt.dd_constraint:
            return None

        # 6) Compute value
        val = -met[metric] if metric == 'MaxDrawdown' else met[metric]
        return val, lb, met

    # build full and coarse lookback lists
    all_lbs = list(lb_range)
    coarse_lbs = all_lbs[::2]

    # Coarse pass
    coarse_results = [res for lb in coarse_lbs if (res := _evaluate(lb)) is not None]


    # Fallback if none
    if not coarse_results:
        raw_lb = current_lbs[target_regime]
        print(f"No lookback for {target_regime} meets drawdown  {_bt.DRAWDOWN_CONSTRAINT}, using raw LB {raw_lb}")
        # Recompute metrics at raw_lb
        base = (_bt.compute_indicators(df_full.copy(), _bt.DEFAULT_LB)
                if 'EMA_20' not in df_full else df_full.copy())
        raw = _bt.create_regime_signals(base, current_lbs, regimes)
        sig = _bt.parse_signals(raw, df_full['time'])
        _, met_raw, *_ = _bt.backtest(base, sig)
        return raw_lb, None, met_raw


    # sort by val descending
    coarse_results.sort(key=lambda x: x[0], reverse=True)
    best_val, best_lb, best_rrr, best_met = coarse_results[0]

    # 2) Fine pass: test neighbors of best_lb
    idx = all_lbs.index(best_lb)
    candidates = [(best_val, best_lb, best_rrr, best_met)]
    for neighbor_idx in (idx - 1, idx + 1):
        if 0 <= neighbor_idx < len(all_lbs):
            res = _evaluate(all_lbs[neighbor_idx])
            if res is not None:
                candidates.append(res)

    # pick the best overall
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, final_lb, final_rrr, final_met = candidates[0]
    return final_lb, final_rrr, final_met
def backtest_continuous_regime(df, best_lbs):
    """
    Continuous regime-switching backtest with RRR applied exactly
    as in optimiser().
    """
    # 1) compute EMAs once
    dfi = df.copy()
    for span in (20, 200, 900):
        dfi[f'EMA_{span}'] = dfi['close'].ewm(span=span, adjust=False).mean()
    for lb in set(best_lbs.values()):
        if lb is not None:
            dfi[f'EMA_{lb}'] = dfi['close'].ewm(span=lb, adjust=False).mean()

    # 2) label regimes
    regimes = _bt.detect_regimes(dfi)

    # 3) build full raw & filter
    ema20    = dfi['EMA_20'].shift(1)
    ema_lbs  = {r: dfi[f'EMA_{lb}'].shift(1) for r, lb in best_lbs.items()}
    raw_full = np.zeros(len(dfi), dtype=np.int8)
    for reg, lb in best_lbs.items():
        m = regimes == reg
        raw_full[m] = np.where(ema20[m] > ema_lbs[reg][m], 1, -1)
    raw_full = _bt.filter_raw_signals(raw_full, regimes)

    # 4) RRR optimisation on the full signal (exactly like optimiser)
    if _bt.OPTIMIZE_RRR:
        old_tp, old_flag = _bt.TP_PERCENTAGE, _bt.USE_TP
        _bt.TP_PERCENTAGE = 5 * _bt.SL_PERCENTAGE
        _bt.USE_TP        = True

        _bt._runtime_state['last_unfiltered_raw'] = raw_full.copy()
        trades_probe, _, _, _, _ = _bt.backtest(dfi, _bt.parse_signals(raw_full, dfi['time']))

        # compute peak_Rs & close_Rs
        peak_Rs, close_Rs = [], []
        for side, e, x, _, _, *_ in trades_probe:
            entry = dfi['close'].iloc[e]
            risk  = entry * _bt.SL_PERCENTAGE/100
            is_long = (side == 'long') if _bt.LEGACY_SIDE_BUG else (side == 1)
            if is_long:
                seg       = dfi['high'].iloc[e:x+1].values
                peak_R    = (seg.max() - entry) / risk
                close_R   = (dfi['close'].iloc[x] - entry) / risk
            else:
                seg       = dfi['low'].iloc[e:x+1].values
                peak_R    = (entry - seg.min()) / risk
                close_R   = (entry - dfi['close'].iloc[x]) / risk

            peak_Rs.append(min(peak_R, 5.0))
            close_Rs.append(close_R)

        arr_peak  = np.array(peak_Rs)
        arr_close = np.array(close_Rs)
        sums = {
            R: np.where(arr_peak>=R, float(R), arr_close).sum()
            for R in range(1,6)
        }
        best_rrr = max(sums, key=sums.get)

        # apply it
        _bt.TP_PERCENTAGE = best_rrr * _bt.SL_PERCENTAGE
        _bt.USE_TP        = True

        # restore after final backtest below
    else:
        best_rrr = None
        old_tp, old_flag = _bt.TP_PERCENTAGE, _bt.USE_TP

    # 5) final full-series backtest with RRR applied
    _bt._runtime_state['last_unfiltered_raw'] = raw_full.copy()
    sig = _bt.parse_signals(raw_full, dfi['time'])
    trades, metrics, eq_frac, rets, _ = _bt.backtest(dfi, sig)
    if best_rrr is not None:
        metrics['RRR'] = best_rrr

    # restore globals
    _bt.TP_PERCENTAGE = old_tp
    _bt.USE_TP        = old_flag


    return trades, metrics, eq_frac, rets

def optimize_regimes_sequential(is_df, config: "Optional[_bt.Config]" = None):
    """
    Sequential per-regime LB optimisation with coarse/fine search (and optional
    RRR optimisation). Iterates over `REGIME_LABELS` in order; on phase k the
    LB for label k is searched while LBs for labels 0..k-1 are held at their
    previously chosen values and labels k+1.. are held at DEFAULT_LB.
    Returns:
      best_lbs:  dict {label: lb}        (one entry per REGIME_LABELS)
      best_rrrs: dict {label: rrr | None}
    Works with REGIME_LABELS of any length in {2, 3, 4, 5}.

    `config` is optional; see `optimiser` docstring for the contract.
    """
    with _bt.with_config(config):
        return _optimize_regimes_sequential_impl(is_df)


def _optimize_regimes_sequential_impl(is_df):
    import math
    # Was: `global last_unfiltered_raw`. Moved to _runtime_state.

    dfi = is_df.copy()
    dfi['EMA_20']  = is_df['close'].ewm(span=20,  adjust=False).mean()
    dfi['EMA_200'] = is_df['close'].ewm(span=200, adjust=False).mean()
    dfi['EMA_900'] = is_df['close'].ewm(span=900, adjust=False).mean()
    regimes = _bt.detect_regimes(dfi)

    ema20 = dfi['EMA_20'].shift(1).values

    lbs_candidates = [
        lb for lb in range(*_bt.LOOKBACK_RANGE)
        if lb != _bt.FAST_EMA_SPAN
    ]
    slow_emas = {
        lb: is_df['close'].ewm(span=lb, adjust=False).mean().shift(1).values
        for lb in lbs_candidates
    }

    regs = list(_bt.REGIME_LABELS)
    best_lbs  = {r: _bt.DEFAULT_LB for r in regs}
    best_rrrs = {r: None       for r in regs}

    # Phase-by-phase search
    for reg in regs:
        print(f" Phase: optimize {reg}")

        # helper to evaluate one lookback for this regime
        def _evaluate(lb):
            # build lookback map with candidate for this regime
            temp_lbs = best_lbs.copy()
            temp_lbs[reg] = lb

            # construct raw signal array
            raw = np.empty(len(dfi), dtype=np.int8)
            for i, r in enumerate(regimes):
                chosen_lb = temp_lbs[r]
                raw[i]    = 1 if ema20[i] > slow_emas[chosen_lb][i] else -1

            _bt._runtime_state['last_unfiltered_raw'] = None
            sig = _bt.parse_signals(raw, dfi['time'])

            # backtest (with or without RRR probe)
            if _bt.OPTIMIZE_RRR:
                old_tp, old_tp_flag = _bt.TP_PERCENTAGE, _bt.USE_TP
                _bt.TP_PERCENTAGE = 5 * _bt.SL_PERCENTAGE
                _bt.USE_TP        = True

                _bt._runtime_state['last_unfiltered_raw'] = None
                trades_p, _, _, _, _ = _bt.backtest(dfi, sig)

                # collect peak/close R for this target regime
                peak_Rs, close_Rs = [], []
                for side, ent, exi, entry, exit_p, *_ in trades_p:
                    if regimes.iloc[ent] != reg:
                        continue
                    risk = entry * _bt.SL_PERCENTAGE / 100.0
                    is_long = (side == 'long') if _bt.LEGACY_SIDE_BUG else (side == 1)
                    if is_long:
                        seg = dfi['high'].iloc[ent:exi+1].values
                        peak_Rs.append(min((seg.max() - entry) / risk, 5.0))
                        close_Rs.append((exit_p - entry) / risk)
                    else:
                        seg = dfi['low'].iloc[ent:exi+1].values
                        peak_Rs.append(min((entry - seg.min()) / risk, 5.0))
                        close_Rs.append((entry - exit_p) / risk)

                # choose best RRR if any trades
                best_rrr_cand = None
                if peak_Rs:
                    arr_peak = np.array(peak_Rs)
                    arr_close = np.array(close_Rs)
                    sums = {
                        r: np.where(arr_peak >= r, r, arr_close).sum()
                        for r in range(1, 6)
                    }
                    best_rrr_cand = max(sums, key=sums.get)

                # rerun backtest with chosen RRR
                if best_rrr_cand is not None:
                    _bt.TP_PERCENTAGE = best_rrr_cand * _bt.SL_PERCENTAGE
                    _bt.USE_TP        = True

                _bt._runtime_state['last_unfiltered_raw'] = None
                trades, met, eq, rets, _ = _bt.backtest(dfi, sig)
                if best_rrr_cand is not None:
                    met['RRR'] = best_rrr_cand

                # restore TP settings
                _bt.TP_PERCENTAGE = old_tp
                _bt.USE_TP        = old_tp_flag

                rrr_used = best_rrr_cand
            else:
                _bt._runtime_state['last_unfiltered_raw'] = None
                trades, met, eq, rets, _ = _bt.backtest(dfi, sig)
                rrr_used = None

            # skip too-few trades
            if met['Trades'] < _bt.MIN_TRADES:
                return None

            # compute score (maximize)
            score = -met[_bt.OPT_METRIC] if _bt.OPT_METRIC == 'MaxDrawdown' else met[_bt.OPT_METRIC]
            return score, lb, rrr_used, met

        # build coarse subset and run coarse pass
        coarse_lbs = lbs_candidates[::2]
        coarse_results = []
        for lb in coarse_lbs:
            res = _evaluate(lb)
            if res is not None:
                coarse_results.append(res)

        if not coarse_results:
            print(f"{reg:>9} no valid lookbacks!")
            best_lbs[reg] = None
            best_rrrs[reg] = None
            continue

        coarse_results.sort(key=lambda x: x[0], reverse=True)
        best_score, best_lb_coarse, best_rrr_coarse, best_met_coarse = coarse_results[0]

        # fine-tune: test neighbors of best coarse
        idx = lbs_candidates.index(best_lb_coarse)
        candidates = [(best_score, best_lb_coarse, best_rrr_coarse, best_met_coarse)]
        for neighbor_idx in (idx - 1, idx + 1):
            if 0 <= neighbor_idx < len(lbs_candidates):
                res = _evaluate(lbs_candidates[neighbor_idx])
                if res is not None:
                    candidates.append(res)

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, final_lb, final_rrr, final_met = candidates[0]

        # save and report
        best_lbs[reg]  = final_lb
        best_rrrs[reg] = final_rrr
        rrr_info = f" | RRR={final_rrr}" if final_rrr is not None else ""
        print(f"{reg:>9} best LB = {final_lb} | {_bt.OPT_METRIC}: {final_met[_bt.OPT_METRIC]:.4f}{rrr_info}")

    return best_lbs, best_rrrs

## 10. CLASSIC SINGLE-RUN  updated so that RRR is optimized and applied per regime
# 10. CLASSIC SINGLE-RUN  corrected to collect OOS-regime returns by segment
def classic_single_run(df, config: "Optional[_bt.Config]" = None):
    """Single-window IS/OOS run with optional RRR optimisation.

    `config` is optional; see `optimiser` docstring for the contract.
    """
    with _bt.with_config(config):
        return _classic_single_run_impl(df)


def _classic_single_run_impl(df):
    # Was: `global TP_PERCENTAGE, USE_TP, signals_cache` — redundant.
    # TP/USE_TP rebinds inside this function are done via `_bt.X=...`
    # (the only path that the engine respects when reading them back through
    # `backtest()` -> `_backtest_numba_core`); signals_cache is mutated
    # in-place. Removed in v0.4.0.
    m1 = None
    m2 = None
    m1r = None
    m2r = None

    _bt.signals_cache['mode'] = 'classic'   # <-- add this

    if os.path.exists(_bt.EXPORT_PATH):
        _bt._safe_remove_trade_csv(_bt.EXPORT_PATH)
    first_export = True


    N = len(df)
    is_df  = df.iloc[N - _bt.OOS_CANDLES - _bt.BACKTEST_CANDLES : N - _bt.OOS_CANDLES].reset_index(drop=True)
    oos_df = df.iloc[N - _bt.OOS_CANDLES : N].reset_index(drop=True)

    # ---------- RAW baseline (console-only) --------------------------------
    for tag, subset in [('IS-raw', is_df), ('OOS-raw', oos_df)]:
        # 1) compute indicators
        dfi = _bt.compute_indicators(subset, _bt.DEFAULT_LB)
        # 2) build raw signals
        raw = _bt.create_raw_signals(dfi, _bt.DEFAULT_LB)
        # 3) reset true-raw buffer so exits align with this segment
        # Was: `global last_unfiltered_raw; last_unfiltered_raw = None`
        _bt._runtime_state['last_unfiltered_raw'] = None
        # 4) parse signals (entries + exits)
        sig = _bt.parse_signals(raw, dfi['time'])
        # 5) backtest
        tr, m, eq, rets, _ = _bt.backtest(dfi, sig)

        # --- print & export ---
        if tag == 'IS-raw':
            _bt.prettyprint('IS-raw', m)
        else:  # OOS-raw
            if _bt.USE_OOS2:
                # split trades into OOS1 vs OOS2
                o1 = [t for t in tr if t[2] < _bt.ORIGINAL_OOS]
                o2 = [t for t in tr if t[2] >= _bt.ORIGINAL_OOS]
                m1 = _bt._metrics_from_trades(o1)
                m2 = _bt._metrics_from_trades(o2)
                _bt.prettyprint('OOS1-raw', m1)
                _bt.prettyprint('OOS2-raw', m2)
            else:
                _bt.prettyprint('OOS-raw', m)

        # store IS / OOS metrics & equity
        if tag == 'IS-raw':
            tr_is_raw,   met_is_raw,   eq_is_raw,   rets_is_raw   = tr, m, eq, rets
        else:
            tr_oos_raw,  met_oos_raw,  eq_oos_raw,  rets_oos_raw  = tr, m, eq, rets
    if _bt.PRINT_EQUITY_CURVE: import numpy as np, matplotlib.pyplot as plt; plt.figure(figsize=(10,5)); plt.plot((np.concatenate((eq_is_raw, eq_oos_raw + (eq_is_raw[-1]-1))) * _bt.ACCOUNT_SIZE), label='RAW Equity'); plt.title('RAW Equity Curve'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    print("\n Replication BEFORE optimisation ")
    for mm in _bt.METRICS:
        r = met_oos_raw[mm] / met_is_raw[mm] if met_is_raw[mm] else math.nan
        print(f"  {mm:>12}: {r:6.3f}")

    # ---------- CASE A  regime segmentation ------------------------------
    if _bt.USE_REGIME_SEG and not _bt.USE_WFO:
        # 1) find best LBs per regime on IS
        dfi_full   = _bt.compute_indicators(is_df, _bt.DEFAULT_LB)
        regimes_is = _bt.detect_regimes(dfi_full)
        best_lbs, best_rrrs = _bt.optimize_regimes_sequential(is_df)

        # report RRRs
        print("\n Best RRR per regime ")
        for reg in _bt.REGIME_LABELS:
            rrr = best_rrrs.get(reg)
            print(f"  {reg:>9}: RRR = {rrr if rrr is not None else 'None'}")

        # 2) IS backtest by regime
        tr_is_reg, met_is_reg, eq_is_reg, rets_is_reg = \
            _bt.backtest_continuous_regime(is_df, best_lbs)
        print("\nIS-regime backtest")
        _bt.prettyprint('IS-reg', met_is_reg)

        if not _bt.USE_WFO:
            _bt.export_trades(
                tr_is_reg, is_df,
                'EMA-crossover-regime',      # strategy name
                f"LB-regime",                # window label, you can customize
                'IS-reg',                    # sample label
                _bt.EXPORT_PATH,
                first_export                 # True on very first call
            )
            first_export = False

        # 3) apply filters if enabled
        _bt.evaluate_filters(tr_is_reg, rets_is_reg,
                         _bt.detect_regimes(_bt.compute_indicators(is_df, _bt.DEFAULT_LB)))
        print("\n Filter Conclusion for OOS:")
        if _bt.FILTER_REGIMES:
            print(f"  Regimes removed: {', '.join(_bt.blocked_regimes) or 'None'}")
        if _bt.FILTER_DIRECTIONS:
            print(f"  Directions removed: {', '.join(_bt.blocked_directions) or 'None'}")
        if _bt.FILTER_REGIMES and _bt.FILTER_DIRECTIONS and _bt.blocked_pairs:
            print("  Regime-Direction pairs removed:")
            for r, dirs in _bt.blocked_pairs.items():
                if dirs:
                    print(f"    {r}: {', '.join(dirs)}")

        # 4) OOS backtest by regime
        tr_oos_reg, met_oos_reg, eq_oos_reg, rets_oos_reg = _bt.backtest_continuous_regime(oos_df, best_lbs)
        print("\nOOS-regime backtest")
        if _bt.USE_OOS2:
            # split into first vs. second window
            o1r = [t for t in tr_oos_reg if t[2] < _bt.ORIGINAL_OOS]
            o2r = [t for t in tr_oos_reg if t[2] >= _bt.ORIGINAL_OOS]
            m1r = _bt._metrics_from_trades(o1r)
            m2r = _bt._metrics_from_trades(o2r)
            _bt.prettyprint('OOS1-reg', m1r)
            _bt.prettyprint('OOS2-reg', m2r)
        else:
            _bt.prettyprint('OOS-reg', met_oos_reg)

        # 5) count how many of those OOS trades exited in the first ORIGINAL_OOS bars
        if _bt.USE_OOS2:
            n_oos1_trades = sum(
                1
                for side, ent, exit_i, *rest in tr_oos_reg
                if exit_i < _bt.ORIGINAL_OOS
            )
            o1r = [t for t in tr_oos_reg if t[2] < _bt.ORIGINAL_OOS]
            o2r = [t for t in tr_oos_reg if t[2] >= _bt.ORIGINAL_OOS]
            if not _bt.USE_WFO:
                _bt.export_trades(o1r, oos_df, 'EMA-crossover-regime', 'LB-regime', 'OOS1-reg', _bt.EXPORT_PATH, first_export)
                _bt.export_trades(o2r, oos_df, 'EMA-crossover-regime', 'LB-regime', 'OOS2-reg', _bt.EXPORT_PATH, first_export)
        else:
            n_oos1_trades = None
            if not _bt.USE_WFO:
                _bt.export_trades(tr_oos_reg, oos_df, 'EMA-crossover-regime', 'LB-regime', 'OOS-reg', _bt.EXPORT_PATH, first_export)

        # 6) build equity for plotting
        if _bt.PRINT_EQUITY_CURVE:
            # 1) offset OOS so it starts where IS left off
            offset = eq_is_reg[-1] - 1
            eq_baseline_oos = eq_oos_reg + offset
            # 2) stitch IS + OOS into one curve
            eq_baseline = np.concatenate((eq_is_reg, eq_baseline_oos))
        else:
            eq_baseline = None


        # cache for robustness
        _bt.signals_cache['mode']     = 'regime'
        _bt.signals_cache['is_df']    = is_df
        _bt.signals_cache['oos_df']   = oos_df
        _bt.signals_cache['best_lbs'] = best_lbs

        # 7) compute trade-based splits
        split1 = len(eq_is_reg) - 1
        split2 = (split1 + n_oos1_trades) if n_oos1_trades is not None else None

        # --- 4) Monte Carlo on optimised IS ---
        if _bt.USE_MONTE_CARLO:
            _bt.monte_carlo(rets_is_reg, met_is_reg, _bt.MC_RUNS)

        # 8) return everything
        return {
            'met_is':          met_is_reg,
            'met_oos_reg':     met_oos_reg,
            'eq_is':           eq_is_reg,
            'met_is_opt':      None,
            'eq_is_reg':       eq_is_reg,
            'eq_oos_reg':      eq_oos_reg,
            'eq_baseline_is':  eq_is_reg,
            'eq_baseline_oos': eq_baseline_oos,
            'n_oos1_trades':   n_oos1_trades,
            'split1':          split1,
            'split2':          split2,
            'met_oos1_reg':     m1r,
            'met_oos2_reg':     m2r,
            'eq_baseline':     eq_baseline,

        }


    # ---------- CASE B  classic optimisation (no regime segment) ----------
    best_lb, met_is_opt = _bt.optimiser(is_df, range(*_bt.LOOKBACK_RANGE), _bt.OPT_METRIC, _bt.MIN_TRADES)

    # Item #1: emit the baseline IS objective surface (opt-in, default off).
    if _bt.EMIT_OPT_SURFACE:
        import backtester as _bt2
        from backtester import opt_surface as _osf
        _hdr = not os.path.exists(_osf._surface_path(_bt.EXPORT_PATH, _osf._resolve_format()))
        _osf.emit_surface_classic(_bt2, is_df, "baseline", write_header=_hdr)

    if best_lb:
        best_rrr = met_is_opt.get('RRR') if _bt.OPTIMIZE_RRR else None
        rrr_note = f"  |  Best RRR = {best_rrr}" if best_rrr is not None else ""
        print(f"\nBest {_bt.OPT_METRIC} look-back = {best_lb}{rrr_note}\n")
        _bt.prettyprint('IS-opt', met_is_opt, best_lb)

        # --- 1) IS-opt run ---
        tp_old, tp_flag_old = _bt.TP_PERCENTAGE, _bt.USE_TP
        if best_rrr is not None:
            _bt.TP_PERCENTAGE = best_rrr * _bt.SL_PERCENTAGE
            _bt.USE_TP        = True

        dfi_is_opt = _bt.compute_indicators(is_df, best_lb)
        raw_is_opt = _bt.create_raw_signals(dfi_is_opt, best_lb)
        raw_is_opt = _bt.filter_raw_signals(raw_is_opt, None)
        sig_is_opt = _bt.parse_signals(raw_is_opt, dfi_is_opt['time'])
        tr_is_opt, met_is_opt, eq_is_opt, rets_is_opt, _ = _bt.backtest(dfi_is_opt, sig_is_opt)

        _bt.TP_PERCENTAGE = tp_old
        _bt.USE_TP        = tp_flag_old

        # --- 2) OOS-opt run ---
        tp_old, tp_flag_old = _bt.TP_PERCENTAGE, _bt.USE_TP
        if best_rrr is not None:
            _bt.TP_PERCENTAGE = best_rrr * _bt.SL_PERCENTAGE
            _bt.USE_TP        = True

        dfo_opt     = _bt.compute_indicators(oos_df, best_lb)
        raw_tmp     = _bt.create_raw_signals(dfo_opt, best_lb)
        raw_tmp     = _bt.filter_raw_signals(raw_tmp, None)
        sig_oos_opt = _bt.parse_signals(raw_tmp, dfo_opt['time'])
        tr_oos_opt, met_oos_opt, eq_oos_opt, rets_oos_opt, _ = _bt.backtest(dfo_opt, sig_oos_opt)
        if best_rrr is not None:
            met_oos_opt['RRR'] = best_rrr

        if not _bt.USE_WFO:
            _bt.export_trades(tr_is_opt, dfi_is_opt,
                          'EMA-crossover', f'LB{best_lb}', 'IS-opt',
                          _bt.EXPORT_PATH, first_export)
            first_export = False

        # split into OOS1 vs OOS2
        if _bt.USE_OOS2:
            o1 = [t for t in tr_oos_opt if t[2] < _bt.ORIGINAL_OOS]
            o2 = [t for t in tr_oos_opt if t[2] >= _bt.ORIGINAL_OOS]
            n_oos1_trades = len(o1)

            m1 = _bt._metrics_from_trades(o1)
            m2 = _bt._metrics_from_trades(o2)
            print(f"\nOOS1 back-test (first {_bt.ORIGINAL_OOS} bars, LB{best_lb})")
            _bt.prettyprint('OOS1-opt', m1, best_lb)
            print(f"\nOOS2 back-test (last {_bt.ORIGINAL_OOS} bars, LB{best_lb})")
            _bt.prettyprint('OOS2-opt', m2, best_lb)
            if not _bt.USE_WFO:
                _bt.export_trades(o1, dfo_opt,
                              'EMA-crossover', f'LB{best_lb}', 'OOS1-opt',
                              _bt.EXPORT_PATH, first_export)
                _bt.export_trades(o2, dfo_opt,
                              'EMA-crossover', f'LB{best_lb}', 'OOS2-opt',
                              _bt.EXPORT_PATH, first_export)
        else:
            _bt.prettyprint('OOS-opt', met_oos_opt, best_lb)
            n_oos1_trades = None
            if not _bt.USE_WFO:
                _bt.export_trades(tr_oos_opt, dfo_opt,
                              'EMA-crossover', f'LB{best_lb}', 'OOS-opt',
                              _bt.EXPORT_PATH, first_export)

        _bt.TP_PERCENTAGE = tp_old
        _bt.USE_TP        = tp_flag_old

        print("\n Replication OOS-opt / IS-opt ")
        for mm in _bt.METRICS:
            r = met_oos_opt[mm] / met_is_opt[mm] if met_is_opt[mm] else math.nan
            print(f"  {mm:>12}: {r:6.3f}")

        # --- 3) Build equity curve for plotting (additive, not multiplicative) ---
        if _bt.PRINT_EQUITY_CURVE:
            # 3.a) combine IS and OOS trades in chronological order
            trades_all = tr_is_opt + tr_oos_opt

            # 3.b) extract pnl from each trade tuple
            pnl_list = [trade[-1] for trade in trades_all]  # last element is pnl

            # 3.c) cumulative-sum into an equity curve
            if _bt.FOREX_MODE:
                # pnl_list is in R-units; equity starts at 0R
                eq_baseline = np.concatenate(([0.0], np.cumsum(pnl_list)))
            else:
                # pnl_list is in USD; equity starts at ACCOUNT_SIZE
                eq_usd      = np.concatenate(([_bt.ACCOUNT_SIZE], _bt.ACCOUNT_SIZE + np.cumsum(pnl_list)))
                eq_baseline = eq_usd / _bt.ACCOUNT_SIZE
        else:
            eq_baseline = None

        # --- 4) Monte Carlo on optimised IS ---
        if _bt.USE_MONTE_CARLO:
            _bt.monte_carlo(rets_is_opt, met_is_opt, _bt.MC_RUNS)

        _bt.signals_cache['best_lb'] = best_lb
        _bt.signals_cache['best_rrr'] = best_rrr
        _bt.signals_cache['sig_is'] = sig_is_opt
        _bt.signals_cache['dfi_is'] = dfi_is_opt
        _bt.signals_cache['sig_oos'] = sig_oos_opt
        _bt.signals_cache['dfo_oos'] = dfo_opt

        split1 = len(eq_is_opt) - 1
        split2 = split1 + n_oos1_trades if n_oos1_trades is not None else None


        return {
            'met_is':         met_is_raw,
            'eq_is':          eq_is_raw,
            'met_is_opt':     met_is_opt,
            'eq_is_opt':      eq_is_opt,
            'met_oos_opt':    met_oos_opt,
            'eq_oos_opt':     eq_oos_opt,
            'eq_baseline':    eq_baseline,
            'best_lb':        best_lb,
            'best_rrr':       best_rrr,
            'split1':       split1,
            'split2':       split2,
            'n_oos1_trades':n_oos1_trades,
            'met_oos1_opt':   m1,
            'met_oos2_opt':   m2
        }

    # --------- fallback (no valid optimisation) ----------------------------
    if _bt.PRINT_EQUITY_CURVE:
        offset = eq_is_raw[-1] - 1
        eq_baseline = np.concatenate((eq_is_raw, eq_oos_raw + offset))
    else:
        eq_baseline = None

    if _bt.USE_MONTE_CARLO:
        _bt.monte_carlo(rets_is_raw, met_is_raw, _bt.MC_RUNS)

    # count raw OOS-1 trades
    if _bt.USE_OOS2:
        n_oos1_trades = sum(
            1
            for side, ent, exit_i, *rest in tr_oos_raw
            if exit_i < _bt.ORIGINAL_OOS
        )
    else:
        n_oos1_trades = None

    split1 = len(eq_is_raw) - 1
    split2 = split1 + n_oos1_trades if n_oos1_trades is not None else None


    return {
        'met_is'     : met_is_raw,
        'eq_is'      : eq_is_raw,
        'met_is_opt' : None,
        'eq_baseline': eq_baseline,
        'split1':       split1,
        'split2':       split2
    }

# 12. MAIN  add equity-curve plot when USE_WFO is False


# 11. WALK-FORWARD (only windows)
def _run_wfo_window(is_df, oos_df, lb, window_tag, regimes_is, regimes_oos, rb_scenarios, export_is=False, best_lbs=None):
    """
    Backtest a single WFO window (IS + OOS), optionally with robustness tweaks.

    If `best_lbs` is given (regime-segmentation mode), the active LB rotates per
    bar according to `regimes_is` / `regimes_oos`. Otherwise a single `lb` is
    used for the whole window. The WFO walk cadence (window boundaries) is
    decided by the caller and is *not* affected by regime changes — only the
    per-bar choice of LB inside the window changes.
    """
    use_regime = best_lbs is not None

    def _run_segment(df_seg, lb_use, regimes_seg, drift=False, best_lbs_seg=None):
        if best_lbs_seg is not None:
            dfi = df_seg.copy()
            for span in (20, 200, 900):
                dfi[f'EMA_{span}'] = dfi['close'].ewm(span=span, adjust=False).mean()
            for lb_v in {v for v in best_lbs_seg.values() if v is not None}:
                dfi[f'EMA_{lb_v}'] = dfi['close'].ewm(span=lb_v, adjust=False).mean()
            raw = _bt.create_regime_signals(dfi, best_lbs_seg, regimes_seg)
        else:
            dfi = _bt.compute_indicators(df_seg.copy(), lb_use)
            raw = _bt.create_raw_signals(dfi, lb_use)
        raw = _bt.filter_raw_signals(raw, regimes_seg)
        sig = _bt.parse_signals(raw, dfi['time'])
        if drift:
            sig = drift_entries(sig)
        return _bt.backtest(dfi, sig)

    bs_lbs = best_lbs if use_regime else None
    lb_tag = (",".join(f"{r}:{v}" for r, v in best_lbs.items() if v is not None)
              if use_regime else lb)

    # baseline
    tr_is,  met_is,  eq_is,  _, _ = _run_segment(is_df,  lb, regimes_is, drift=False, best_lbs_seg=bs_lbs)
    tr_oos, met_oos, _, rets_oos, _ = _run_segment(oos_df, lb, regimes_oos, drift=False, best_lbs_seg=bs_lbs)
    _bt.prettyprint(f"{window_tag} IS",  met_is,  lb_tag)
    _bt.prettyprint(f"{window_tag} OOS", met_oos, lb_tag)

    # export WFO trades for this window (IS + OOS) to the shared trade list
    header_needed = not os.path.exists(_bt.EXPORT_PATH)
    if export_is:
        export_trades(tr_is,  is_df,  'EMA-crossover-WFO', window_tag, 'IS',  _bt.EXPORT_PATH, header_needed)
        header_needed = False  # header already written if needed
    export_trades(tr_oos, oos_df, 'EMA-crossover-WFO', window_tag, 'OOS', _bt.EXPORT_PATH, header_needed)

    # robustness overlays (one line per scenario)
    rb_rets    = {}
    rb_eq_is   = {}
    for label, opts in rb_scenarios:
        fee_mult  = opts["fee_mult"]
        slip_mult = opts["slip_mult"]
        news_on   = opts["news_on"]
        drift_on  = opts["drift_on"]
        var_on    = opts["var_on"]
        # IND_VARIANCE_SEED: previously unseeded (paper-time review noted
        # the +/- 1 LB perturbation propagated to a different optimised
        # parameter across host architectures). Seeded to 42 here so the
        # perturbation is deterministic. Mirrors IND_VARIANCE_SEED in
        # quant-research-framework-rs/src/lib.rs.
        rng       = opts.get("rng", random.Random(42))

        if fee_mult == 1 and slip_mult == 1 and not news_on and not drift_on and not var_on:
            continue

        fee_old, slip_old = _bt.FEE_PCT, _bt.SLIPPAGE_PCT
        _bt.FEE_PCT      = fee_old  * fee_mult
        _bt.SLIPPAGE_PCT = slip_old * slip_mult

        is_work  = inject_news_candles(is_df.copy())  if news_on else is_df
        oos_work = inject_news_candles(oos_df.copy()) if news_on else oos_df

        if use_regime:
            lb_rb_dict = {r: (max(1, v + rng.choice([-1, 1])) if (var_on and v is not None) else v)
                          for r, v in best_lbs.items()}
            lb_rb_tag = ",".join(f"{r}:{v}" for r, v in lb_rb_dict.items() if v is not None)
            try:
                _, met_is_rb, eq_is_rb, _, _ = _run_segment(is_work,  lb, regimes_is, drift=drift_on, best_lbs_seg=lb_rb_dict)
                _, met_oos_rb, _, rets_rb, _ = _run_segment(oos_work, lb, regimes_oos, drift=drift_on, best_lbs_seg=lb_rb_dict)
            finally:
                _bt.FEE_PCT, _bt.SLIPPAGE_PCT = fee_old, slip_old
            _bt.prettyprint(f"{window_tag} IS+{label}",  met_is_rb,  lb_rb_tag)
            _bt.prettyprint(f"{window_tag} OOS+{label}", met_oos_rb, lb_rb_tag)
        else:
            lb_rb    = max(1, lb + rng.choice([-1, 1])) if var_on else lb
            try:
                _, met_is_rb, eq_is_rb, _, _ = _run_segment(is_work,  lb_rb, regimes_is, drift=drift_on)
                _, met_oos_rb, _, rets_rb, _ = _run_segment(oos_work, lb_rb, regimes_oos, drift=drift_on)
            finally:
                _bt.FEE_PCT, _bt.SLIPPAGE_PCT = fee_old, slip_old
            _bt.prettyprint(f"{window_tag} IS+{label}",  met_is_rb,  lb_rb)
            _bt.prettyprint(f"{window_tag} OOS+{label}", met_oos_rb, lb_rb)
        rb_rets[label] = rets_rb
        rb_eq_is[label] = eq_is_rb

    return rets_oos, rb_rets, eq_is, rb_eq_is


def walk_forward(df, met_is_baseline, eq_is_baseline, config: "Optional[_bt.Config]" = None):
    """Rolling walk-forward driver. `config` is optional; see `optimiser`."""
    with _bt.with_config(config):
        return _walk_forward_impl(df, met_is_baseline, eq_is_baseline)


def _walk_forward_impl(df, met_is_baseline, eq_is_baseline):
    # Build robustness scenarios using the queued flags
    if _bt.ROBUSTNESS_SCENARIOS:
        items = list(_bt.ROBUSTNESS_SCENARIOS.items())[:_bt.MAX_ROBUSTNESS_SCENARIOS]
    else:
        default_flags = []
        if _bt.FEE_SHOCK:              default_flags.append("fee_shock")
        if _bt.SLIPPAGE_SHOCK:         default_flags.append("slippage_shock")
        if _bt.NEWS_CANDLES_INJECTION: default_flags.append("news_candles_injection")
        if _bt.ENTRY_DRIFT:            default_flags.append("entry_drift")
        if _bt.INDICATOR_VARIANCE:     default_flags.append("indicator_variance")
        items = [((" + ".join(default_flags)), tuple(default_flags))]

    rb_scenarios = []
    for name, flags in items:
        opts = _opts_from_flags(flags)
        if not any([opts["fee_mult"] != 1, opts["slip_mult"] != 1, opts["news_on"], opts["drift_on"], opts["var_on"]]):
            continue
        label = _label_from_flags(flags)
        rb_scenarios.append((label, opts))

    # ===== 1.  WFO **with** regime segmentation ============================
    # Rewritten in v0.2.0: WFO walks the standard cadence (candles or trades).
    # Regime segmentation only changes which per-regime LB is active for each
    # OOS bar; the WFO test/train boundaries are NOT shifted by regime changes.
    if _bt.USE_WFO and _bt.USE_REGIME_SEG:
        n           = len(df)
        start_total = n - _bt.OOS_CANDLES
        cur_start   = start_total
        window_no   = 1
        all_oos_rets = []
        rb_totals = {label: [] for label, _ in rb_scenarios}
        eq_is_first = None
        rb_eq_seed: dict[str, np.ndarray] = {}

        # full-data regimes (computed once; pluggable detector)
        dfi_full     = _bt.compute_indicators(df, _bt.DEFAULT_LB)
        regimes_full = _bt.detect_regimes(dfi_full)

        # one-shot evaluate_filters on the very first IS window (identical to
        # the legacy behaviour, but anchored to the WFO IS window not a regime
        # stretch). This matters when FILTER_REGIMES / FILTER_DIRECTIONS is on.
        is_start_init = max(0, start_total - _bt.BACKTEST_CANDLES)
        is_df_init    = df.iloc[is_start_init:start_total].reset_index(drop=True)
        regimes_init  = regimes_full.iloc[is_start_init:start_total].reset_index(drop=True)
        initial_lbs, _ = _bt.optimize_regimes_sequential(is_df_init)
        if initial_lbs and any(v is not None for v in initial_lbs.values()):
            dfi_init = is_df_init.copy()
            for span in (20, 200, 900):
                dfi_init[f'EMA_{span}'] = dfi_init['close'].ewm(span=span, adjust=False).mean()
            for lb_v in {v for v in initial_lbs.values() if v is not None}:
                dfi_init[f'EMA_{lb_v}'] = dfi_init['close'].ewm(span=lb_v, adjust=False).mean()
            raw_init = _bt.create_regime_signals(dfi_init, initial_lbs, regimes_init)
            sig_init = _bt.parse_signals(raw_init, dfi_init['time'])
            tr_init, _, _, rets_init, _ = _bt.backtest(dfi_init, sig_init)
            _bt.evaluate_filters(tr_init, rets_init, regimes_init)

        while cur_start < n:
            # --- decide window end (same cadence as the no-regime path) ---
            if _bt.WFO_TRIGGER_MODE == 'candles':
                cur_end = min(cur_start + _bt.WFO_TRIGGER_VAL, n)
            else:   # by trade-count
                is_win_start_p = cur_start - _bt.BACKTEST_CANDLES
                is_df_p        = df.iloc[is_win_start_p:cur_start].reset_index(drop=True)
                best_lbs_p, _  = _bt.optimize_regimes_sequential(is_df_p)
                if not best_lbs_p or all(v is None for v in best_lbs_p.values()):
                    break
                dfo_p     = df.iloc[cur_start:n].reset_index(drop=True)
                regimes_p = regimes_full.iloc[cur_start:n].reset_index(drop=True)
                dfi_p = dfo_p.copy()
                for span in (20, 200, 900):
                    dfi_p[f'EMA_{span}'] = dfi_p['close'].ewm(span=span, adjust=False).mean()
                for lb_v in {v for v in best_lbs_p.values() if v is not None}:
                    dfi_p[f'EMA_{lb_v}'] = dfi_p['close'].ewm(span=lb_v, adjust=False).mean()
                raw_p = _bt.create_regime_signals(dfi_p, best_lbs_p, regimes_p)
                raw_p = _bt.filter_raw_signals(raw_p, regimes_p)
                sig_p = _bt.parse_signals(raw_p, dfi_p['time'])
                tr_p, _, _, _, _ = _bt.backtest(dfi_p, sig_p)
                if not tr_p:
                    cur_end = n
                else:
                    idx     = min(_bt.WFO_TRIGGER_VAL, len(tr_p)) - 1
                    cur_end = min(cur_start + tr_p[idx][2] + 1, n)

            # --- IS slice + per-regime optimisation -----------------------
            is_win_start = cur_start - _bt.BACKTEST_CANDLES
            is_df_roll   = df.iloc[is_win_start:cur_start].reset_index(drop=True)
            regimes_is   = regimes_full.iloc[is_win_start:cur_start].reset_index(drop=True)

            best_lbs, _  = _bt.optimize_regimes_sequential(is_df_roll)
            if not best_lbs or all(v is None for v in best_lbs.values()):
                break
            if _bt.EMIT_OPT_SURFACE:
                import backtester as _bt2
                from backtester import opt_surface as _osf
                _hdr = not os.path.exists(_osf._surface_path(_bt.EXPORT_PATH, _osf._resolve_format()))
                _osf.emit_surface_regime(_bt2, is_df_roll, best_lbs,
                                         f"{window_no:02d}", write_header=_hdr)

            # --- OOS slice with regime-rotated signals --------------------
            dfo          = df.iloc[cur_start:cur_end].reset_index(drop=True)
            regimes_oos  = regimes_full.iloc[cur_start:cur_end].reset_index(drop=True)

            rets_oos, rb_rets_window, eq_is_window, rb_eq_is_window = _run_wfo_window(
                is_df_roll, dfo, lb=None,
                window_tag=f"W{window_no:02d}",
                regimes_is=regimes_is,
                regimes_oos=regimes_oos,
                rb_scenarios=rb_scenarios,
                export_is=(window_no == 1),
                best_lbs=best_lbs,
            )
            if eq_is_first is None:
                eq_is_first = eq_is_window
                for label, eq_is_rb in rb_eq_is_window.items():
                    rb_eq_seed[label] = eq_is_rb
            all_oos_rets.extend(rets_oos)
            for label in rb_totals:
                rb_totals[label].extend(rb_rets_window.get(label, []))

            cur_start = cur_end
            window_no += 1

        eq_seed = eq_is_first if eq_is_first is not None else eq_is_baseline
        all_oos_rets = np.array(all_oos_rets, dtype=float)
        eq_wfo = np.concatenate([eq_seed,
                                 eq_seed[-1] + np.cumsum(all_oos_rets)])

        rb_eq_curves: dict[str, np.ndarray] = {}
        if rb_totals:
            for label, vals in rb_totals.items():
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                seed_rb = rb_eq_seed.get(label, eq_seed)
                rb_eq_curves[label] = np.concatenate([seed_rb,
                                                      seed_rb[-1] + np.cumsum(arr)])

        split_wfo_is = len(eq_seed) - 1
        return all_oos_rets, eq_wfo, rb_eq_curves, split_wfo_is

    # ===== 2.  WFO **without** regime segmentation =========================
    n           = len(df)
    start_total = n - _bt.OOS_CANDLES
    cur_start   = start_total
    window_no   = 1
    all_oos_rets = []
    rb_totals = {label: [] for label, _ in rb_scenarios}
    eq_is_first = None
    rb_eq_seed: dict[str, np.ndarray] = {}

    while cur_start < n:
        # --- decide window end ---------------------------------------------
        if _bt.WFO_TRIGGER_MODE == 'candles':
            cur_end = min(cur_start + _bt.WFO_TRIGGER_VAL, n)
        else:   # by trade-count
            is_win_start = cur_start - _bt.BACKTEST_CANDLES
            is_df_roll   = df.iloc[is_win_start:cur_start].reset_index(drop=True)
            lb_roll, _   = _bt.optimiser(is_df_roll, range(*_bt.LOOKBACK_RANGE), _bt.OPT_METRIC, _bt.MIN_TRADES)
            if not lb_roll:
                break
            dfo_tmp      = df.iloc[cur_start:n].reset_index(drop=True)
            raw_tmp      = _bt.create_raw_signals(_bt.compute_indicators(dfo_tmp, lb_roll), lb_roll)
            raw_tmp      = _bt.filter_raw_signals(raw_tmp, None)          # just dirs
            sig_tmp      = _bt.parse_signals(raw_tmp, dfo_tmp['time'])
            tr_tmp, _, _, _, _ = _bt.backtest(dfo_tmp, sig_tmp)
            if not tr_tmp:
                cur_end = n
            else:
                idx     = min(_bt.WFO_TRIGGER_VAL, len(tr_tmp)) - 1
                cur_end = min(cur_start + tr_tmp[idx][2] + 1, n)

        # --- real rolling IS  current OOS ---------------------------------
        is_win_start = cur_start - _bt.BACKTEST_CANDLES
        is_df_roll   = df.iloc[is_win_start:cur_start].reset_index(drop=True)
        lb_roll, _   = _bt.optimiser(is_df_roll, range(*_bt.LOOKBACK_RANGE), _bt.OPT_METRIC, _bt.MIN_TRADES)
        if not lb_roll:
            break
        if _bt.EMIT_OPT_SURFACE:
            import backtester as _bt2
            from backtester import opt_surface as _osf
            _hdr = not os.path.exists(_osf._surface_path(_bt.EXPORT_PATH, _osf._resolve_format()))
            _osf.emit_surface_classic(_bt2, is_df_roll, f"{window_no:02d}", write_header=_hdr)

        dfo          = df.iloc[cur_start:cur_end].reset_index(drop=True)
        rets_oos, rb_rets_window, eq_is_window, rb_eq_is_window = _run_wfo_window(
            is_df_roll, dfo, lb_roll,
            window_tag=f"W{window_no:02d}",
            regimes_is=None,
            regimes_oos=None,
            rb_scenarios=rb_scenarios,
            export_is=(window_no == 1)  # only export IS for first WFO window
        )
        if eq_is_first is None:
            eq_is_first = eq_is_window
            for label, eq_is_rb in rb_eq_is_window.items():
                rb_eq_seed[label] = eq_is_rb
        all_oos_rets.extend(rets_oos)
        for label in rb_totals:
            rb_totals[label].extend(rb_rets_window.get(label, []))

        cur_start = cur_end
        window_no += 1

    eq_seed = eq_is_first if eq_is_first is not None else eq_is_baseline
    all_oos_rets = np.array(all_oos_rets, dtype=float)
    eq_wfo = np.concatenate([eq_seed,
                             eq_seed[-1] + np.cumsum(all_oos_rets)])

    rb_eq_curves = {}
    if rb_totals:
        for label, vals in rb_totals.items():
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            seed_rb = rb_eq_seed.get(label, eq_seed)
            rb_eq_curves[label] = np.concatenate([seed_rb,
                                                  seed_rb[-1] + np.cumsum(arr)])

    split_wfo_is = len(eq_seed) - 1
    return all_oos_rets, eq_wfo, rb_eq_curves, split_wfo_is

def inject_news_candles(df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """Return **new** DataFrame where every 5001000 bars a burst of 12 candles
    gets oversized wicks (25 average true range of the previous 100 bars).
    Open/Close stay unchanged; only High/Low are stretched.
    """
    rng = random.Random(seed)
    i = 0
    n = len(df)
    while i < n:
        i += rng.randint(500, 1000)           # skip 5001000 bars
        if i >= n:
            break
        burst = rng.randint(1, 2)             # 12 candles in this burst
        for j in range(burst):
            idx = i + j
            if idx >= n:
                break
            w_start = max(0, idx - 100)
            avg_range = (df['high'].iloc[w_start:idx] - df['low'].iloc[w_start:idx]).abs().mean()
            if math.isnan(avg_range) or avg_range == 0:
                avg_range = (df['high'] - df['low']).abs().mean()
            extent = avg_range * rng.uniform(2, 5)
            direction = rng.choice(['up', 'down', 'both'])
            op   = df.at[idx, 'open']
            cp   = df.at[idx, 'close']
            hi   = df.at[idx, 'high']
            lo   = df.at[idx, 'low']
            top  = max(op, cp, hi)
            bot  = min(op, cp, lo)
            if direction in ('up', 'both'):
                hi_new = top + extent
            else:
                hi_new = top
            if direction in ('down', 'both'):
                lo_new = bot - extent
            else:
                lo_new = bot
            # Safety: ensure logical ordering
            if hi_new < max(op, cp):
                hi_new = max(op, cp)
            if lo_new > min(op, cp):
                lo_new = min(op, cp)
            df.at[idx, 'high'] = hi_new
            df.at[idx, 'low']  = lo_new
        i += burst
    return df

def apply_news_injection(config: "Optional[_bt.Config]" = None):
    """Injects synthetic newswick candles and rebacktests with *fixed* params.

    `config` is optional; see `optimiser` docstring for the contract.
    """
    with _bt.with_config(config):
        return _apply_news_injection_impl()


def _apply_news_injection_impl():
    # Was: `global TP_PERCENTAGE, USE_TP`. Replaced with the
    # `_bt.X=...` pattern used elsewhere in this file so we do
    # not need a `global` keyword. The save/restore semantics are
    # identical: snapshot before the override, restore after.
    if 'df' not in _bt.signals_cache:
        raise RuntimeError("Baseline DF not cached  run main() first!")

    df_mod = inject_news_candles(_bt.signals_cache['df'].copy())
    N      = len(df_mod)
    is_df  = df_mod.iloc[N - _bt.OOS_CANDLES - _bt.BACKTEST_CANDLES : N - _bt.OOS_CANDLES].reset_index(drop=True)
    oos_df = df_mod.iloc[N - _bt.OOS_CANDLES : N].reset_index(drop=True)

    # === Classic optimisation path =======================================
    if _bt.signals_cache.get('mode') == 'classic':
        best_lb  = _bt.signals_cache['best_lb']
        best_rrr = _bt.signals_cache['best_rrr']

        # set fixed RRR (if any)
        old_tp, old_flag = _bt.TP_PERCENTAGE, _bt.USE_TP
        if best_rrr is not None:
            _bt.TP_PERCENTAGE = best_rrr * _bt.SL_PERCENTAGE
            _bt.USE_TP        = True

        # IS
        dfi = _bt.compute_indicators(is_df.copy(), best_lb)
        raw = _bt.create_raw_signals(dfi, best_lb)
        raw = _bt.filter_raw_signals(raw, None)
        sig = _bt.parse_signals(raw, dfi['time'])
        _, met_is_rb, eq_is_rb, _, _ = _bt.backtest(dfi, sig)

        # OOS
        dfo = _bt.compute_indicators(oos_df.copy(), best_lb)
        raw = _bt.create_raw_signals(dfo, best_lb)
        raw = _bt.filter_raw_signals(raw, None)
        sig = _bt.parse_signals(raw, dfo['time'])
        _, met_oos_rb, eq_oos_rb, _, _ = _bt.backtest(dfo, sig)

        eq_baseline_rb = np.concatenate((eq_is_rb, eq_oos_rb + (eq_is_rb[-1] - 1)))

        # restore TP settings
        _bt.TP_PERCENTAGE = old_tp
        _bt.USE_TP = old_flag

    # === Regimesegmentation path ========================================
    else:
        best_lbs = _bt.signals_cache['best_lbs']
        # IS
        _, met_is_rb, eq_is_rb, _   = backtest_continuous_regime(is_df.copy(),  best_lbs)
        # OOS
        _, met_oos_rb, eq_oos_rb, _ = backtest_continuous_regime(oos_df.copy(), best_lbs)
        eq_baseline_rb = np.concatenate((eq_is_rb, eq_oos_rb + (eq_is_rb[-1] - 1)))

    return {
        'met_is_rb':      met_is_rb,
        'met_oos_rb':     met_oos_rb,
        'eq_baseline_rb': eq_baseline_rb
    }

def drift_entries(sig: np.ndarray) -> np.ndarray:
    """Return a copy where codes 1 & 3 are shifted to the *next* bar.
    Pure exits (2, 4) stay in place.  If an entry falls on the last bar
    it is simply dropped."""
    out = np.zeros_like(sig)
    for i, code in enumerate(sig):
        if code in (1, 3):                 # entries that also auto-exit the
            if i + 1 < len(sig):           # opposite side get drifted too
                out[i + 1] = code
        elif code in (2, 4):
            out[i] = code
    return out


def apply_combined_robustness(
        fee_mult=1,
        slip_mult=1,
        inject_news=False,
        drift=False,
        variance=False):
    """Layer *all* enabled robustness tweaks in one pass.

    Parameters
    ----------
    fee_mult, slip_mult : int or float
        Multipliers for commission and slippage.
    inject_news : bool
        If True add synthetic newswick candles.
    drift : bool
        If True delay every entry signal by one bar.
    variance : bool
        If True perturb the chosen slowEMA lookbacks by 1.
    """
    # Was: `global FEE_PCT, SLIPPAGE_PCT, TP_PERCENTAGE, USE_TP`. Replaced
    # with `_bt.X=...` so we do not need a `global` keyword.
    # Save/restore semantics unchanged.

    # 1) build working price series ---------------------------------------
    df_work = _bt.signals_cache['df'].copy()
    if inject_news:
        df_work = inject_news_candles(df_work)

    # split IS / OOS --------------------------------------------------------
    N      = len(df_work)
    is_df  = df_work.iloc[N - _bt.BACKTEST_CANDLES - _bt.OOS_CANDLES : N - _bt.OOS_CANDLES].reset_index(drop=True)
    oos_df = df_work.iloc[N - _bt.OOS_CANDLES : N].reset_index(drop=True)

    # 2) fee / slippage shock ---------------------------------------------
    fee_old, slip_old = _bt.FEE_PCT, _bt.SLIPPAGE_PCT
    _bt.FEE_PCT      = fee_old * fee_mult
    _bt.SLIPPAGE_PCT = slip_old * slip_mult

    # 3) choose variant LBs -----------------------------------------------
    # IND_VARIANCE_SEED: seeded to 42 (was unseeded; cross-arch review
    # flagged that the +/- 1 LB perturbation could pick a different
    # optimum LB on different hosts and silently propagate into the
    # printed metric ledger).
    rng = random.Random(42)

    if _bt.signals_cache['mode'] == 'classic':
        base_lb   = _bt.signals_cache['best_lb']
        best_rrr  = _bt.signals_cache['best_rrr']
        lb_use    = base_lb
        if variance:
            offset   = rng.choice([-1, 1])
            lb_use   = max(1, base_lb + offset)

        # apply RRR if any
        tp_old, flag_old = _bt.TP_PERCENTAGE, _bt.USE_TP
        if best_rrr is not None:
            _bt.TP_PERCENTAGE = best_rrr * _bt.SL_PERCENTAGE
            _bt.USE_TP        = True

        # ---------- IS -----------------------------------------------
        dfi = _bt.compute_indicators(is_df.copy(), lb_use)
        raw = _bt.filter_raw_signals(_bt.create_raw_signals(dfi, lb_use), None)
        sig = _bt.parse_signals(raw, dfi['time'])
        if drift:
            sig = drift_entries(sig)
        _, met_is_rb, eq_is_rb, _, _ = _bt.backtest(dfi, sig)

        # ---------- OOS ----------------------------------------------
        dfo = _bt.compute_indicators(oos_df.copy(), lb_use)
        raw = _bt.filter_raw_signals(_bt.create_raw_signals(dfo, lb_use), None)
        sig = _bt.parse_signals(raw, dfo['time'])
        if drift:
            sig = drift_entries(sig)
        _, met_oos_rb, eq_oos_rb, _, _ = _bt.backtest(dfo, sig)

        _bt.TP_PERCENTAGE = tp_old
        _bt.USE_TP        = flag_old

    else:  # regime segmentation path
        base_lbs = _bt.signals_cache['best_lbs'].copy()
        lbs_use  = base_lbs
        if variance:
            lbs_use = {r: max(1, lb + rng.choice([-1, 1])) for r, lb in base_lbs.items()}

        # IS
        _, met_is_rb, eq_is_rb, _   = backtest_continuous_regime(is_df.copy(),  lbs_use)
        # OOS
        _, met_oos_rb, eq_oos_rb, _ = backtest_continuous_regime(oos_df.copy(), lbs_use)

    # 4) restore globals ----------------------------------------------------
    _bt.FEE_PCT      = fee_old
    _bt.SLIPPAGE_PCT = slip_old

    # 5) stitch equity curve ----------------------------------------------
    eq_baseline_rb = np.concatenate((eq_is_rb, eq_oos_rb + (eq_is_rb[-1] - 1)))

    return {
        'met_is_rb':      met_is_rb,
        'met_oos_rb':     met_oos_rb,
        'eq_baseline_rb': eq_baseline_rb
    }


# === Run robustness tests ===
# === Run robustness tests (all-in-one print & plot) ===
def _normalize_rb_flag(flag: str) -> str:
    return flag.strip().lower().replace(" ", "_")


def _opts_from_flags(flags: tuple[str, ...] | list[str] | None) -> dict[str, object]:
    flags = flags or ()
    tokens = {_normalize_rb_flag(f) for f in flags if isinstance(f, str)}
    fee_on  = any(t in tokens for t in ("fee_shock", "fee", "fees"))
    slip_on = any(t in tokens for t in ("slippage_shock", "slippage", "slip"))
    news_on = any(t in tokens for t in ("news_candles_injection", "news"))
    drift_on= any(t in tokens for t in ("entry_drift", "drift"))
    var_on  = any(t in tokens for t in ("indicator_variance", "variance", "lb_variance"))
    return {
        "fee_mult": 2 if fee_on else 1,
        "slip_mult":3 if slip_on else 1,
        "news_on":  news_on,
        "drift_on": drift_on,
        "var_on":   var_on,
    }


def _label_from_flags(flags: tuple[str, ...] | list[str]) -> str:
    """Build a short label using 3-char abbreviations joined by '+'."""
    norm = [_normalize_rb_flag(f) for f in flags if isinstance(f, str)]
    pretty_map = {
        "fee_shock": "FEE",
        "slippage_shock": "SLI",
        "news_candles_injection": "NWS",
        "entry_drift": "ENT",
        "indicator_variance": "IND",
    }
    parts = []
    for t in norm:
        parts.append(pretty_map.get(t, t[:3].upper()))
    return "+".join(parts) if parts else "NONE"


def run_robustness_tests():
    """Runs one or more robustness scenarios separately (no overlapping toggles)."""

    def _run_single(name: str, opts: dict[str, object]):
        fee_mult  = opts["fee_mult"]
        slip_mult = opts["slip_mult"]
        news_on   = opts["news_on"]
        drift_on  = opts["drift_on"]
        var_on    = opts["var_on"]
        flags_current = _bt.ROBUSTNESS_SCENARIOS.get(name, ()) if _bt.ROBUSTNESS_SCENARIOS else ()

        if fee_mult == 1 and slip_mult == 1 and not news_on and not drift_on and not var_on:
            return None

        # if we've doubled OOS_CANDLES, shrink it back so robustness only hits OOS1
        if _bt.USE_OOS2:
            old_oos = _bt.OOS_CANDLES
            _bt.OOS_CANDLES = _bt.ORIGINAL_OOS
        else:
            old_oos = None

        parts = []
        if fee_mult > 1: parts.append('Fee')
        if slip_mult>1:  parts.append('Slippage')
        if news_on:      parts.append('News')
        if drift_on:     parts.append('Drift')
        if var_on:       parts.append('LB')
        label = _label_from_flags(flags_current if flags_current else parts)
        verbose_name = name if name else (' + '.join(parts) + ' Robustness')

        print(f"\n Robustness Test: {label} ({verbose_name}) ")
        res = apply_combined_robustness(
            fee_mult, slip_mult,
            inject_news=news_on,
            drift=drift_on,
            variance=var_on
        )

        # restore full OOS window for plotting & WFO
        if old_oos is not None:
            _bt.OOS_CANDLES = old_oos

        _bt.prettyprint(f"{label} IS",  res['met_is_rb'])
        _bt.prettyprint(f"{label} OOS1", res['met_oos_rb'])
        return label, res

    # decide which scenarios to run: custom queue first, else fall back to legacy combined toggle
    if _bt.ROBUSTNESS_SCENARIOS:
        items = list(_bt.ROBUSTNESS_SCENARIOS.items())[:_bt.MAX_ROBUSTNESS_SCENARIOS]
    else:
        default_flags = []
        if _bt.FEE_SHOCK:              default_flags.append("fee_shock")
        if _bt.SLIPPAGE_SHOCK:         default_flags.append("slippage_shock")
        if _bt.NEWS_CANDLES_INJECTION: default_flags.append("news_candles_injection")
        if _bt.ENTRY_DRIFT:            default_flags.append("entry_drift")
        if _bt.INDICATOR_VARIANCE:     default_flags.append("indicator_variance")
        items = [((' + '.join(f.title().replace('_', ' ') for f in default_flags) + ' Robustness') if default_flags else "", tuple(default_flags))]

    results = {}
    for name, flags in items:
        label_res = _run_single(str(name), _opts_from_flags(flags))
        if label_res is None:
            continue
        label, res = label_res
        results[label] = res

    return results


def age_dataset(df, age):
    """
    Remove the last `age` rows (most recent candles) from df.
    """
    if age <= 0:
        return df
    if age >= len(df):
        raise ValueError(f"AGE_DATASET ({age}) >= dataset length ({len(df)})")
    # drop last `age` rows and reset index
    return df.iloc[:-age].reset_index(drop=True)


# 12. MAIN
def main(config: "Optional[_bt.Config]" = None):
    """
    Top-level entry point. Runs the full backtest pipeline:
    classic single run -> robustness -> walk-forward -> plot.

    `config` is optional. When provided, applies that Config to module
    globals for the duration of this call (and restores prior values on
    exit). Pass `bt.Config()` to use library defaults regardless of the
    current module state, or `bt.Config.from_module()` to snapshot the
    current state and tweak fields. When omitted, reads from module
    globals — the legacy `bt.X = Y` API works exactly as before.
    """
    with _bt.with_config(config):
        return _main_impl()


def _main_impl():
    # Explicit early check so users running `python -m backtester` get a
    # clear error before any heavy imports / numba JIT warm-up happens.
    if not os.path.exists(_bt.CSV_FILE):
        raise FileNotFoundError(
            f"CSV file not found: {_bt.CSV_FILE}\n\n"
            "Put your OHLC CSV at that path, or change CSV_FILE / set BT_CSV.\n"
            "You can generate one with binance_ohlc_downloader.py (see README)."
        )
    df = _bt.load_ohlc(_bt.CSV_FILE)

    df = age_dataset(df, _bt.AGE_DATASET)

    df['is_traded'] = df['time'].apply(lambda ts: _bt.in_session(ts) if _bt.TRADE_SESSIONS else True)

    # item #3: clear any stale side-channel from a prior main() call in the
    # same process. On the regime-no-WFO path optimiser() is never called,
    # so without this the report could surface a phantom trial vector from
    # an unrelated earlier run (Lens A3). An empty capture then honestly
    # reports N=0 instead of stale-N.
    if _bt.OVERFIT_REPORT:
        _bt._runtime_state.pop('_last_trial_sharpes', None)

    # 1) baseline run
    base = classic_single_run(df)
    # item #3: snapshot the baseline IS optimisation's distinct trial
    # Sharpes NOW, before the WFO loop's per-window optimiser() calls
    # overwrite the side-channel. Canonical "strategies tried" set.
    _overfit_trials = list(_bt._runtime_state.get('_last_trial_sharpes', [])) \
        if _bt.OVERFIT_REPORT else []
    _bt.signals_cache['df'] = df.copy()

    # 1.a) print baseline metrics
    print(" Baseline Optimized Metrics ")
    # --- Classic optimization path ---
    if base.get('met_is_opt') is not None:
        _bt.prettyprint('Baseline IS', base['met_is_opt'])
        # two-window?
        if base.get('met_oos1_opt') is not None:
            _bt.prettyprint('Baseline OOS1', base['met_oos1_opt'])
            _bt.prettyprint('Baseline OOS2', base['met_oos2_opt'])
        else:
            _bt.prettyprint('Baseline OOS', base['met_oos_opt'])

    # --- Regime segmentation path ---
    elif base.get('met_is') is not None:
        # met_is holds the IS-regime metrics
        _bt.prettyprint('Baseline IS-Regime', base['met_is'])
        # two-window?
        if base.get('met_oos1_reg') is not None:
            _bt.prettyprint('Baseline OOS1-Regime', base['met_oos1_reg'])
            _bt.prettyprint('Baseline OOS2-Regime', base['met_oos2_reg'])
        else:
            _bt.prettyprint('Baseline OOS-Regime', base['met_oos_reg'])


    # 2) robustness tests
    rb = run_robustness_tests()

    # 3) plotting
    if _bt.USE_WFO:
        print(" Running Walk-Forward Windows ")
        oos_rets, eq_wfo, rb_eq_wfo, split_wfo_is = walk_forward(df, base['met_is'], base['eq_is'])
        # item #3: opt-in overfitting diagnostics. Additive lines only; none
        # carry the LINE_RE metric body, so parity harnesses are unaffected.
        # trial count = distinct baseline strategies (NOT windows*combos). The
        # chosen Sharpe is recomputed FROM oos_rets inside emit() in the
        # Bailey-LdP sqrt(T)*mean/std convention (NOT met_is).
        if _bt.OVERFIT_REPORT:
            from backtester import overfit_report
            overfit_report.emit(
                trial_sharpes=_overfit_trials,
                oos_returns=oos_rets,
                sharpe_mode=_bt.SHARPE_MODE,
                equity_matrix=_bt._runtime_state.get('_overfit_equity_matrix'),
            )
        # Plot baseline and WFO + robustness
        if _bt.PRINT_EQUITY_CURVE:
            plt.figure(figsize=(10,5))
            # WFO equity only
            plt.plot(eq_wfo * _bt.ACCOUNT_SIZE, label='WFO Rolling')
            # robustness scenarios (on the WFO rolling curve)
            for name, eq_rb in rb_eq_wfo.items():
                plt.plot(eq_rb * _bt.ACCOUNT_SIZE, label=name)
            # split at the end of the first WFO IS segment (length of eq_is passed to walk_forward)
            plt.axvline(split_wfo_is, color='red', linestyle='--', label='End IS (W1)')
            plt.title('Equity Curve: WFO & Robustness')
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    else:
    # single-run plotting
        if _bt.PRINT_EQUITY_CURVE:
            plt.figure(figsize=(10,5))

            #  pick the right equity curve
            if _bt.USE_REGIME_SEG and base.get('eq_baseline') is not None:
                # regimesegmented: always use the unified array we built
                eq_all = base['eq_baseline']
            else:
                # fallback for classic or any path without unified baseline
                eq_all = base.get('eq_baseline')
                if eq_all is None:
                    # legacy stitch
                    eq_all = np.concatenate((base['eq_baseline_is'], base['eq_baseline_oos']))


            plt.plot(eq_all * _bt.ACCOUNT_SIZE, label='Baseline Equity')

            # 2) robustness overlays
            for name, res in rb.items():
                eq_rb = res['eq_baseline_rb']
                if base.get('split2') is not None:
                    # splice OOS2 from the baseline returns onto the end of the robustness-tested OOS1
                    split2 = base['split2']
                    eq_base_split2 = eq_all[split2]
                    baseline_tail = eq_all[split2+1:]
                    rel_changes = baseline_tail - eq_base_split2
                    tail = rel_changes + eq_rb[-1]
                    eq_rb_plot = np.concatenate((eq_rb, tail))
                else:
                    eq_rb_plot = eq_rb

                plt.plot(eq_rb_plot * _bt.ACCOUNT_SIZE, label=name)

            # 3) vertical lines at trade-based splits
            split1 = base['split1']
            plt.axvline(split1, color='red', linestyle='--', label='End IS')

            split2 = base.get('split2')
            if split2 is not None:
                plt.axvline(split2, color='red', linestyle='--', label='End OOS1')

            # 4) finalize
            plt.title('Equity Curve with Robustness Tests')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
