#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optimiser + Monte-Carlo extracted from ``backtester/__init__.py``.

Mirrors the Rust port ``src/objectives.rs``. Pure move/re-export: every
function here behaves byte-for-byte as it did in its prior location.

All module-level globals (OPTIMIZE_RRR, TP_PERCENTAGE, USE_TP,
SL_PERCENTAGE, LEGACY_SIDE_BUG, DEFAULT_LB, DRAWDOWN_CONSTRAINT,
SMART_OPTIMIZATION, OVERFIT_REPORT, METRICS, PRINT_EQUITY_CURVE,
ACCOUNT_SIZE, dd_constraint, ...), the runtime-state dict, the
``with_config`` context manager, and every engine function that STAYS in
``__init__`` (``compute_indicators``, ``create_raw_signals``,
``parse_signals``, ``backtest``) are accessed LIVE via ``_bt.<name>`` so
the ``Config.with_config()`` runtime-mutation contract keeps working.

Globals that the optimiser temporarily rebinds (TP_PERCENTAGE, USE_TP)
are written with ``setattr(_bt, 'X', ...)`` — i.e. ``_bt.X = ...`` — so
the mutation lands on the ``backtester`` module namespace (which the
engine reads back through ``backtest()``), exactly as the old
``globals()['X'] = ...`` did when this code lived in ``__init__``.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib.gridspec import GridSpec
from typing import Optional

import backtester as _bt

__all__ = [
    "optimiser",
    "_optimiser_impl",
    "monte_carlo",
    "_monte_carlo_impl",
]


# 6. OPTIMISER  now with Smart Optimization support
# 6. OPTIMISER  now with Smart Optimization support (patched for WFO)
def optimiser(df, lb_range, metric, min_trades, config: "Optional[_bt.Config]" = None):
    """
    Updated optimiser:
      1) Coarse search in steps of 2 over lb_range.
      2) Take the best coarse lookback.
      3) Fine-tune by testing best_lb - 1 and best_lb + 1.
      4) Return the overall best of those three.

    `config` is optional. When provided, the engine uses cfg's values for
    the duration of this call (and restores prior values on exit). When
    omitted, the call reads from module globals as it always has —
    the documented `bt.X = Y` / `monkeypatch.setattr(bt, "X", Y)` API
    keeps working unchanged.
    """
    with _bt.with_config(config):
        return _optimiser_impl(df, lb_range, metric, min_trades)


def _optimiser_impl(df, lb_range, metric, min_trades):
    # Was: `global last_unfiltered_raw`. Moved to _runtime_state dict
    # so we can mutate without `global`. See _runtime_state docstring.
    eval_cache = {}

    # Helper to evaluate a single lookback
    def _evaluate(lb):
        if lb in eval_cache:
            return eval_cache[lb]

        # 1) Compute indicators & raw signals for this lookback
        dfi = _bt.compute_indicators(df, lb)
        raw = _bt.create_raw_signals(dfi, lb)
        _bt._runtime_state['last_unfiltered_raw'] = raw.copy()
        sig = _bt.parse_signals(raw, dfi['time'])

        # 2) Backtest (with or without RRR optimisation)
        if not _bt.OPTIMIZE_RRR:
            _, met, _, _, _ = _bt.backtest(dfi, sig)
        else:
            # --- quick probe at fixed 5 R ---
            old_tp, old_tp_flag = _bt.TP_PERCENTAGE, _bt.USE_TP
            _bt.TP_PERCENTAGE = 5 * _bt.SL_PERCENTAGE
            _bt.USE_TP        = True

            _bt._runtime_state['last_unfiltered_raw'] = raw.copy()
            trades_probe, _, _, _, _ = _bt.backtest(dfi, sig)

            # compute peak and close R multiples
            peak_Rs, close_Rs = [], []
            for side, e, x, *_ in trades_probe:
                entry_price = dfi['close'].iloc[e]
                risk = entry_price * _bt.SL_PERCENTAGE / 100.0

                # NOTE: pre-v0.2.5 used `side == 'long'`, which compared int8
                # against a str and always took the else branch. Default now is
                # the corrected `side == 1` test; LEGACY_SIDE_BUG=True reverts.
                is_long = (side == 'long') if _bt.LEGACY_SIDE_BUG else (side == 1)
                if is_long:
                    high_slice = dfi['high'].iloc[e:x+1].values
                    peak_R = (high_slice.max() - entry_price) / risk
                    close_R = (dfi['close'].iloc[x] - entry_price) / risk
                else:  # short
                    low_slice = dfi['low'].iloc[e:x+1].values
                    peak_R = (entry_price - low_slice.min()) / risk
                    close_R = (entry_price - dfi['close'].iloc[x]) / risk

                peak_Rs.append(min(peak_R, 3.0))
                close_Rs.append(close_R)

            peak_Rs = np.array(peak_Rs, dtype=float)
            close_Rs = np.array(close_Rs, dtype=float)

            # sum R for each candidate R_target
            sum_R_for_candidates = {
                R_target: np.where(peak_Rs >= R_target, R_target, close_Rs).sum()
                for R_target in range(1, 4)
            }

            # pick best RRR
            best_rrr = max(sum_R_for_candidates, key=sum_R_for_candidates.get)

            # re-run with optimal TP
            _bt.TP_PERCENTAGE = best_rrr * _bt.SL_PERCENTAGE
            _bt._runtime_state['last_unfiltered_raw'] = raw.copy()
            _, met, _, _, _ = _bt.backtest(dfi, sig)
            met['RRR'] = best_rrr

            # restore TP settings
            _bt.TP_PERCENTAGE = old_tp
            _bt.USE_TP        = old_tp_flag

        # 3) filter by minimum trades
        if met['Trades'] < min_trades:
            eval_cache[lb] = None
            return None

        # 4) Drawdown constraint
        if _bt.dd_constraint is not None and met['MaxDrawdown'] > _bt.dd_constraint:
            eval_cache[lb] = None
            return None

        # 5) Compute value to optimize (flip for MaxDrawdown)
        val = -met[metric] if metric == 'MaxDrawdown' else met[metric]
        eval_cache[lb] = (val, lb, met)
        return eval_cache[lb]

    # Build list of all lookbacks and coarse subset
    all_lbs = list(lb_range)
    coarse_lbs = all_lbs[::2]

    # 1) Coarse pass
    coarse_results = [res for lb in coarse_lbs if (res := _evaluate(lb)) is not None]


    if not coarse_results:
        print(f"No lookback meets drawdown  {_bt.DRAWDOWN_CONSTRAINT}, using raw LB {_bt.DEFAULT_LB}")
        dfi = _bt.compute_indicators(df, _bt.DEFAULT_LB)
        raw = _bt.create_raw_signals(dfi, _bt.DEFAULT_LB)
        sig = _bt.parse_signals(raw, dfi['time'])
        _, met_raw, *_ = _bt.backtest(dfi, sig)
        return _bt.DEFAULT_LB, met_raw


    # Pick best coarse, then fine-tune
    coarse_results.sort(key=lambda x: x[0], reverse=True)
    best_val, best_lb, best_met = coarse_results[0]

    idx = all_lbs.index(best_lb)
    candidates = [(best_val, best_lb, best_met)]
    for ni in (idx - 1, idx + 1):
        if 0 <= ni < len(all_lbs):
            if (res := _evaluate(all_lbs[ni])) is not None:
                candidates.append(res)

    candidates.sort(key=lambda x: x[0], reverse=True)

    # --- Smart optimization guard against spiky PF -------------------------
    selected = candidates[0]
    if _bt.SMART_OPTIMIZATION:
        best_candidate = candidates[0]
        best_lb = best_candidate[1]
        all_lb_set = set(all_lbs)

        for cand in candidates:
            _, lb_cand, met_cand = cand
            pf_cand = met_cand['PF']

            ok = True
            for neigh in (lb_cand - 1, lb_cand + 1):
                if neigh in all_lb_set:
                    neigh_res = _evaluate(neigh)
                    if neigh_res is None:
                        continue
                    pf_neigh = neigh_res[2]['PF']
                    if pf_cand > 1.10 * pf_neigh:
                        ok = False
                        break
            if ok:
                selected = cand
                if lb_cand != best_lb:
                    print(f"Smart Optimization: switched from LB {best_lb} to LB {lb_cand} because PF spike exceeded 10% vs neighbors.")
                break

    # --- item #3 opt-in side-channel: capture the distinct trial Sharpes ---
    # eval_cache maps each EVALUATED lookback -> (val, lb, met). The set of
    # distinct lookbacks IS the "strategies tried" set for this single IS
    # optimisation — window-count-free (effective-trials discipline). Pure
    # write to the scratch dict; no return value, printed line, or
    # control-flow change. Gated so default runs are inert.
    if _bt.OVERFIT_REPORT:
        _bt._runtime_state['_last_trial_sharpes'] = [
            float(v[2]['Sharpe']) for v in eval_cache.values()
            if v is not None and 'Sharpe' in v[2]
        ]

    return selected[1], selected[2]

# 7. MONTE CARLO
def monte_carlo(arr, actual, runs, config: "Optional[_bt.Config]" = None):
    """Monte Carlo bootstrap & shuffle of the realised return series.

    `config` is optional; see `optimiser` docstring for the contract.
    The MC routine itself is config-free (no engine knobs participate),
    but the parameter is accepted for API symmetry with the rest of
    the public surface.
    """
    with _bt.with_config(config):
        return _monte_carlo_impl(arr, actual, runs)


def _monte_carlo_impl(arr, actual, runs):
    N = arr.size
    if N == 0:
        print(" Monte Carlo skipped: no return series provided.")
        return

    # bootstrap and shuffle
    sims_boot  = np.random.choice(arr, size=(runs, N), replace=True)
    sims_shuff = np.array([np.random.permutation(arr) for _ in range(runs)])
    sims_all   = np.concatenate((sims_boot, sims_shuff), axis=0)

    # --- metric calculations ---
    roi    = sims_all.sum(axis=1)
    wins   = np.where(sims_all > 0, sims_all, 0)
    losses = np.where(sims_all <= 0, -sims_all, 0)

    wins_sum   = wins.sum(axis=1)
    losses_sum = losses.sum(axis=1)

    # Profit factor: avoid inf by using NaNlarge finite
    pf = np.divide(
        wins_sum,
        losses_sum,
        out=np.full_like(wins_sum, np.nan),
        where=losses_sum > 0
    )
    pf = np.where(np.isnan(pf), 1e9, pf)

    wr  = np.mean(sims_all > 0, axis=1)
    mw  = np.where(
        wins_sum > 0,
        wins_sum / np.maximum(np.count_nonzero(wins, axis=1), 1),
        0
    )
    ml  = np.where(
        losses_sum > 0,
        losses_sum / np.maximum(np.count_nonzero(losses, axis=1), 1),
        0
    )
    exp = mw * wr - ml * (1 - wr)
    std = sims_all.std(axis=1)
    shp = np.where(std > 0, sims_all.mean(axis=1) / std * sqrt(N), 0)

    # consistency
    weights = np.array([0.0117, 0.0317, 0.0861, 0.2341, 0.6364])
    cons = np.empty(sims_all.shape[0])
    for i, sim in enumerate(sims_all):
        segs = np.array_split(sim, 5)
        rois = [s.sum() for s in segs]
        cons[i] = 0.6 * np.dot(weights, rois) + 0.4 * sim.sum()

    # equity curves & drawdown
    eqs = 1.0 + np.cumsum(sims_all, axis=1)
    hw  = np.maximum.accumulate(eqs, axis=1)
    dd  = ((hw - eqs) / hw).max(axis=1)

    dist = {
        'ROI':         roi,
        'PF':          pf,
        'WinRate':     wr,
        'Exp':         exp,
        'Sharpe':      shp,
        'MaxDrawdown': dd,
        'Consistency': cons
    }

    # prints
    print("\n Monte-Carlo Percentile Ranks vs ACTUAL ")
    for m in _bt.METRICS + ['Consistency']:
        pct = np.mean(dist[m] <= actual[m]) * 100
        print(f"  {m:>12}: {pct:6.1f}th percentile")

    percs  = [5, 25, 50, 75, 95]
    eq_pct = np.percentile(eqs, percs, axis=0)

    print("\n Equity Curve Final Value Percentiles ")
    for p, val in zip(percs, eq_pct[:, -1]):
        print(f"  {p:>2}th pct: {val:9.4f}")
    loss_pct = np.mean(roi < 0) * 100
    dd80_pct = np.mean(dd > 0.80) * 100
    print(f"\nSimulations ending with LOSS:           {loss_pct:5.1f}%")
    print(f"Simulations max-DD > 80 %:              {dd80_pct:5.1f}%\n")

    if _bt.PRINT_EQUITY_CURVE:
        x   = np.arange(1, N + 1)
        fig = plt.figure(figsize=(10, 5))
        ax  = fig.add_subplot(1, 1, 1)

        # actual equity
        ax.plot(_bt.ACCOUNT_SIZE * (1.0 + np.cumsum(arr)),
                color='black', label='Actual equity ($)', linewidth=1.6)

        # simulated bands
        ax.fill_between(x,
                        _bt.ACCOUNT_SIZE * eq_pct[0],
                        _bt.ACCOUNT_SIZE * eq_pct[-1],
                        alpha=0.18, label='5-95% band')
        ax.fill_between(x,
                        _bt.ACCOUNT_SIZE * eq_pct[1],
                        _bt.ACCOUNT_SIZE * eq_pct[-2],
                        alpha=0.28, label='25-75% band')
        ax.plot(x,
                _bt.ACCOUNT_SIZE * eq_pct[2],
                color='royalblue',
                label='Median (50%)',
                linewidth=1.2)

        ax.set_title('Monte-Carlo Equity Curve Bands (USD)')
        ax.legend(); ax.grid(True)
        fig.tight_layout()

        # histograms
        hist_keys = [
          ('ROI',         'ROI'),
          ('MaxDrawdown','Max DD'),
          ('PF',          'Profit Factor'),
          ('Sharpe',      'Sharpe'),
          ('Consistency','Consistency')
        ]
        fig2 = plt.figure(figsize=(14, 8))
        gs   = GridSpec(2, 3, fig2)
        for idx, (k, title) in enumerate(hist_keys):
            r, c = divmod(idx, 3)
            axh  = fig2.add_subplot(gs[r, c])

            data = dist[k]
            # drop any non-finite just in case
            data = data[np.isfinite(data)]
            if data.size == 0:
                axh.text(0.5, 0.5, 'No data', ha='center', va='center')
            else:
                # clip PF for readability
                if k == 'PF':
                    data = np.clip(data, 0, 50)
                axh.hist(data, bins=100)
                axh.axvline(min(actual[k], 50 if k=='PF' else actual[k]),
                            color='red', linestyle='--', linewidth=1.5)

            axh.set_title(title)
            axh.grid(True)

        fig2.add_subplot(gs[1, 2]).axis('off')
        fig2.suptitle('Simulation Metric Distributions')
        fig2.tight_layout(rect=[0, 0, 1, 0.96])
