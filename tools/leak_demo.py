#!/usr/bin/env python3
"""Look-ahead leak demo — pollute-and-verify on the strategy contract.

The framework's no-look-ahead guarantee is a property, not a code comment:

    raw[..k]  must be invariant under ANY change to bars[k:].

A causal signal can only read the past, so perturbing the future leaves its
earlier values untouched. A signal that peeks ahead changes its *past*
output the instant the future is perturbed — that is the leak, and it is
mechanically detectable, no labels required.

This script demonstrates the property directly on two `RawSignalsFn`s — the
exact contract a user writes against (`raw[i] in {-1,0,+1}`, usable only with
information at or before bar i-1):

  * causal      — EMA(lb) vs EMA(4·lb) crossover, read at i-1. No look-ahead
                  by construction (this is the shipped reference strategy).
  * forward-peek — the identical crossover read `H` bars into the FUTURE
                  (`raw[i]` uses bar i+H). A textbook, realistic bug: acting
                  on a signal H bars before it could be known. (Equivalent
                  failure mode: a *centered* smoother, `rolling(center=True)`
                  / `.shift(-H)`, that averages in bars that haven't happened.)

We build one synthetic OHLC series, run both, NaN-pollute the future
(bars[cut:]), and recompute. We count how many *past* signal bars (indices
< cut) changed. The causal signal leaks 0. The forward-peek signal leaks
exactly its horizon H — the H bars in [cut-H, cut-1] that reach across the
pollution boundary.

This is the same invariant CI enforces over a Hypothesis-generated input
space in the Python reference's tests/test_invariants_property.py (which
applies it to the shipped `parse_signals` / `detect_regimes`); here it is
one deterministic, eyeball-able run with a deliberately-planted bug for
contrast.

Run:
    python tools/leak_demo.py                 # n=800, cut=400, H=4
    python tools/leak_demo.py --horizon 10    # leaks 10 past bars

Exit 0 iff the causal signal leaks 0 bars AND the forward-peek signal is
caught (leaks exactly H). So the demo is itself a regression guard, and it
needs nothing but numpy (no engine import, no network).
"""
from __future__ import annotations

import argparse

import numpy as np


def _synthetic_close(seed: int, n: int) -> np.ndarray:
    """GBM close series (same generator as the property test)."""
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.00005, 0.01, size=n)))


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """adjust=False EMA, NaN-propagating (matches pandas .ewm(adjust=False))."""
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = (1.0 - a) * out[i - 1] + a * x[i]   # NaN at x[i] propagates
    return out


def _sign_cross(fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """+1 where fast>slow, -1 where fast<slow, 0 otherwise (NaN -> 0)."""
    raw = np.zeros(len(fast), dtype=np.int8)
    raw[fast > slow] = 1
    raw[fast < slow] = -1
    return raw


def causal_signal(close: np.ndarray, lb: int) -> np.ndarray:
    """raw[i] = sign(fast[i-1] - slow[i-1]) — reads only the past."""
    fast, slow = _ema(close, lb), _ema(close, 4 * lb)
    raw = np.zeros(len(close), dtype=np.int8)
    raw[1:] = _sign_cross(fast, slow)[:-1]            # shift +1: i reads i-1
    return raw


def forward_peek_signal(close: np.ndarray, lb: int, horizon: int) -> np.ndarray:
    """raw[i] = sign(fast[i+H] - slow[i+H]) — reads H bars into the FUTURE."""
    fast, slow = _ema(close, lb), _ema(close, 4 * lb)
    cross = _sign_cross(fast, slow)
    raw = np.zeros(len(close), dtype=np.int8)
    if horizon < len(close):
        raw[:-horizon] = cross[horizon:]              # shift -H: i reads i+H
    return raw


def _leaked_bars(sig_fn, close: np.ndarray, lb: int, cut: int, *a) -> int:
    """Past bars (< cut) whose signal changes when bars[cut:] are polluted."""
    polluted = close.copy()
    polluted[cut:] = np.nan
    full = sig_fn(close, lb, *a)
    after = sig_fn(polluted, lb, *a)
    return int((full[:cut] != after[:cut]).sum())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=800, help="bars (default 800)")
    ap.add_argument("--cut", type=int, default=400,
                    help="pollution cut = past bars checked (default 400)")
    ap.add_argument("--horizon", type=int, default=4,
                    help="forward-peek bug reads H bars ahead (default 4)")
    ap.add_argument("--lb", type=int, default=50, help="fast-EMA lookback")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n, cut, H = args.n, args.cut, args.horizon
    close = _synthetic_close(args.seed, n)

    causal_leaked = _leaked_bars(causal_signal, close, args.lb, cut)
    peek_leaked = _leaked_bars(forward_peek_signal, close, args.lb, cut, H)

    print(f"look-ahead leak demo  —  n={n} bars, pollute future at cut={cut} "
          f"(lb={args.lb}, seed={args.seed})")
    print()
    print(f"  causal EMA-cross (shipped)    : {causal_leaked:>3d}/{cut} past bars "
          f"changed   {'PASS — no look-ahead' if causal_leaked == 0 else 'FAIL'}")
    print(f"  forward-peek EMA-cross (H={H})  : {peek_leaked:>3d}/{cut} past bars "
          f"changed   {'LEAK CAUGHT' if peek_leaked > 0 else 'NOT CAUGHT'}")
    print()

    ok = causal_leaked == 0 and peek_leaked == H
    if ok:
        print(f"OK — the causal signal is invariant under future pollution; the "
              f"forward-peek bug leaks exactly its horizon ({H} bars).")
        return 0
    print(f"FAIL — expected causal=0 and forward-peek={H}, got "
          f"{causal_leaked} and {peek_leaked}.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
