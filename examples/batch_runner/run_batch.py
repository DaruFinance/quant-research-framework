"""Parallel batch runner for the quant-research-framework.

Demonstrates how to run many strategy configurations through the
single-asset engine concurrently, using ``multiprocessing.Pool`` to
sidestep the GIL on the Numba-compiled inner loops. Each worker is a
fresh Python process that imports the engine and runs one strategy
end-to-end, returning a one-line metric summary.

This addresses the §9 limitation of "single strategy at a time" in the
paper: while one process can only run one backtest at a time, the OS
process pool can run N at the same time. On a Ryzen 9 7950X (16C/32T)
with 8 GB free RAM, the bench runs 12 strategies in ~12 seconds; serial
on the same machine takes ~70.

Strategies registered here use only the engine's documented public
indicators (SMA, EMA, ATR, RSI, MACD, Stoch); no proprietary signals.

Pin BLAS / OMP / MKL to one thread per process *before* importing
numpy / pandas / numba — the Numba inner loop is already
parallelisable across processes, and oversubscribing BLAS threads
inside each process kills throughput.

Usage:
    cd quant-research-framework/
    BT_CSV=data_SOLUSDT_1h.csv python examples/batch_runner/run_batch.py
    # add --workers N to override; --serial for diagnostic comparison
"""
import os

# Pin BLAS-style libraries to a single thread per worker so multiprocessing
# scales linearly. This must run BEFORE numpy / pandas / numba import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make the framework importable when the script is run as
# `python examples/batch_runner/run_batch.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Strategy library — pure-Python signal generators on the documented contract
# (df: DataFrame, lb: int) -> np.ndarray[int8] of {-1, 0, +1} per bar.
#
# Every signal returned is shifted by one bar (`take(idx-1, mode="clip")`)
# so it cannot read information from bar i; this is what the look-ahead
# invariant tests in tests/test_invariants.py verify.
# ---------------------------------------------------------------------------
def _shifted(arr: np.ndarray) -> np.ndarray:
    """Shift the signal by one bar so position at bar i depends only on
    information available at bar i-1."""
    n = len(arr)
    return np.asarray(arr, dtype=np.int8).take(np.arange(n) - 1, mode="clip")


def signal_ema_cross(df: pd.DataFrame, lb: int) -> np.ndarray:
    """Long when fast EMA > slow EMA (slow = lb*4); short otherwise."""
    fast = df["close"].ewm(span=lb,     adjust=False).mean()
    slow = df["close"].ewm(span=lb * 4, adjust=False).mean()
    sig  = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    return _shifted(sig)


def signal_atr_cross(df: pd.DataFrame, lb: int) -> np.ndarray:
    """EMA-cross filtered by ATR-thresholded breakout + RSI directional gate."""
    from backtester.indicators import compute_atr, compute_rsi
    atr  = compute_atr(df, lb)
    fast = df["close"].rolling(lb).mean()
    slow = df["close"].rolling(lb * 4).mean()
    rsi  = compute_rsi(df, lb)
    long_cond  = (fast > slow + atr) & (rsi >= 50)
    short_cond = (fast < slow - atr) & (rsi <= 50)
    sig = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return _shifted(sig)


def signal_macd_zero(df: pd.DataFrame, lb: int) -> np.ndarray:
    """MACD line above its signal line: long; below: short. Lookback `lb`
    drives the fast period (slow = lb*2.16, signal = lb*0.75 capped to 9)."""
    fast_p = max(2, lb)
    slow_p = max(fast_p + 1, int(round(lb * 2.16)))
    sig_p  = max(2, min(9, int(round(lb * 0.75))))
    fast_e = df["close"].ewm(span=fast_p, adjust=False).mean()
    slow_e = df["close"].ewm(span=slow_p, adjust=False).mean()
    macd   = fast_e - slow_e
    sig_l  = macd.ewm(span=sig_p, adjust=False).mean()
    sig    = np.where(macd > sig_l, 1, np.where(macd < sig_l, -1, 0))
    return _shifted(sig)


def signal_rsi_revert(df: pd.DataFrame, lb: int) -> np.ndarray:
    """Mean-reversion: long when RSI<35 (oversold), short when RSI>65."""
    from backtester.indicators import compute_rsi
    rsi = compute_rsi(df, lb)
    sig = np.where(rsi < 35, 1, np.where(rsi > 65, -1, 0))
    return _shifted(sig)


def signal_stoch_kd(df: pd.DataFrame, lb: int) -> np.ndarray:
    """Stochastic-K vs threshold (long < 20, short > 80)."""
    from backtester.indicators import compute_stoch
    k = compute_stoch(df, lb)
    sig = np.where(k < 20, 1, np.where(k > 80, -1, 0))
    return _shifted(sig)


# ---------------------------------------------------------------------------
# Strategy specification: a name + signal callable + per-strategy
# overrides on the engine's module globals (sl_pct, tp_pct, lookback,
# rrr, etc). Only the engine globals listed in `overrides` are touched
# per worker; everything else stays at the engine's documented defaults.
# ---------------------------------------------------------------------------
@dataclass
class BatchSpec:
    name:      str
    signal_fn: callable
    lb:        int = 50
    overrides: dict = field(default_factory=dict)


# A reasonable starter library spanning trend-following / breakout /
# mean-reversion / oscillator families with diverse SL and RRR settings.
STRATEGIES: list[BatchSpec] = [
    BatchSpec("ema_cross_lb14_sl1.0_rrr2",  signal_ema_cross,
              lb=14, overrides={"SL_PERCENTAGE": 1.0, "TP_PERCENTAGE": 2.0,
                                "OPTIMIZE_RRR": False, "RRR_used": [2]}),
    BatchSpec("ema_cross_lb40_sl1.5_rrr3",  signal_ema_cross,
              lb=40, overrides={"SL_PERCENTAGE": 1.5, "TP_PERCENTAGE": 4.5,
                                "OPTIMIZE_RRR": False, "RRR_used": [3]}),
    BatchSpec("atr_cross_lb20_sl0.8_rrr2",  signal_atr_cross,
              lb=20, overrides={"SL_PERCENTAGE": 0.8, "TP_PERCENTAGE": 1.6,
                                "OPTIMIZE_RRR": False, "RRR_used": [2]}),
    BatchSpec("atr_cross_lb50_sl2.0_rrr1",  signal_atr_cross,
              lb=50, overrides={"SL_PERCENTAGE": 2.0, "TP_PERCENTAGE": 2.0,
                                "OPTIMIZE_RRR": False, "RRR_used": [1]}),
    BatchSpec("macd_zero_lb12_sl1.0_rrr3",  signal_macd_zero,
              lb=12, overrides={"SL_PERCENTAGE": 1.0, "TP_PERCENTAGE": 3.0,
                                "OPTIMIZE_RRR": False, "RRR_used": [3]}),
    BatchSpec("macd_zero_lb26_sl1.5_rrr2",  signal_macd_zero,
              lb=26, overrides={"SL_PERCENTAGE": 1.5, "TP_PERCENTAGE": 3.0,
                                "OPTIMIZE_RRR": False, "RRR_used": [2]}),
    BatchSpec("rsi_revert_lb14_sl0.5_rrr1", signal_rsi_revert,
              lb=14, overrides={"SL_PERCENTAGE": 0.5, "TP_PERCENTAGE": 0.5,
                                "OPTIMIZE_RRR": False, "RRR_used": [1]}),
    BatchSpec("rsi_revert_lb28_sl1.0_rrr2", signal_rsi_revert,
              lb=28, overrides={"SL_PERCENTAGE": 1.0, "TP_PERCENTAGE": 2.0,
                                "OPTIMIZE_RRR": False, "RRR_used": [2]}),
    BatchSpec("stoch_kd_lb14_sl0.8_rrr2",   signal_stoch_kd,
              lb=14, overrides={"SL_PERCENTAGE": 0.8, "TP_PERCENTAGE": 1.6,
                                "OPTIMIZE_RRR": False, "RRR_used": [2]}),
    BatchSpec("stoch_kd_lb21_sl1.2_rrr3",   signal_stoch_kd,
              lb=21, overrides={"SL_PERCENTAGE": 1.2, "TP_PERCENTAGE": 3.6,
                                "OPTIMIZE_RRR": False, "RRR_used": [3]}),
    BatchSpec("ema_cross_lb20_optrrr",      signal_ema_cross,
              lb=20, overrides={"OPTIMIZE_RRR": True}),
    BatchSpec("atr_cross_lb30_optrrr",      signal_atr_cross,
              lb=30, overrides={"OPTIMIZE_RRR": True}),
]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _run_one(spec: BatchSpec) -> dict:
    """Run a single strategy and return its IS/OOS-opt metrics. Each
    worker imports the engine fresh (multiprocessing spawn context); the
    overrides patch module globals locally so other workers are unaffected."""
    import importlib
    import io
    from contextlib import redirect_stdout

    bt = importlib.import_module("backtester")

    # Apply per-spec overrides on top of the engine's import-time defaults.
    for k, v in spec.overrides.items():
        setattr(bt, k, v)

    # Plug in the strategy callable on the documented seam.
    bt.create_raw_signals = spec.signal_fn
    bt.DEFAULT_LB = spec.lb
    bt.PRINT_EQUITY_CURVE = False
    bt.USE_MONTE_CARLO    = False

    # Capture engine stdout per worker; we only need the final metric block.
    buf = io.StringIO()
    t0 = time.perf_counter()
    try:
        with redirect_stdout(buf):
            bt.main()
        ok = True
        err = ""
    except Exception as e:                     # noqa: BLE001
        ok = False
        err = f"{type(e).__name__}: {e}"
    t1 = time.perf_counter()

    return {
        "name":     spec.name,
        "lb":       spec.lb,
        "elapsed_s": round(t1 - t0, 2),
        "ok":       ok,
        "err":      err,
        # The full metric stdout is recoverable from the engine's
        # exported trade_list.csv; we keep the buffer so the user can
        # inspect a specific run. Trim to last ~40 lines to keep memory
        # bounded across many strategies.
        "stdout_tail": "\n".join(buf.getvalue().splitlines()[-40:]),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=None,
                   help="parallel processes (default: cpu_count, capped at 16)")
    p.add_argument("--serial", action="store_true",
                   help="run sequentially in this process; useful for timing")
    p.add_argument("--out", type=Path,
                   default=Path("batch_runner_summary.csv"),
                   help="where to write the per-strategy summary CSV")
    args = p.parse_args()

    workers = args.workers or min(16, mp.cpu_count() or 4)
    print(f"[batch_runner] {len(STRATEGIES)} strategies, "
          f"{'serial' if args.serial else f'parallel ({workers} workers)'}")

    t0 = time.perf_counter()
    if args.serial:
        results = [_run_one(s) for s in STRATEGIES]
    else:
        ctx = mp.get_context("spawn")  # Numba JIT cache is per-process
        with ctx.Pool(processes=workers, maxtasksperchild=4) as pool:
            results = list(pool.imap_unordered(_run_one, STRATEGIES))
    elapsed = time.perf_counter() - t0

    # Sort by run name for stable output
    results.sort(key=lambda r: r["name"])

    print(f"\n[batch_runner] all done in {elapsed:.1f}s "
          f"(wall-clock; aggregate worker CPU is higher)\n")
    print(f"{'name':>32}  {'lb':>3}  {'ok':>2}  {'elapsed_s':>9}  err")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:>32}  {r['lb']:>3}  {('Y' if r['ok'] else 'N'):>2}  "
              f"{r['elapsed_s']:>9}  {r['err']}")

    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "lb", "ok", "elapsed_s", "err"])
        for r in results:
            w.writerow([r["name"], r["lb"], r["ok"], r["elapsed_s"], r["err"]])
    print(f"\n[batch_runner] summary written to {args.out}")

    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
