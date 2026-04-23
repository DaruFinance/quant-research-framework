#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a synthetic OHLC CSV so you can kick the tyres on backtester.py
without needing real market data or network access.

Close prices follow a geometric Brownian motion (drift + vol), and each
bar's open/high/low are derived from close with small random perturbations
that respect the high >= {open, close} >= low invariant. The output CSV
format matches what binance_ohlc_downloader.py emits, i.e. the columns
`time,open,high,low,close` where `time` is UNIX seconds (UTC).

Usage:
    python gen_synthetic.py                     # writes data/SYNTHETIC.csv
    python gen_synthetic.py --bars 100000       # longer series
    python gen_synthetic.py --interval 30m      # 30-minute bars
    python gen_synthetic.py --out data/foo.csv  # custom output path
    python gen_synthetic.py --seed 7            # deterministic output

Then:
    BT_CSV=data/SYNTHETIC.csv python backtester.py
"""

import argparse
import os
import sys
import numpy as np


INTERVAL_SECONDS = {
    "1m":   60,
    "5m":   300,
    "15m":  900,
    "30m":  1800,
    "1h":   3600,
    "4h":   14400,
    "1d":   86400,
}


def generate(n_bars: int, interval_s: int, start_unix: int, seed: int,
             start_price: float = 100.0, drift: float = 0.00002,
             vol: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, size=n_bars)
    close = start_price * np.exp(np.cumsum(returns))

    # open[i] = close[i-1] + tiny gap; open[0] = start_price.
    open_ = np.empty(n_bars)
    open_[0] = start_price
    gaps = rng.normal(0.0, vol * 0.25, size=n_bars - 1)
    open_[1:] = close[:-1] * (1.0 + gaps)

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick_up = np.abs(rng.normal(0.0, vol * 0.8, size=n_bars))
    wick_dn = np.abs(rng.normal(0.0, vol * 0.8, size=n_bars))
    high = body_hi * (1.0 + wick_up)
    low = body_lo * (1.0 - wick_dn)

    times = start_unix + np.arange(n_bars, dtype=np.int64) * interval_s
    return np.column_stack([times, open_, high, low, close])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate a synthetic OHLC CSV.")
    p.add_argument("--bars", type=int, default=50_000,
                   help="number of bars (default: 50000)")
    p.add_argument("--interval", default="1h",
                   choices=sorted(INTERVAL_SECONDS.keys()),
                   help="candle interval (default: 1h)")
    p.add_argument("--out", default="data/SYNTHETIC.csv",
                   help="output CSV path (default: data/SYNTHETIC.csv)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility (default: 42)")
    p.add_argument("--start-price", type=float, default=100.0,
                   help="first-bar price (default: 100.0)")
    p.add_argument("--start-unix", type=int, default=1_600_000_000,
                   help="UNIX seconds for the first bar (default: 2020-09-13 12:26:40 UTC)")
    args = p.parse_args(argv)

    interval_s = INTERVAL_SECONDS[args.interval]
    data = generate(args.bars, interval_s, args.start_unix, args.seed,
                    start_price=args.start_price)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # time as int, OHLC to 8 decimal places to match the Binance downloader format.
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("time,open,high,low,close\n")
        for t, o, h, l, c in data:
            f.write(f"{int(t)},{o:.8f},{h:.8f},{l:.8f},{c:.8f}\n")

    print(f"Wrote {args.bars} bars ({args.interval}) to {args.out}")
    print(f"Next: BT_CSV={args.out} python backtester.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
