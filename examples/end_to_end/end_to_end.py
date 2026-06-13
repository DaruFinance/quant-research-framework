#!/usr/bin/env python3
"""Minimal end-to-end walk-through: your strategy -> IS/OOS -> WFO -> overfit report.

The whole pipeline in one file. It defines a small, original strategy, plugs
it into the engine with a single assignment, and runs the default flow:

    IS / OOS baseline
      -> smart-optimised lookback + auto-RRR (in-sample)
      -> rolling walk-forward windows (per-window re-optimise, forward test)
      -> robustness stress tests (entry drift, fee/slippage shock, ...)
      -> overfitting diagnostics (DSR / PSR / MinTRL / MinBTL / PBO).

The overfitting block is opt-in: it is enabled here by setting QRF_OVERFIT=1
*before* importing the engine (the flag is read at import time). Everything
else is the engine's normal default pipeline — no private APIs.

Run:
    python examples/end_to_end/end_to_end.py
    BT_CSV=data/EURUSD_1h.csv python examples/end_to_end/end_to_end.py
"""
import os
import sys
import tempfile
from pathlib import Path

# Enable the opt-in overfitting report. MUST precede `import backtester`,
# which reads QRF_OVERFIT at module load.
os.environ.setdefault("QRF_OVERFIT", "1")

# Headless / automated run: force a non-interactive matplotlib backend BEFORE
# the engine imports pyplot, so nothing blocks on an interactive window.
import matplotlib                                           # noqa: E402
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

# Runs on the full bundled SOL/USDT 1h series (REAL data, ~48k bars, ~7s).
# Override with BT_CSV=... to use your own data, or EXAMPLE_BARS=N to run on
# only the last N bars (faster, fewer walk-forward windows; N must exceed the
# 10_000-bar IS window).
_BUNDLED = ROOT / "data" / "SOLUSDT_1h.csv"
_BARS = int(os.environ.get("EXAMPLE_BARS", "0"))
if "BT_CSV" not in os.environ:
    if _BARS > 0:
        _slice_path = Path(tempfile.gettempdir()) / f"qrf_e2e_SOL_{_BARS}.csv"
        pd.read_csv(_BUNDLED).tail(_BARS).to_csv(_slice_path, index=False)
        os.environ["BT_CSV"] = str(_slice_path)
    else:
        os.environ["BT_CSV"] = str(_BUNDLED)

import backtester as bt                                     # noqa: E402


def sma_cross_rsi(df: pd.DataFrame, lb: int) -> np.ndarray:
    """Dual-SMA crossover with an RSI(14) momentum floor — small and original.

    Every indicator is `.shift(1)`-ed, so `raw[i]` uses only bars <= i-1 and
    obeys the no-look-ahead contract the engine enforces at the ledger level.
    Returns an int8 array of +1 / -1 / 0 (long / short / flat), length len(df).
    The optimiser sweeps `lb`; the slow leg scales with it (3*lb).
    """
    close = df["close"]
    fast = close.rolling(lb, min_periods=lb).mean()
    slow = close.rolling(3 * lb, min_periods=3 * lb).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))

    f1, s1, r1 = fast.shift(1), slow.shift(1), rsi.shift(1)   # bar i decided from i-1
    long_ = (f1 > s1) & (r1 >= 50.0)
    short_ = (f1 < s1) & (r1 < 50.0)

    raw = np.zeros(len(df), dtype=np.int8)
    raw[long_.fillna(False).to_numpy()] = 1
    raw[short_.fillna(False).to_numpy()] = -1
    return raw


# Re-route every engine call site (classic_single_run / optimiser /
# walk_forward / robustness) to our strategy with one assignment.
bt.create_raw_signals = sma_cross_rsi
bt.PRINT_EQUITY_CURVE = False   # automated example: no interactive figures

if __name__ == "__main__":
    print(f"end-to-end | strategy=sma_cross_rsi | csv={os.environ['BT_CSV']} | "
          f"overfit_report={'on' if os.environ.get('QRF_OVERFIT') == '1' else 'off'}")
    bt.main()
