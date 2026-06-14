#!/usr/bin/env python3
"""Implementation-risk demonstration: same strategy + data + signals + fills +
costs through two popular, independent backtesters (vectorbt, Backtesting.py);
report the divergence in their headline metrics. Long-only SMA(20/50) crossover
on real BTCUSDT 30m. Fills aligned to the signal bar's CLOSE on both engines, so
residual divergence is metric ACCOUNTING, not fill convention."""
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/BTCUSDT_30m.csv"
FAST, SLOW = 20, 50

df = pd.read_csv(CSV)
df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
df = df.set_index("dt")
close = df["close"]

# --- one shared signal: long when SMA_fast > SMA_slow, flat otherwise ---
sma_f = close.rolling(FAST).mean()
sma_s = close.rolling(SLOW).mean()
target_long = (sma_f > sma_s).fillna(False)
# entries = flat->long crossing, exits = long->flat crossing
entries = target_long & ~target_long.shift(1, fill_value=False)
exits = ~target_long & target_long.shift(1, fill_value=False)
n_entries = int(entries.sum())
print(f"data: {len(df):,} bars BTCUSDT 30m | shared signal entries: {n_entries}")


def vbt_run(fee):
    import vectorbt as vbt
    pf = vbt.Portfolio.from_signals(
        close, entries, exits,
        fees=fee, slippage=0.0, init_cash=100_000,
        freq="30min",
    )
    tr = pf.trades
    wr = float(tr.win_rate()) * 100 if len(tr) else float("nan")
    return {
        "total_return_pct": float(pf.total_return()) * 100,
        "n_trades": int(len(tr)),
        "win_rate_pct": wr,
        "sharpe": float(pf.sharpe_ratio()),
        "max_dd_pct": float(pf.max_drawdown()) * 100,
    }


def bt_run(fee):
    from backtesting import Backtest, Strategy
    data = pd.DataFrame({
        "Open": df["open"], "High": df["high"],
        "Low": df["low"], "Close": df["close"],
    }, index=df.index)
    ent = entries.values
    ext = exits.values
    class SmaCross(Strategy):
        def init(self):
            self.i = 0
        def next(self):
            k = len(self.data) - 1  # current bar index
            if ent[k] and not self.position:
                self.buy()
            elif ext[k] and self.position:
                self.position.close()
    bt = Backtest(data, SmaCross, cash=100_000, commission=fee,
                  trade_on_close=True, exclusive_orders=True,
                  finalize_trades=True)
    stats = bt.run()
    return {
        "total_return_pct": float(stats["Return [%]"]),
        "n_trades": int(stats["# Trades"]),
        "win_rate_pct": float(stats["Win Rate [%]"]),
        "sharpe": float(stats["Sharpe Ratio"]),
        "max_dd_pct": float(stats["Max. Drawdown [%]"]),
    }


METRICS = ["total_return_pct", "n_trades", "win_rate_pct", "sharpe", "max_dd_pct"]
for fee, label in [(0.0, "ZERO COST"), (0.0005, "0.05% per fill")]:
    print(f"\n===== regime: {label} =====")
    v = vbt_run(fee)
    b = bt_run(fee)
    print(f"{'metric':<18}{'vectorbt':>14}{'Backtesting.py':>16}{'rel.div':>12}")
    for m in METRICS:
        a, c = v[m], b[m]
        denom = max(abs(a), abs(c), 1e-9)
        rel = abs(a - c) / denom * 100
        print(f"{m:<18}{a:>14.4f}{c:>16.4f}{rel:>11.2f}%")
