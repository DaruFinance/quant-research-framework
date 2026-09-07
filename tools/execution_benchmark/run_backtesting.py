#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from shared import (
    COMBINED_CHARGE_RATE,
    DEFAULT_CSV,
    INIT_CASH,
    LEDGER_COLUMNS,
    load_event_sets,
    load_bars,
    write_outputs,
)

import numpy as np
import pandas as pd
import backtesting
import backtesting.backtesting as backtesting_core
from backtesting import Backtest, Strategy

# Progress rendering is UI, not execution work.  Backtesting.py has no public
# run() switch for it, so replace only its module-local iterator wrapper.
backtesting_core._tqdm = lambda sequence, **_kwargs: sequence


def _strategy_for(codes: np.ndarray):
    class SharedEvents(Strategy):
        def init(self):
            pass

        def next(self):
            # next() is called just before close[i].  Submit the event for i+1;
            # with trade_on_close=False it is filled at open[i+1].
            next_i = len(self.data)
            if next_i >= len(codes):
                return
            code = int(codes[next_i])
            if code == 1:
                self.buy(size=1)
            elif code == 2 and self.position.is_long:
                self.position.close()

    return SharedEvents


def _ledger(stats, strategy: str) -> pd.DataFrame:
    trades = stats["_trades"]
    if not len(trades):
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    quantity = trades["Size"].abs().to_numpy(dtype=float)
    side = np.sign(trades["Size"].to_numpy(dtype=float)).astype(np.int8)
    entry_price = trades["EntryPrice"].to_numpy(dtype=float)
    exit_price = trades["ExitPrice"].to_numpy(dtype=float)
    gross = side * quantity * (exit_price - entry_price)
    charge = trades["Commission"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "strategy": strategy,
            "trade_id": np.arange(len(trades), dtype=np.int64),
            "side": side,
            "entry_idx": trades["EntryBar"].to_numpy(dtype=np.int64),
            "exit_idx": trades["ExitBar"].to_numpy(dtype=np.int64),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "gross_pnl": gross,
            "execution_charge": charge,
            "net_pnl": trades["PnL"].to_numpy(dtype=float),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bars", type=int, default=None, help="real-data API smoke only")
    args = parser.parse_args()

    df = load_bars(args.csv, args.bars)
    event_sets = load_event_sets(args.events, df)
    bt_data = pd.DataFrame(
        {
            "Open": df["open"].to_numpy(dtype=float),
            "High": df["high"].to_numpy(dtype=float),
            "Low": df["low"].to_numpy(dtype=float),
            "Close": df["close"].to_numpy(dtype=float),
        },
        index=pd.DatetimeIndex(df["time"]),
    )
    ledgers = []
    elapsed = []
    for spec, codes in event_sets:
        strategy = _strategy_for(codes)
        t0 = time.perf_counter_ns()
        runner = Backtest(
            bt_data,
            strategy,
            cash=INIT_CASH,
            spread=0.0,
            commission=COMBINED_CHARGE_RATE,
            trade_on_close=False,
            hedging=False,
            exclusive_orders=False,
            finalize_trades=False,
        )
        stats = runner.run()
        elapsed.append((time.perf_counter_ns() - t0) / 1e9)
        ledgers.append(_ledger(stats, spec.name))
    write_outputs(
        out_dir=args.out,
        engine="Backtesting.py",
        csv_path=args.csv,
        event_path=args.events,
        df=df,
        event_sets=event_sets,
        ledgers=ledgers,
        engine_seconds=elapsed,
        versions={"backtesting": backtesting.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
