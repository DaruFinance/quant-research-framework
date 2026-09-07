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
import vectorbt as vbt


def _ledger(portfolio, strategy: str) -> pd.DataFrame:
    records = portfolio.trades.records
    if not len(records):
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    side = np.where(records["direction"].to_numpy() == 0, 1, -1)
    quantity = records["size"].to_numpy(dtype=float)
    entry_price = records["entry_price"].to_numpy(dtype=float)
    exit_price = records["exit_price"].to_numpy(dtype=float)
    gross = side * quantity * (exit_price - entry_price)
    charge = (
        records["entry_fees"].to_numpy(dtype=float)
        + records["exit_fees"].to_numpy(dtype=float)
    )
    return pd.DataFrame(
        {
            "strategy": strategy,
            "trade_id": np.arange(len(records), dtype=np.int64),
            "side": side,
            "entry_idx": records["entry_idx"].to_numpy(dtype=np.int64),
            "exit_idx": records["exit_idx"].to_numpy(dtype=np.int64),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "gross_pnl": gross,
            "execution_charge": charge,
            "net_pnl": records["pnl"].to_numpy(dtype=float),
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
    open_price = df["open"].to_numpy(dtype=float)
    close_price = df["close"].to_numpy(dtype=float)
    ledgers = []
    elapsed = []
    for spec, codes in event_sets:
        entries = codes == 1
        exits = codes == 2
        t0 = time.perf_counter_ns()
        portfolio = vbt.Portfolio.from_signals(
            close=close_price,
            entries=entries,
            exits=exits,
            price=open_price,
            size=1.0,
            fees=COMBINED_CHARGE_RATE,
            slippage=0.0,
            init_cash=INIT_CASH,
            accumulate=False,
            freq="30min",
        )
        elapsed.append((time.perf_counter_ns() - t0) / 1e9)
        ledgers.append(_ledger(portfolio, spec.name))
    write_outputs(
        out_dir=args.out,
        engine="vectorbt",
        csv_path=args.csv,
        event_path=args.events,
        df=df,
        event_sets=event_sets,
        ledgers=ledgers,
        engine_seconds=elapsed,
        versions={"vectorbt": vbt.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
