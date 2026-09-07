#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from shared import (
    COMBINED_CHARGE_RATE,
    DEFAULT_CSV,
    LEDGER_COLUMNS,
    QRF_PY_REPO,
    load_event_sets,
    write_outputs,
)

import numpy as np
import pandas as pd


def _ledger(trades, strategy: str) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    rows = []
    for trade_id, trade in enumerate(trades):
        side, entry_idx, exit_idx, entry_price, exit_price, quantity = trade[:6]
        fee, slippage, funding, gross_pnl, net_pnl = trade[9:14]
        if side != 1 or quantity <= 0:
            raise AssertionError(f"{strategy}: expected a positive long quantity")
        rows.append(
            {
                "strategy": strategy,
                "trade_id": trade_id,
                "side": 1,
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "quantity": 1.0,
                "gross_pnl": float(gross_pnl / quantity),
                "execution_charge": float((fee + slippage + funding) / quantity),
                "net_pnl": float(net_pnl / quantity),
            }
        )
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-repo", type=Path, default=QRF_PY_REPO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bars", type=int, default=None, help="real-data API smoke only")
    args = parser.parse_args()

    sys.path.insert(0, str(args.python_repo))
    import backtester as bt

    qrf_df = bt.load_ohlc(str(args.csv))
    if args.bars is not None:
        qrf_df = qrf_df.iloc[: args.bars].reset_index(drop=True)
    elif len(qrf_df) != 150_000:
        raise AssertionError(f"formal workload requires exactly 150,000 bars, got {len(qrf_df)}")
    event_sets = load_event_sets(args.events, qrf_df)
    if bt.MAX_HOLD_BARS != 0:
        raise AssertionError("QRF module MAX_HOLD_BARS must remain zero")

    config = bt.Config(
        fee_pct=COMBINED_CHARGE_RATE * 100.0,
        slippage_pct=0.0,
        funding_fee=0.0,
        trade_sessions=False,
        use_sl=False,
        use_tp=False,
        optimize_rrr=False,
        use_wfo=False,
        use_monte_carlo=False,
        use_regime_seg=False,
        print_equity_curve=False,
    )
    ledgers = []
    elapsed = []
    with bt.with_config(config):
        for spec, codes in event_sets:
            t0 = time.perf_counter_ns()
            trades, _metrics, _equity, _returns, _carry = bt.backtest(qrf_df, codes)
            elapsed.append((time.perf_counter_ns() - t0) / 1e9)
            ledgers.append(_ledger(trades, spec.name))
    write_outputs(
        out_dir=args.out,
        engine="qrf-python",
        csv_path=args.csv,
        event_path=args.events,
        df=qrf_df,
        event_sets=event_sets,
        ledgers=ledgers,
        engine_seconds=elapsed,
        versions={"qrf-python": bt.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
