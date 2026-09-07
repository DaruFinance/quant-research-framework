#!/usr/bin/env python3
"""Compare two normalized execution-workload result directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXACT_COLUMNS = ("strategy", "trade_id", "side", "entry_idx", "exit_idx")
FLOAT_COLUMNS = (
    "entry_price",
    "exit_price",
    "quantity",
    "gross_pnl",
    "execution_charge",
    "net_pnl",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    left_meta = json.loads((args.left / "results.json").read_text())
    right_meta = json.loads((args.right / "results.json").read_text())
    if left_meta["bars"] != right_meta["bars"]:
        raise AssertionError("bar counts differ")
    if left_meta["combined_execution_charge_per_fill"] != right_meta["combined_execution_charge_per_fill"]:
        raise AssertionError("execution charges differ")
    left_results = {row["name"]: row for row in left_meta["results"]}
    right_results = {row["name"]: row for row in right_meta["results"]}
    if left_results.keys() != right_results.keys():
        raise AssertionError("strategy sets differ")

    max_abs = {column: 0.0 for column in FLOAT_COLUMNS}
    total_trades = 0
    rows = []
    for name in left_results:
        left_result = left_results[name]
        right_result = right_results[name]
        if left_result["event_sha256"] != right_result["event_sha256"]:
            raise AssertionError(f"{name}: event hashes differ")
        left = pd.read_csv(left_result["ledger"])
        right = pd.read_csv(right_result["ledger"])
        if len(left) != len(right):
            raise AssertionError(f"{name}: trade counts differ ({len(left)} != {len(right)})")
        total_trades += len(left)
        for column in EXACT_COLUMNS:
            if not left[column].equals(right[column]):
                raise AssertionError(f"{name}: exact column differs: {column}")
        for column in FLOAT_COLUMNS:
            left_values = left[column].to_numpy(dtype=float)
            right_values = right[column].to_numpy(dtype=float)
            if len(left_values):
                delta = float(np.max(np.abs(left_values - right_values)))
                max_abs[column] = max(max_abs[column], delta)
            if not np.allclose(left_values, right_values, rtol=args.rtol, atol=args.atol):
                raise AssertionError(f"{name}: float column differs: {column}")
        rows.append({"name": name, "trades": len(left), "status": "match"})

    report = {
                "left_engine": left_meta["engine"],
                "right_engine": right_meta["engine"],
                "bars": left_meta["bars"],
                "strategies": len(rows),
                "trades": total_trades,
                "rtol": args.rtol,
                "atol": args.atol,
                "max_abs_delta": max_abs,
                "status": "match",
                "results": rows,
            }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
