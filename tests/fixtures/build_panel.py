#!/usr/bin/env python3
"""Idempotent builder for the DS-PANEL-3 fixture.

Inner-joins SOL/BTC/ETH 1h OHLC on the 1000-bar window that matches
DS-SOL-1K. Inputs live in tests/fixtures/sources/ and are checked in
so this build runs offline; re-running produces a byte-identical
Parquet output.

Output schema (long form, 3000 rows = 1000 timestamps x 3 assets):
    time:int64, asset:str, open:f64, high:f64, low:f64, close:f64

Used by Phase 2 panel-plugin verification.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources"
OUT = HERE / "panel_sol_btc_eth_1h_1000.parquet"

SOURCE_FILES = {
    "SOL": "SOLUSDT_1h_30000_31000.csv",
    "BTC": "BTCUSDT_1h_jan_feb_2024.csv",
    "ETH": "ETHUSDT_1h_jan_feb_2024.csv",
}


def _load(path: Path, asset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["asset"] = asset
    return df[["time", "asset", "open", "high", "low", "close"]]


def build() -> pd.DataFrame:
    per_asset = {a: _load(SOURCES / fname, a) for a, fname in SOURCE_FILES.items()}

    # Inner-join timestamps on the intersection across all three assets.
    shared = set(per_asset["SOL"]["time"])
    for a in ("BTC", "ETH"):
        shared &= set(per_asset[a]["time"])
    shared_sorted = sorted(shared)
    if len(shared_sorted) != 1000:
        print(f"warning: shared timestamp count = {len(shared_sorted)} (expected 1000)",
              file=sys.stderr)

    # Reject any gaps within the window. 1h bars must be contiguous.
    gaps = [
        (a, b) for a, b in zip(shared_sorted[:-1], shared_sorted[1:])
        if b - a != 3600
    ]
    if gaps:
        raise SystemExit(f"panel has timestamp gaps: first 3 = {gaps[:3]}")

    parts = []
    for a in ("SOL", "BTC", "ETH"):
        df = per_asset[a]
        df = df[df["time"].isin(shared_sorted)].sort_values("time").reset_index(drop=True)
        parts.append(df)
    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["time", "asset"]).reset_index(drop=True)
    return panel


def main() -> int:
    panel = build()
    panel.to_parquet(OUT, index=False)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT}")
    print(f"  rows   : {len(panel)}")
    print(f"  assets : {sorted(panel['asset'].unique())}")
    print(f"  span   : {panel['time'].min()} .. {panel['time'].max()}")
    print(f"  sha256 : {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
