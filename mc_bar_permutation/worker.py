#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import backtester as bt

from common import atomic_json, json_number, sha256


BAR_DTYPE = np.dtype([
    ("time", "<i8"),
    ("open", "<f8"),
    ("high", "<f8"),
    ("low", "<f8"),
    ("close", "<f8"),
    ("volume", "<f8"),
])
ALLOWED_CONFIG = {
    "account_size", "risk_amount", "position_size", "fee_pct",
    "slippage_pct", "funding_fee", "use_sl", "sl_percentage", "use_tp",
    "tp_percentage", "forex_mode", "trade_sessions", "session_start",
    "session_end", "max_hold_bars", "sharpe_mode", "pip_size",
    "clamp_results",
}


def load_binary(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        if handle.read(8) != b"QRFMCB01":
            raise ValueError("permuted bar file has an invalid header")
        count = int.from_bytes(handle.read(8), "little")
        records = np.fromfile(handle, dtype=BAR_DTYPE, count=count)
        if records.size != count or handle.read(1):
            raise ValueError("permuted bar file has an invalid length")
    return pd.DataFrame({
        "time": pd.to_datetime(records["time"], unit="s", utc=True),
        "open": records["open"],
        "high": records["high"],
        "low": records["low"],
        "close": records["close"],
        "volume": records["volume"],
    })


def load_callable(target: str):
    if ":" not in target:
        raise ValueError("python-callable must use the form package.module:function")
    module_name, function_name = target.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{target!r} is not callable")
    return function


def strategy_signals(df: pd.DataFrame, strategy: dict) -> tuple[pd.DataFrame, np.ndarray]:
    lookback = int(strategy.get("lookback", 0))
    if lookback <= 0:
        raise ValueError("strategy.lookback must be positive")
    kind = strategy.get("kind")
    if kind == "ema-crossover":
        prepared = bt.compute_indicators(df, lookback)
        raw = bt.create_raw_signals(prepared, lookback)
    elif kind == "python-callable":
        function = load_callable(str(strategy.get("callable", "")))
        prepared = df.copy()
        raw = function(prepared, lookback, dict(strategy.get("parameters", {})))
        bt._runtime_state["_last_df"] = prepared
        bt._runtime_state["_last_lb"] = lookback
    else:
        raise ValueError(
            f"unsupported strategy kind {kind!r}; choose ema-crossover or python-callable"
        )
    raw = np.asarray(raw)
    if raw.shape != (len(prepared),):
        raise ValueError(f"strategy returned shape {raw.shape}; expected {(len(prepared),)}")
    try:
        finite = np.isfinite(raw).all()
    except TypeError as exc:
        raise ValueError("raw strategy signals must be finite numeric values") from exc
    if not finite:
        raise ValueError("raw strategy signals must be finite")
    if not np.isin(raw, (-1, 0, 1)).all():
        raise ValueError("raw strategy signals must be -1, 0 or 1")
    return prepared, raw.astype(np.int8, copy=False)


def config_from_spec(values: dict) -> bt.Config:
    unknown = sorted(set(values) - ALLOWED_CONFIG)
    if unknown:
        raise ValueError(f"unsupported config fields: {', '.join(unknown)}")
    cfg = bt.Config()
    for name, value in values.items():
        setattr(cfg, name, value)
    cfg.use_monte_carlo = False
    cfg.print_equity_curve = False
    cfg.use_wfo = False
    cfg.optimize_rrr = False
    return cfg


def save_ledger(path: Path, trades) -> None:
    fields = {
        "side": np.asarray([trade[0] for trade in trades], dtype=np.int8),
        "entry_idx": np.asarray([trade[1] for trade in trades], dtype=np.int32),
        "exit_idx": np.asarray([trade[2] for trade in trades], dtype=np.int32),
        "entry_price": np.asarray([trade[3] for trade in trades], dtype=np.float64),
        "exit_price": np.asarray([trade[4] for trade in trades], dtype=np.float64),
        "qty": np.asarray([trade[5] for trade in trades], dtype=np.float64),
        "net_pnl": np.asarray([trade[13] for trade in trades], dtype=np.float64),
        "fee": np.asarray([trade[9] for trade in trades], dtype=np.float64),
        "slippage": np.asarray([trade[10] for trade in trades], dtype=np.float64),
        "funding": np.asarray([trade[11] for trade in trades], dtype=np.float64),
        "gross_pnl": np.asarray([trade[12] for trade in trades], dtype=np.float64),
    }
    np.savez_compressed(path, **fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--barperm-bin", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    started = time.perf_counter()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    bars_path = run_dir / "bars.bin"
    spec_path = Path(args.spec).resolve()
    source_path = Path(args.input).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    subprocess.run(
        [args.barperm_bin, str(source_path), str(bars_path), str(args.seed)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    permuted_hash = sha256(bars_path)
    df = load_binary(bars_path)
    cfg = config_from_spec(dict(spec.get("config", {})))
    with bt.with_config(cfg):
        prepared, raw = strategy_signals(df, dict(spec.get("strategy", {})))
        signals = bt.parse_signals(raw, prepared["time"])
        trades, metrics, _, _, _ = bt.backtest(prepared, signals)
    ledger_path = run_dir / "ledger.npz"
    save_ledger(ledger_path, trades)
    metrics_path = run_dir / "metrics.json"
    atomic_json(metrics_path, {
        "seed": args.seed,
        "bars": len(df),
        "trades": len(trades),
        "metrics": {name: json_number(value) for name, value in metrics.items()},
        "ledger": ledger_path.name,
        "ledger_sha256": sha256(ledger_path),
        "permuted_bars_sha256": permuted_hash,
        "elapsed_seconds": time.perf_counter() - started,
    })
    atomic_json(run_dir / "status.json", {
        "status": "complete",
        "seed": args.seed,
        "source_sha256": sha256(source_path),
        "spec_sha256": sha256(spec_path),
        "metrics_sha256": sha256(metrics_path),
        "ledger_sha256": sha256(ledger_path),
    })
    bars_path.unlink()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
