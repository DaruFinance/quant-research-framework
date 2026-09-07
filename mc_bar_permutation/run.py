#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import atomic_json, sha256


DEFAULT_RUNS = 500
MAX_WORKERS = 4
ALLOWED_CONFIG = {
    "account_size", "position_size", "fee_pct", "slippage_pct",
    "funding_fee", "use_sl", "sl_percentage", "use_tp", "tp_percentage",
    "forex_mode", "max_hold_bars", "sharpe_mode",
}
CALLBACK_CONFIG = ALLOWED_CONFIG | {
    "risk_amount", "trade_sessions", "session_start", "session_end",
    "pip_size", "clamp_results",
}


def load_spec(path: Path) -> tuple[dict, dict]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    unknown_top = sorted(set(spec) - {"strategy", "config"})
    if unknown_top:
        raise ValueError(f"unsupported top-level fields: {', '.join(unknown_top)}")
    strategy = dict(spec.get("strategy", {}))
    kind = strategy.get("kind")
    if kind not in {"ema-crossover", "python-callable"}:
        raise ValueError(
            f"unsupported strategy kind {kind!r}; choose ema-crossover or python-callable"
        )
    allowed_strategy = {"kind", "lookback", "parameters"}
    if kind == "python-callable":
        allowed_strategy.add("callable")
    unknown_strategy = sorted(set(strategy) - allowed_strategy)
    if unknown_strategy:
        raise ValueError(f"unsupported strategy fields: {', '.join(unknown_strategy)}")
    parameters = strategy.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("strategy.parameters must be an object")
    if kind == "ema-crossover" and parameters:
        raise ValueError("ema-crossover does not accept strategy.parameters")
    if kind == "python-callable" and not strategy.get("callable"):
        raise ValueError("python-callable requires package.module:function in strategy.callable")
    lookback = int(strategy.get("lookback", 0))
    if lookback <= 0:
        raise ValueError("strategy.lookback must be positive")
    config = dict(spec.get("config", {}))
    allowed_config = CALLBACK_CONFIG if kind == "python-callable" else ALLOWED_CONFIG
    unknown_config = sorted(set(config) - allowed_config)
    if unknown_config:
        raise ValueError(f"unsupported config fields: {', '.join(unknown_config)}")
    missing = sorted((ALLOWED_CONFIG - {"max_hold_bars", "sharpe_mode"}) - set(config))
    if missing:
        raise ValueError(f"missing config fields: {', '.join(missing)}")
    return strategy, config


def build_generator(root: Path, target: Path) -> Path:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(root / "rust" / "Cargo.toml")],
        check=True,
        env=env,
    )
    name = "qrf-bar-permute.exe" if os.name == "nt" else "qrf-bar-permute"
    return target / "release" / name


def complete_status(
    run_dir: Path, source_hash: str, spec_hash: str, seed: int, worker_hash: str,
) -> bool:
    path = run_dir / "status.json"
    if not path.exists():
        return False
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metrics = run_dir / "metrics.json"
    ledger_name = status.get("ledger_file")
    if not isinstance(ledger_name, str) or Path(ledger_name).name != ledger_name:
        return False
    ledger = run_dir / ledger_name
    if not metrics.is_file() or not ledger.is_file():
        return False
    return (
        status.get("status") == "complete"
        and status.get("source_sha256") == source_hash
        and status.get("spec_sha256") == spec_hash
        and status.get("seed") == seed
        and status.get("worker_sha256") == worker_hash
        and status.get("metrics_sha256") == sha256(metrics)
        and status.get("ledger_sha256") == sha256(ledger)
    )


def run_one(command: list[str], environment: dict[str, str], run_dir: Path,
            seed: int, source_hash: str, spec_hash: str, worker_hash: str) -> dict:
    for name in ("metrics.json", "ledger.bin", "ledger.npz", "status.json"):
        (run_dir / name).unlink(missing_ok=True)
    started = time.perf_counter()
    process = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment,
    )
    if process.returncode != 0:
        status = {
            "status": "failed",
            "seed": seed,
            "returncode": process.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "stderr": process.stderr[-4000:],
        }
    else:
        metrics = run_dir / "metrics.json"
        ledger = run_dir / ("ledger.bin" if (run_dir / "ledger.bin").is_file() else "ledger.npz")
        status = {
            "status": "complete", "seed": seed,
            "source_sha256": source_hash, "spec_sha256": spec_hash,
            "metrics_sha256": sha256(metrics), "ledger_sha256": sha256(ledger),
            "ledger_file": ledger.name, "worker_sha256": worker_hash,
            "elapsed_seconds": time.perf_counter() - started,
        }
    atomic_json(run_dir / "status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run expensive full-backtest bar permutations with frozen strategy parameters."
    )
    parser.add_argument("--mode", choices=("permutation", "resampling", "bar-permutation"),
                        default="bar-permutation")
    parser.add_argument("--input")
    parser.add_argument("--returns", help="JSON array or one-return-per-line file for trade modes")
    parser.add_argument("--spec", default=str(Path(__file__).with_name("spec.example.json")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--barperm-bin")
    parser.add_argument("--forex", action="store_true", help="use absolute-R trade drawdown")
    args = parser.parse_args()
    if args.runs is None:
        args.runs = DEFAULT_RUNS if args.mode == "bar-permutation" else 1_000
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    if args.seed < 0 or args.seed + args.runs - 1 > 2**64 - 1:
        parser.error("seed range must fit unsigned 64-bit integers")

    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode != "bar-permutation":
        if not args.returns:
            parser.error("--returns is required for permutation and resampling")
        returns_path = Path(args.returns).resolve()
        text = returns_path.read_text(encoding="utf-8").strip()
        values = json.loads(text) if text.startswith("[") else [
            float(line) for line in text.splitlines() if line.strip()
        ]
        sys.path.insert(0, str(root.parent))
        from backtester.monte_carlo import run_trade_monte_carlo
        result = run_trade_monte_carlo(
            values, mode=args.mode, runs=args.runs, seed=args.seed,
            forex_mode=args.forex,
        )
        manifest = result.summary()
        manifest.update({
            "returns": str(returns_path),
            "returns_sha256": sha256(returns_path),
            "warning": "trade-return modes do not rebuild market bars",
        })
        atomic_json(output / "manifest.json", manifest)
        return
    if not args.input:
        parser.error("--input is required for bar-permutation")
    source = Path(args.input).resolve()
    spec = Path(args.spec).resolve()
    strategy, config = load_spec(spec)
    source_hash = sha256(source)
    spec_hash = sha256(spec)
    barperm = Path(args.barperm_bin).resolve() if args.barperm_bin else build_generator(
        root, output / ".cargo-target"
    )
    worker = root / "worker.py"
    environment = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[name] = "1"

    pending = []
    statuses = {}
    for index in range(args.runs):
        seed = args.seed + index
        run_dir = output / "runs" / f"{index:06d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        worker_hash = sha256(barperm)
        if strategy.get("kind") == "python-callable":
            worker_hash = ":".join((
                worker_hash, sha256(worker),
                sha256(root.parent / "backtester" / "__init__.py"),
            ))
        if complete_status(run_dir, source_hash, spec_hash, seed, worker_hash):
            statuses[index] = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            continue
        if strategy.get("kind") == "ema-crossover":
            command = [
                str(barperm), "backtest", "--input", str(source),
                "--output", str(run_dir), "--seed", str(seed),
                "--strategy", "ema-crossover", "--lookback", str(strategy["lookback"]),
                "--fee-pct", str(config["fee_pct"]),
                "--slippage-pct", str(config["slippage_pct"]),
                "--funding-fee", str(config["funding_fee"]),
                "--account-size", str(config["account_size"]),
                "--position-size", str(config["position_size"]),
                "--use-sl", str(config["use_sl"]).lower(),
                "--sl-percentage", str(config["sl_percentage"]),
                "--use-tp", str(config["use_tp"]).lower(),
                "--tp-percentage", str(config["tp_percentage"]),
                "--forex", str(config["forex_mode"]).lower(),
                "--max-hold-bars", str(config.get("max_hold_bars", 0)),
                "--sharpe-mode", str(config.get("sharpe_mode", "trade")),
            ]
        else:
            command = [
                sys.executable, str(worker), "--barperm-bin", str(barperm),
                "--input", str(source), "--spec", str(spec),
                "--run-dir", str(run_dir), "--seed", str(seed),
            ]
        pending.append((index, seed, run_dir, command, worker_hash))

    print(
        f"bar-permutation is expensive: {args.runs} complete strategy backtests "
        f"over the full input, workers={args.workers}",
        flush=True,
    )
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                run_one, command, environment, run_dir, seed, source_hash, spec_hash, worker_hash
            ): (index, seed, run_dir)
            for index, seed, run_dir, command, worker_hash in pending
        }
        for future in concurrent.futures.as_completed(future_map):
            index, seed, run_dir = future_map[future]
            status = future.result()
            statuses[index] = status
            completed = sum(value.get("status") == "complete" for value in statuses.values())
            failed = sum(value.get("status") == "failed" for value in statuses.values())
            print(f"finished={len(statuses)}/{args.runs} complete={completed} failed={failed}", flush=True)

    ordered = [statuses[index] for index in range(args.runs)]
    completed = sum(status.get("status") == "complete" for status in ordered)
    failed = args.runs - completed
    manifest = {
        "mode": "bar-permutation",
        "method": "independent no-replacement shuffles of close log returns and OHLCV templates",
        "timestamps": "preserved",
        "volume": "moves with its OHLC template",
        "strategy_parameters": "frozen; signals regenerated on every permuted path",
        "warning": "expensive: every iteration runs the complete backtest",
        "source": str(source),
        "source_sha256": source_hash,
        "spec": str(spec),
        "spec_sha256": spec_hash,
        "seed": args.seed,
        "requested_runs": args.runs,
        "completed_runs": completed,
        "failed_runs": failed,
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - started,
        "runs": ordered,
        "p_values_reported": False,
    }
    atomic_json(output / "manifest.json", manifest)
    if failed:
        raise SystemExit(f"{failed} bar-permutation runs failed; no p-value was produced")


if __name__ == "__main__":
    main()
