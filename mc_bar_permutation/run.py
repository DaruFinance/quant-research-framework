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


def complete_status(run_dir: Path, source_hash: str, spec_hash: str, seed: int) -> bool:
    path = run_dir / "status.json"
    if not path.exists():
        return False
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        status.get("status") == "complete"
        and status.get("source_sha256") == source_hash
        and status.get("spec_sha256") == spec_hash
        and status.get("seed") == seed
    )


def run_one(command: list[str], environment: dict[str, str], run_dir: Path,
            seed: int, source_hash: str, spec_hash: str) -> dict:
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
    elif (run_dir / "status.json").exists():
        return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    else:
        metrics = run_dir / "metrics.json"
        ledger = run_dir / "ledger.bin"
        status = {
            "status": "complete", "seed": seed,
            "source_sha256": source_hash, "spec_sha256": spec_hash,
            "metrics_sha256": sha256(metrics), "ledger_sha256": sha256(ledger),
            "elapsed_seconds": time.perf_counter() - started,
        }
    atomic_json(run_dir / "status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run expensive full-backtest bar permutations with frozen strategy parameters."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--spec", default=str(Path(__file__).with_name("spec.example.json")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--barperm-bin")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    if args.seed < 0 or args.seed + args.runs - 1 > 2**64 - 1:
        parser.error("seed range must fit unsigned 64-bit integers")

    root = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.input).resolve()
    spec = Path(args.spec).resolve()
    specification = json.loads(spec.read_text(encoding="utf-8"))
    strategy = dict(specification.get("strategy", {}))
    config = dict(specification.get("config", {}))
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
        if complete_status(run_dir, source_hash, spec_hash, seed):
            statuses[index] = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            continue
        if strategy.get("kind") == "ema-crossover":
            if config.get("use_sl") or config.get("use_tp"):
                raise ValueError(
                    "the built-in Rust path requires use_sl=false and use_tp=false; "
                    "use python-callable for a Python fallback with stop exits"
                )
            command = [
                str(barperm), "backtest", "--input", str(source),
                "--output", str(run_dir), "--seed", str(seed),
                "--strategy", "ema-crossover", "--lookback", str(strategy["lookback"]),
                "--fee-pct", str(config["fee_pct"]),
                "--slippage-pct", str(config["slippage_pct"]),
                "--funding-fee", str(config["funding_fee"]),
                "--account-size", str(config["account_size"]),
                "--position-size", str(config["position_size"]),
                "--sharpe-mode", str(config.get("sharpe_mode", "trade")),
            ]
        else:
            command = [
                sys.executable, str(worker), "--barperm-bin", str(barperm),
                "--input", str(source), "--spec", str(spec),
                "--run-dir", str(run_dir), "--seed", str(seed),
            ]
        pending.append((index, seed, run_dir, command))

    print(
        f"bar-permutation is expensive: {args.runs} complete strategy backtests "
        f"over the full input, workers={args.workers}",
        flush=True,
    )
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                run_one, command, environment, run_dir, seed, source_hash, spec_hash
            ): (index, seed, run_dir)
            for index, seed, run_dir, command in pending
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
