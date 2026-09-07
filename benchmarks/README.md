# Performance measurements

These single-run observations are recorded in
[`2026-09-07-results.json`](2026-09-07-results.json). For current code, use
this record rather than the paper's older workload. Each engine started in a
fresh process, although OS and library caches were not reset.

Python's first actual in-process JIT work is included. No sample is presented
as a cold-cache or steady-state measurement.

## Data

Both workloads use 150,000 real BTCUSDT spot 30-minute bars from 2017-11-01
00:00 UTC through 2026-05-28 10:30 UTC. The final CSV SHA-256 is
`6ba414b666f71b20e38915dc9a99144d9f71c24bb1007d53be281b9cd861ce87`.

Rust's repository contains the first 146,168 rows. Rebuild the measured file
without altering that fixture:

```bash
cp quant-research-framework-rs/data/BTCUSDT_30m.csv BTCUSDT_30m_150k.csv
python quant-research-framework/binance_ohlc_downloader.py \
  --symbol BTCUSDT --interval 30m --market spot --source api --resume \
  --until 2026-05-28T10:30:00 --out BTCUSDT_30m_150k.csv
sha256sum BTCUSDT_30m_150k.csv
```

Binance's public `/api/v3/klines` endpoint appends 3,832 rows using a fixed
final `endTime` of `1779964200000`. The source fixture retains its 29
historical cadence gaps.

## Full walk-forward workload

One fresh Python process and one fresh Rust process each run ten fixed
configurations, consisting of five signal families with two lookbacks each.
Every configuration covers 10,000 in-sample bars and 140,000 out-of-sample
bars in 28 windows. The run includes fees, slippage, funding, dynamic RRR
selection and persisted trade ledgers.

Classic baseline, Monte Carlo and stress-test overlays are outside this
measurement. Python's first Numba compilation is included; Rust compilation
is excluded.

| Engine | Wall | User + sys | Peak RSS |
|---|---:|---:|---:|
| Python reference | 112.37 s | 109.60 s | 287.37 MiB |
| Rust port | 7.20 s | 5.54 s | 10.75 MiB |

This batch recorded 15.61 times less wall time and 26.73 times less peak
memory for Rust. All ten ledger row counts matched, with a largest
deterministic metric difference of `6.67e-14`.

Runners are `tools/bench_python_wfo.py` and `examples/wfo_benchmark.rs` in the
Rust repository.

From the Rust repository, after setting `QRF_PY_DIR` to the sibling Python
checkout, run:

```bash
python tools/bench_paper.py --csv ../BTCUSDT_30m_150k.csv --out wfo-run
```

## Matched execution workload

This narrower benchmark compares QRF with the two frameworks used by the
repository's existing implementation-risk script: vectorbt and
Backtesting.py. It is an execution-kernel test, not a WFO comparison.

A frozen event file supplies the same ten long-only event streams to all four
engines. Each process loads the same OHLC and event files, then runs ten serial
full-history execution passes. Event generation and strategy calculation are
outside the timed process.

WFO, optimization, SL/TP and funding are also excluded. Trades execute at the
next bar's open with a combined 7 bp charge per fill, consisting of a 5 bp
taker fee plus a 2 bp slippage proxy with no price offset.

| Engine | Version | Whole-process wall | Engine loop | Peak RSS |
|---|---:|---:|---:|---:|
| QRF Rust | 0.7.6 | 0.46 s | 0.0104 s | 17.50 MiB |
| QRF Python | 0.7.6 | 2.66 s | 0.414 s | 280.10 MiB |
| Backtesting.py | 0.6.5 | 7.43 s | 5.92 s | 180.22 MiB |
| vectorbt | 1.0.0 | 8.43 s | 3.06 s | 479.75 MiB |

`/usr/bin/time` resolves to 0.01 seconds, limiting the precision of the
0.46-second observation. Across all four engines, all 20,137 trades matched on
chronology, side, prices and normalized quantity. The largest numeric ledger
difference was `7.28e-12`.

Portable adapters, event generator, contract and ledger verifier are in
`tools/execution_benchmark/` in the Python repository. Rust's adapter is
`examples/execution_benchmark.rs`. Generate the frozen events before
starting any timed process:

```bash
python quant-research-framework/tools/execution_benchmark/write_events.py \
  --csv BTCUSDT_30m_150k.csv --out events_150k.csv

cd quant-research-framework-rs
cargo build --locked --release --example execution_benchmark \
  --features benchmark-internal
```

Run each engine once and serially with the thread environment shown below.
Each Python adapter accepts the CSV, frozen events and output directory; the
Rust binary accepts `OHLC EVENT_CSV OUT_DIR`.

After each run, compare its output directory with
`tools/execution_benchmark/verify_ledgers.py`.

Measured competitor versions were installed with:

```bash
python -m pip install vectorbt==1.0.0 backtesting==0.6.5
```

Exact process commands, shown from the directory containing both repository
checkouts:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1

/usr/bin/time -f 'wall_s=%e,user_s=%U,sys_s=%S,max_rss_kb=%M,cpu_pct=%P' \
  quant-research-framework-rs/target/release/examples/execution_benchmark \
  BTCUSDT_30m_150k.csv events_150k.csv qrf-rust
/usr/bin/time -f 'wall_s=%e,user_s=%U,sys_s=%S,max_rss_kb=%M,cpu_pct=%P' \
  python quant-research-framework/tools/execution_benchmark/run_qrf_python.py \
  --csv BTCUSDT_30m_150k.csv --events events_150k.csv --out qrf-python
/usr/bin/time -f 'wall_s=%e,user_s=%U,sys_s=%S,max_rss_kb=%M,cpu_pct=%P' \
  python quant-research-framework/tools/execution_benchmark/run_backtesting.py \
  --csv BTCUSDT_30m_150k.csv --events events_150k.csv --out backtesting
/usr/bin/time -f 'wall_s=%e,user_s=%U,sys_s=%S,max_rss_kb=%M,cpu_pct=%P' \
  python quant-research-framework/tools/execution_benchmark/run_vectorbt.py \
  --csv BTCUSDT_30m_150k.csv --events events_150k.csv --out vectorbt
```
