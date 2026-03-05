# Batch Strategy Runner

Runs large batches of strategy specifications through the backtesting framework using multiprocessing.

## Features
- Resumable runs (skips completed strategies)
- Spawn-based multiprocessing
- Memory-aware autoscaling (reduces workers after MemoryError)
- Periodic worker recycling (`maxtasksperchild`) to avoid RAM creep
- Organized outputs per strategy (logs, figures, exports)

## How it works
1) Loads OHLC data
2) Builds indicator + transformation specs
3) For each spec, wires a spec-specific signal generator into the backtester
4) Runs the full pipeline (IS/OOS, optional WFO, robustness, plots, exports)
5) Writes output folders and completion markers

## Usage
Edit configuration at the top of `run_strategies.py` (symbols, intervals, output folder, worker count), then run:

```bash
python runner/run_strategies.py