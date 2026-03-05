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

## Overrides vs Backtester Config

The backtester (`backtester.py`) contains default configuration values.

The runner can apply overrides (via CLI flags and/or `--overrides-json`) that **force** specific settings during batch runs. This allows you to:

- keep sensible defaults in `backtester.py`
- run controlled experiments without manually editing the backtester each time
- ensure reproducible parameter sweeps across strategies

### Input data

`--csv-file` specifies the OHLC CSV to use for the run. The runner injects this into the backtester at runtime so you don’t need to edit `CSV_FILE` manually.

## Customization

This runner is designed for research sweeps. You can control behavior via:
- CLI flags (recommended for reproducibility)
- `overrides.json` for grouped experiment settings
- backtester defaults in `backtester.py`

Runner-provided overrides take precedence during execution.

## Usage

Run from the repository root:

```bash
python runner/run_strategies.py \
  --backtester-module Backtester_FOREX_poly \
  --csv-file BTCUSDT_30m_3_25.csv \
  --output outputs/btc_30m \
  --max-workers 16 \
  --overrides-json overrides.json \
  --stop-loss-values 1.5,2.0,3.0
