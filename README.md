# Quant Research Backtester (Walk-Forward + Robustness)
Research-grade Python framework for evaluating systematic trading strategies using walk-forward optimization, statistical validation, and robustness testing.

A research-oriented Python backtesting framework for systematic strategies with **strict no look-ahead rules**, **walk-forward optimization (WFO)**, **robustness stress tests**, and optional **Monte Carlo diagnostics**.

This project is designed to answer one question:

> Does an apparent edge survive **out-of-sample** evaluation under realistic frictions (fees, slippage, funding) — or is it just fitting the past?

## Quick Start

1. Install dependencies

pip install -r requirements.txt

2. Download OHLC data from Binance

python binance_ohlc_downloader.py --symbol DOGEUSDT --interval 30m --market spot --source api --since 2017-11-01 --until now --out data/DOGEUSDT_30m.csv

3. Configure the backtester

Edit the configuration variables at the top of `backtester.py` and set:

CSV_FILE = "data/DOGEUSDT_30m.csv"

4. Run the backtester

python backtester.py

---

## What’s Included

- **`backtester.py`**  
  Main research backtester: in-sample (IS) vs out-of-sample (OOS), optional rolling walk-forward optimization, realism controls, metrics, plots.

- **`binance_ohlc_downloader.py`**  
  Utility to download and format OHLC candles from Binance into the CSV format expected by the backtester.

- **`indicators_tradingview.py`**  
  Helper indicator functions used by the backtester (EMA/SMA/RSI/ATR/etc., depending on your implementation).

- **`examples/`**  
  Tutorial folder showing how to plug a custom strategy into the pipeline. Includes `atr_cross/atr_cross.py` (ATR-cross with an RSI ≥ 50 confluence) and a [`README.md`](examples/README.md) that walks through the raw-signals contract.

---

## Adding your own strategy

A strategy is a single function returning a `numpy.int8` array of `{-1, 0, +1}` per bar:

```python
def my_strategy(df: pd.DataFrame, lb: int) -> np.ndarray:
    ...  # return +1 (long), -1 (short), or 0 (no signal), no look-ahead.

import backtester as bt
bt.create_raw_signals = my_strategy
bt.main()
```

See [`examples/README.md`](examples/README.md) for the full contract and `examples/atr_cross/atr_cross.py` for a worked example.

---

## Key Features

### Walk-Forward Evaluation (WFO)
- Baseline IS and OOS backtests
- Rolling re-optimization and forward testing per window
- Aggregated WFO performance curve + replication checks

### Realism Controls
- Configurable **fees** and **slippage** on entry/exit
- Optional **funding fees** for crypto at scheduled UTC times
- Optional **stop-loss / take-profit** with intrabar checks (high/low)

### Robustness / Stress Tests
Optional scenarios such as:
- fee shocks
- slippage shocks
- indicator variance
- entry drift
- synthetic “news candle” volatility injections

### Statistical Diagnostics (Optional)
- Monte Carlo / bootstrap-style validation to compare realized metrics vs randomized outcomes

---

## Requirements

- Python 3.10+ recommended
- Common packages: `pandas`, `numpy`, `matplotlib`, `numba`, `pytz`  
  (Exact dependencies depend on your local environment.)

Install:

```bash
pip install pandas numpy matplotlib numba pytz
```

---

## Rust port

A 1-to-1 Rust port of this backtester is available at
[**DaruFinance/quant-research-framework-rs**](https://github.com/DaruFinance/quant-research-framework-rs).
Same strategy logic, same metrics, same WFO/robustness pipeline — runs
~24× faster with ~53× lower peak memory on the same CSV. Its
`examples/atr_cross.rs` produces identical IS/OOS numbers to this repo's
`examples/atr_cross/atr_cross.py` when pointed at the same data.

## License

See [LICENSE](LICENSE) for details.
