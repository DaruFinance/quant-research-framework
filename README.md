# Quant Research Backtester (Walk-Forward + Robustness)
Research-grade Python framework for evaluating systematic trading strategies using walk-forward optimization, statistical validation, and robustness testing.

A research-oriented Python backtesting framework for systematic strategies with **strict no look-ahead rules**, **walk-forward optimization (WFO)**, **robustness stress tests**, and optional **Monte Carlo diagnostics**.

This project is designed to answer one question:

> Does an apparent edge survive **out-of-sample** evaluation under realistic frictions (fees, slippage, funding) — or is it just fitting the past?

## Quick Start

```bash
pip install -r requirements.txt

# Zero-setup smoke test: generate a synthetic OHLC CSV and run.
python gen_synthetic.py
BT_CSV=data/SYNTHETIC.csv python backtester.py
```

For real market data, swap the generator for a download:

```bash
python binance_ohlc_downloader.py --symbol DOGEUSDT --interval 30m --market spot \
    --source api --since 2017-11-01 --until now --out data/DOGEUSDT_30m.csv
BT_CSV=data/DOGEUSDT_30m.csv python backtester.py
```

`BT_CSV` overrides the `CSV_FILE` constant in `backtester.py` without touching the source. If you prefer, edit the constant at the top of `backtester.py` instead.

---

## What’s Included

- **`backtester.py`**  
  Main research backtester: in-sample (IS) vs out-of-sample (OOS), optional rolling walk-forward optimization, realism controls, metrics, plots.

- **`binance_ohlc_downloader.py`**  
  Utility to download and format OHLC candles from Binance into the CSV format expected by the backtester.

- **`gen_synthetic.py`**  
  GBM-based synthetic OHLC generator. No network required — use it for smoke-testing or reproducible demos. Writes `data/SYNTHETIC.csv` in the same format the Binance downloader emits.

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
- Rolling re-optimization and forward testing per window (`USE_WFO`,
  `WFO_TRIGGER_MODE` ∈ {`candles`, `trades`}, `WFO_TRIGGER_VAL`)
- Aggregated WFO performance curve + replication checks
- **WFO + regime segmentation**: WFO walks its standard cadence; the
  per-regime LB just rotates inside each window — fixed in v0.2.0 (was
  previously re-anchoring the IS window on every regime change).

### Realism Controls
- Configurable **fees** and **slippage** on entry/exit
- Optional **funding fees** for crypto at scheduled UTC times
- Optional **stop-loss / take-profit** with intrabar checks (high/low)
- **Forex mode** (`FOREX_MODE`): pip-based risk units (auto-detects JPY
  pip size), no funding fees, R-denominated PnL. Off by default.
- **Session mode** (`TRADE_SESSIONS`): restricts trading to a NY-time
  window (`SESSION_START`, `SESSION_END`); positions are force-closed on
  the last in-session bar of each day.
- **Second OOS split** (`USE_OOS2`): doubles the OOS window so the
  framework can score the strategy on two contiguous OOS blocks
  (OOS1 + OOS2) for an extra layer of out-of-sample evidence.

### Regime segmentation
- 3-regime EMA-based detector by default (Uptrend / Downtrend / Ranging)
- **Pluggable**: override `REGIME_LABELS` (length 2..5) and
  `detect_regimes(df) -> pd.Series` to plug in any analytic or ML-based
  detector. See [`examples/regime_custom/`](examples/regime_custom) for
  three demos (vol2, vol4, ml5).
- Per-regime LB optimisation, OOS LB rotation, optional regime/direction
  filters (`FILTER_REGIMES`, `FILTER_DIRECTIONS`).

### ML-driven strategies
- The signal contract `(df, lb) -> np.ndarray[int8]` is unchanged, so
  any model that can produce per-bar long/short scores plugs in.
- Two patterns shipped:
  - [`examples/ml_precomputed/`](examples/ml_precomputed) — train
    offline, attach a `pred` column, threshold inside the strategy fn.
  - [`examples/ml_callback/`](examples/ml_callback) — keep a model in
    memory and call `predict(features)` per bar (online / stateful).

### Robustness / Stress Tests
Optional scenarios such as:
- fee shocks
- slippage shocks
- indicator variance
- entry drift
- synthetic “news candle” volatility injections

### Statistical Diagnostics (Optional)
- Monte Carlo / bootstrap-style validation to compare realized metrics vs randomized outcomes

### Versioning

The framework follows [Semantic Versioning](https://semver.org/). See
[`CHANGELOG.md`](CHANGELOG.md) for what changed in each release; the
`__version__` constant in `backtester.py` is the source of truth.

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
Same strategy logic, same metrics, same WFO/robustness pipeline. The
Rust port reproduces every IS/OOS/baseline/optimised/WFO metric line
byte-for-byte against this Python reference on two independent surfaces:

- **Default config (56/56 metric points byte-identical at 0.001 tol)** — verified by
  [`tools/parity_check.py`](https://github.com/DaruFinance/quant-research-framework-rs/blob/main/tools/parity_check.py).
- **Regime + WFO (14/14 metric tags byte-identical at 0.001 tol)** — verified
  by [`tools/parity_regime.py`](https://github.com/DaruFinance/quant-research-framework-rs/blob/main/tools/parity_regime.py)
  as of the Rust port's v0.3.0 release.

It runs **25–60× faster** and uses **~37× less memory**:

| Bars   | Python (s) | Rust (s) | Speed-up | Python RSS (MB) | Rust RSS (MB) |
|-------:|-----------:|---------:|---------:|----------------:|--------------:|
| 15,000 |       3.03 |     0.05 |   60.60× |             273 |             4 |
| 25,000 |       3.75 |     0.10 |   37.50× |             277 |             6 |
| 35,000 |       4.60 |     0.14 |   32.86× |             282 |             7 |
| 48,000 |       5.78 |     0.23 |   25.13× |             294 |             8 |

(Min wall-clock and max peak RSS over 3 warm runs on the bundled
`SOLUSDT_1h.csv`. Reproduce with `python tools/bench.py` from the
sibling repo.)

## Comparison vs other open-source backtesters

What this framework emphasises that mainstream open-source alternatives do
not (verified against primary docs as of 2026-04):

| Framework              | License                  | Built-in WFO | Per-regime LB optimisation | Strict-LAH property tests | Cross-language byte-parity tests |
|------------------------|--------------------------|:------------:|:--------------------------:|:-------------------------:|:--------------------------------:|
| **this** (Python + Rust) | MIT                    | ✓            | ✓                          | ✓                         | ✓                                |
| [vectorbt][vbt]        | Apache-2.0 + Commons     | ✓ (Splitter) | ✗                          | ✗                         | n/a                              |
| [backtrader][bt]       | GPL-3.0                  | ✗ (community) | ✗                         | ✗                         | n/a                              |
| [NautilusTrader][nt]   | LGPL-3.0                 | ✗ (engine only) | ✗                       | ✗                         | ✗ (bilingual; no parity asserts) |
| [zipline-reloaded][zl] | Apache-2.0               | ✗ (3rd-party) | ✗                         | ✗                         | n/a                              |
| [QuantConnect Lean][lean] | Apache-2.0            | ✓            | ✗                          | ✗                         | n/a                              |
| [bt][btp]              | MIT                      | ✗            | ✗                          | ✗                         | n/a                              |

The **combination** is the contribution: WFO + per-regime LB + strict
no-look-ahead enforced by property tests + a Python reference and Rust
port whose outputs agree byte-for-byte on the deterministic pipeline.
Each cell individually exists somewhere; no other framework ships the
whole bundle.

[vbt]: https://github.com/polakowo/vectorbt
[bt]:  https://github.com/mementum/backtrader
[nt]:  https://github.com/nautechsystems/nautilus_trader
[zl]:  https://github.com/stefan-jansen/zipline-reloaded
[lean]: https://github.com/QuantConnect/Lean
[btp]: https://github.com/pmorissette/bt

## Citation

If you use this framework in academic or research work, please cite via
[`CITATION.cff`](CITATION.cff). The Rust port has its own
[`CITATION.cff`](https://github.com/DaruFinance/quant-research-framework-rs/blob/main/CITATION.cff)
and citing either implies the other (sibling cross-reference).

## License

See [LICENSE](LICENSE) for details.
