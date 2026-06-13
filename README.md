# Quant Research Backtester (Walk-Forward + Robustness)

[![parity](https://github.com/DaruFinance/quant-research-framework/actions/workflows/parity.yml/badge.svg)](https://github.com/DaruFinance/quant-research-framework/actions/workflows/parity.yml)
[![docs](https://github.com/DaruFinance/quant-research-framework/actions/workflows/docs.yml/badge.svg)](https://github.com/DaruFinance/quant-research-framework/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/quant-research-framework.svg)](https://pypi.org/project/quant-research-framework/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19798594.svg)](https://doi.org/10.5281/zenodo.19798594)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DaruFinance/quant-research-framework/main?filepath=examples%2Fnotebook%2Fwalkthrough.ipynb)

**The readable reference half of a dual-engine walk-forward backtester.** Strict no-look-ahead, walk-forward optimization (WFO), robustness stress tests, realism controls (fees, slippage, funding, SL/TP), Monte Carlo diagnostics, and overfitting statistics (DSR/PSR/MinTRL/MinBTL, PBO/CSCV). A [Rust port](https://github.com/DaruFinance/quant-research-framework-rs) reproduces this engine's metrics to within `1e-3`, checked in CI on every push.

It answers one question: *does an apparent edge survive out-of-sample evaluation under realistic frictions, or is it just fitting the past?*

## Two engines, one spec

This repository is the **Python reference** — the readable specification. A separate **Rust port** re-implements it for speed (23.8–57× faster, 33–65× less memory) and a parity oracle runs both on identical input, asserting the metrics agree within `1e-3`. If the port drifts from this reference, CI goes red — the correctness claim is *enforced, not asserted*.

```
                ┌────────────────────────────────────────────────┐
                │                same OHLC input                  │
                │     data/SOLUSDT_1h.csv · EURUSD_1h.csv · …      │
                └───────────────┬────────────────┬───────────────┘
                                │                │
                ┌───────────────▼──────┐  ┌──────▼───────────────┐
                │  Python reference    │  │      Rust port       │
                │  backtester/         │  │  (sibling repo, …-rs)│
                │  (this repo, the spec)│ │   speed: 23.8–57×    │
                └───────────────┬──────┘  └──────┬───────────────┘
                                │ metrics        │ metrics
                                ▼                ▼
                ┌────────────────────────────────────────────────┐
                │              parity oracle  (CI)                │
                │      tools/parity_*.py  ·  assert |Δ| ≤ 1e-3    │
                │   default 56/56 · regime+WFO 98/98 · fx 56/56   │
                └────────────────────────┬───────────────────────┘
                                         │
                                 red if the port drifts
```

## Reproduce it in under 5 minutes

```bash
pip install -r requirements.txt
make repro
```

`make repro` runs the no-look-ahead property suite (Hypothesis, over a generated input space) and the look-ahead leak demo (`listings/lah_demo.py`, the paper's), which replaces every bar after #400 with noise and checks the signals before #400 are unchanged. The causal strategy is untouched; a leaky one peeking 5 bars ahead moves exactly the 4 bars that reach across the boundary:

```
[PASS] good (paper Listing 1, .take(idx-1) shift): 0 of 400 bars affected by post-bar-400 pollution.
[FAIL] buggy (deliberate close.shift(-5) peek): 4 of 400 bars affected by post-bar-400 pollution; first leak at bar 395, last at bar 398.
```

The cross-engine parity surfaces (Python vs Rust — 56/56 · 98/98 · 56/56 metric points at `1e-3`) are driven from the [Rust port repo](https://github.com/DaruFinance/quant-research-framework-rs); run `make parity` there.

## What this is / what it isn't

**It is** a correctness-first WFO backtesting engine with the Python↔Rust equivalence machine-checked: realism controls on by default (fees, slippage, funding, SL/TP with intrabar high/low checks), strict no-look-ahead enforced by ledger-level invariant tests, and overfitting diagnostics (DSR/PSR/MinTRL/MinBTL, PBO/CSCV, deflated multiple-testing haircuts).

**It isn't:**
- **Not alpha.** The bundled strategies (EMA-cross, ATR-cross, …) are plumbing to exercise the engine, not trade signals. There is no edge here to deploy.
- **Not live trading.** No broker connectivity, order management, or execution — it evaluates strategies on historical bars.
- **Tested on crypto, FX, and synthetic GBM only.** SOL/BTC/DOGE-USDT, EUR/USD, USD/JPY, and a GBM generator. Equities, futures, and options are untried.
- **Parity-gated on the core surfaces only.** The stationary-bootstrap module (`backtester/bootstrap.py`) and the `examples/ml_*` strategies are Python-only — no Rust counterpart, no cross-engine check.

## Quick Start

```bash
pip install -r requirements.txt

# Zero-setup smoke test: generate a synthetic OHLC CSV and run.
python gen_synthetic.py
BT_CSV=data/SYNTHETIC.csv python -m backtester
```

For real market data, swap the generator for a download:

```bash
python binance_ohlc_downloader.py --symbol DOGEUSDT --interval 30m --market spot \
    --source api --since 2017-11-01 --until now --out data/DOGEUSDT_30m.csv
BT_CSV=data/DOGEUSDT_30m.csv python -m backtester
```

`BT_CSV` overrides the `CSV_FILE` constant in `backtester/__init__.py` without touching the source. If you prefer, edit the constant at the top of `backtester/__init__.py` instead.

Note: the framework was repackaged from a single `backtester.py` script into a `backtester/` package in v0.3.0; the `python -m backtester` form replaces the legacy `python backtester.py` invocation.

---

## What's Included

- **`backtester/`** — the engine package, restructured in v0.3.0 from the
  legacy single-file `backtester.py` script. Sub-modules:
  - `backtester/__init__.py` — the engine: IS / OOS / WFO + robustness
    overlays + Monte Carlo + trade ledger export.
  - `backtester/__main__.py` — `python -m backtester` CLI entry point.
  - `backtester/indicators.py` — TradingView-style indicator helpers
    (EMA, SMA, RSI, ATR, MACD, Stochastic).
  - `backtester/dsr.py` — Bailey & López de Prado (2014) Deflated Sharpe
    Ratio utility.

- **`binance_ohlc_downloader.py`** — download and format Binance OHLC
  candles into the CSV format the engine reads.

- **`gen_synthetic.py`** — GBM-based synthetic OHLC generator. No
  network required; writes `data/SYNTHETIC.csv` in the same format the
  Binance downloader emits.

- **`indicators_tradingview.py`** — backwards-compatibility shim that
  re-exports `backtester.indicators`. Pre-v0.3.0 user scripts that do
  `from indicators_tradingview import compute_atr` keep working unchanged.

- **`examples/`** — strategy and ML examples + parallel runner + the
  walkthrough notebook:
  - [`examples/atr_cross/`](examples/atr_cross) — ATR-cross with an
    RSI ≥ 50 confluence (worked-example strategy).
  - [`examples/regime_custom/`](examples/regime_custom) — three
    pluggable regime detectors (vol2, vol4, ml5).
  - [`examples/ml_precomputed/`](examples/ml_precomputed) and
    [`examples/ml_callback/`](examples/ml_callback) — two ML-strategy
    integration patterns against the `(df, lb) -> int8[]` contract.
  - [`examples/ml_sklearn/`](examples/ml_sklearn) — scikit-learn
    classifier wired into the strategy contract.
  - [`examples/ml_regime_kmeans/`](examples/ml_regime_kmeans) —
    KMeans-based regime detector matching the `detect_regimes()` API.
  - [`examples/batch_runner/`](examples/batch_runner) — multiprocess
    runner for sweeping a parameter grid in parallel.
  - [`examples/notebook/walkthrough.ipynb`](examples/notebook/walkthrough.ipynb)
    — the 10-cell tour referenced by the Binder badge above.
  - [`examples/README.md`](examples/README.md) — the `(df, lb)` raw-signal
    contract spelled out.

- **`docs/`** — Sphinx + autodoc API reference (Furo theme), built and
  published to GitHub Pages by `.github/workflows/docs.yml`.

- **`tests/`** — pytest suite (32 tests, including Hypothesis property
  tests on `parse_signals` and `walk_forward_regime` invariants).

- **`binder/`** — Binder configuration (`requirements.txt`,
  `runtime.txt`) for the launchable notebook.

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
`version` field in [`pyproject.toml`](pyproject.toml) is the source of
truth (currently `0.6.0`).

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
within `1e-3` relative tolerance against this Python reference on three
independent surfaces:

- **Default config (56/56 metric points)** — verified by
  [`tools/parity_check.py`](https://github.com/DaruFinance/quant-research-framework-rs/blob/main/tools/parity_check.py).
- **Regime + WFO (98/98 metric points)** — verified by
  [`tools/parity_regime.py`](https://github.com/DaruFinance/quant-research-framework-rs/blob/main/tools/parity_regime.py)
  on the Rust port's v0.3.2 release.
- **Forex mode (56/56 metric points on EURUSD 1h)** — verified by
  [`tools/parity_forex.py`](https://github.com/DaruFinance/quant-research-framework-rs/blob/main/tools/parity_forex.py).

These three are the original parity surfaces; v0.6.0 adds more (volume, shared
indicators, IS-surface, overfitting statistics). Maximum observed relative
deviation on the default surfaces is below `5e-5` (the metric ledger's `%.4f`
print precision floor), 20× tighter than the declared `1e-3` tolerance. We
avoid the term *byte-identical* throughout: parity is tolerance-bounded by
construction, not bit-equality, and is enforced continuously by the
`parity_*.py` suite in CI.

It runs **23.8–57× faster** (Python reference vs Rust port) and uses **33–65× less memory**:

| Bars   | Python warm (s) | Rust (s) | Speed-up | Python RSS (MB) | Rust RSS (MB) |
|-------:|----------------:|---------:|---------:|----------------:|--------------:|
|  5,000 |    2.32 ± 0.06  |    0.01  |  232×†   |             270 |           2.8 |
| 15,000 |    2.85 ± 0.05  |    0.05  |  57.0×   |             273 |           4.2 |
| 30,000 |    3.98 ± 0.09  |    0.12  |  33.2×   |             280 |           6.2 |
| 48,000 |    5.71 ± 0.10  |    0.24  |  23.8×   |             294 |           8.8 |

(Median warm wall-clock over n=5 runs after one untimed warm-up, peak RSS as
the max observed, on the bundled `SOLUSDT_1h.csv`. †The 5,000-bar 232× is a
measurement-floor artifact — Rust there sits at the timer resolution — so the
steady-state figure is the 48k row, 23.8×. Same harness and numbers as the
paper; reproduce with `python tools/bench_paper.py --runs 5` from the sibling
Rust repo.)

## Comparison vs other open-source backtesters

What this framework emphasises that mainstream open-source alternatives do
not (verified against primary docs as of 2026-04):

| Framework              | License                  | Built-in WFO | Per-regime LB optimisation | Strict-LAH property tests | Cross-language byte-parity tests |
|------------------------|--------------------------|:------------:|:--------------------------:|:-------------------------:|:--------------------------------:|
| **this** (Python + Rust) | Apache-2.0                    | ✓            | ✓                          | ✓                         | ✓                                |
| [vectorbt][vbt]        | Apache-2.0 + Commons     | ✓ (Splitter) | ✗                          | ✗                         | n/a                              |
| [backtrader][bt]       | GPL-3.0                  | ✗ (community) | ✗                         | ✗                         | n/a                              |
| [NautilusTrader][nt]   | LGPL-3.0                 | ✗ (engine only) | ✗                       | ✗                         | ✗ (bilingual; no parity asserts) |
| [zipline-reloaded][zl] | Apache-2.0               | ✗ (3rd-party) | ✗                         | ✗                         | n/a                              |
| [QuantConnect Lean][lean] | Apache-2.0            | ✓            | ✗                          | ✗                         | n/a                              |
| [bt][btp]              | MIT                      | ✗            | ✗                          | ✗                         | n/a                              |

The **combination** is the contribution: WFO + per-regime LB + strict
no-look-ahead enforced by ledger-level invariant tests + a Python
reference and Rust port whose metric outputs agree within $10^{-3}$
relative tolerance on every validated surface, enforced continuously by the
`parity_*.py` harness suite in CI. Each cell individually exists somewhere; no
other framework ships the whole bundle.

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
