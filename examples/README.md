# Examples: adding your own strategy

The whole backtester — IS/OOS split, optimiser, walk-forward, robustness,
Monte Carlo, trade export — lives in `backtester.py`. A **strategy** is the
one function that takes an OHLC DataFrame and returns raw long/short
intents. Everything else is shared.

```python
def create_raw_signals(df: pd.DataFrame, lb: int) -> np.ndarray:
    ...
```

That's the entire contract. Write a function with that signature, assign
it to `backtester.create_raw_signals`, and then call `backtester.main()`.
Every call site inside backtester (classic_single_run, optimiser,
walk_forward, run_robustness_tests) picks it up via the module-level name.

## The contract

Your function must return a 1-D `numpy.int8` array the same length as `df`
where each element is:

| Value | Meaning |
|------:|---------|
| `+1`  | **Long intent** at this bar — open or hold a long |
| `-1`  | **Short intent** at this bar — open or hold a short |
|  `0`  | **No intent** (indicator not warmed up, or no crossover this bar) |

The `lb` argument is the look-back length the optimiser is sweeping over.
The default search range is `range(int(DEFAULT_LB*0.25), int(DEFAULT_LB*1.5)+1)`
centred on `DEFAULT_LB = 50`, so roughly `lb ∈ [12, 76]`.

### No look-ahead

`raw[i]` must only use information available **at or before bar `i-1`**.
The easiest way in pandas is to always `.shift(1)` the series you're
reading from. The backtester uses `raw[i]` to set the desired position
*at bar i* and executes the fill at `df.open[i]`.

### Entries vs levels

Two equivalent ways to express a crossover:

1. **Sign-of-difference** (dense). Set `raw[i] = +1` whenever `fast.shift(1) > slow.shift(1)`
   at bar i, `-1` otherwise. `parse_signals` picks out the flips. This is
   what `backtester.py`'s baseline EMA-crossover does.

2. **Cross-events** (sparse). Set `raw[i] = +1` **only** at the bar of a
   cross-up (`fast_prev > slow_prev & fast_prev_prev <= slow_prev_prev`),
   `-1` only at a cross-down, `0` in between. This is what the proprietary
   `run_strategies.py` spec builder does and what the ATR example here does.

Both produce the same trades — `parse_signals` flip-detects at position
changes — but the sparse form is tidier when you want to stack a
confluence filter on top.

### Adding a confluence

A "confluence" is just a boolean filter you AND with your primary signal
before returning. Any extra indicator — an RSI threshold, a volatility
floor, an MTF agreement test — is a line of code in your strategy
function. See `atr_cross/atr_cross.py` for a worked RSI ≥ 50 example.

## Running

The examples inherit every config knob (fees, slippage, SL/TP, WFO trigger,
robustness scenarios, Monte Carlo, etc.) from `backtester.py`. Edit the
constants at the top of that file to change any of them — no example
changes required.

```bash
# Reference strategy (EMA crossover) — this is backtester.py
python backtester.py

# ATR-cross with RSI confluence — this folder
python examples/atr_cross/atr_cross.py

# Point either one at a different CSV without editing sources
python examples/atr_cross/atr_cross.py path/to/ohlc.csv
BT_CSV=path/to/ohlc.csv python backtester.py
```

The `BT_CSV` env var is the escape hatch that lets strategy scripts
override the default `CSV_FILE` without touching `backtester.py`.

## Writing your own

1. Copy `examples/atr_cross/atr_cross.py` to `examples/my_strategy/my_strategy.py`.
2. Replace the indicator imports and the body of the raw-signals function.
   Keep the signature `(df, lb) -> np.ndarray[int8]`.
3. Reassign `bt.create_raw_signals = my_raw_signals` at module level.
4. `python examples/my_strategy/my_strategy.py`.

If you need an indicator that isn't in `indicators_tradingview.py`, add
it there or inline it in your strategy file — the helpers in that module
are short and easy to extend.

## Plugging in a machine-learning model

The `(df, lb) -> np.ndarray[int8]` contract is intentionally narrow so any
prediction source — sklearn, lightgbm, torch, an ONNX runtime, an external
service — fits as long as you can turn its output into +1 / -1 / 0 per bar.

There are two patterns. They are not mutually exclusive; pick whichever
matches how you already train.

### 1. Pre-computed predictions column ( `examples/ml_precomputed/` )

Train offline, attach a `pred` column to the OHLC frame (or a sidecar CSV
that you join on `time`), and let the strategy function threshold it.

```python
def ml_precomputed_signals(df, lb):
    pred_prev = df['pred'].shift(1).values     # no look-ahead
    raw = np.zeros(len(df), dtype=np.int8)
    raw[pred_prev >= 0.55] =  1
    raw[pred_prev <= 0.45] = -1
    return raw

bt.create_raw_signals = ml_precomputed_signals
```

Use this when you can train ahead of time. It is the fastest path and
keeps the engine completely framework-agnostic.

### 2. Per-bar callback ( `examples/ml_callback/` )

Keep the model in memory and call `predict(features)` inside the loop.
Slower, but the only way to do online / stateful inference, or to mix
training and backtesting in the same process.

```python
def ml_callback_signals(df, lb):
    raw = np.zeros(len(df), dtype=np.int8)
    for i in range(len(df)):
        feats = extract_window_features(df, i, lb)   # uses df[: i] only
        if feats.size == 0:
            continue
        score = MODEL.predict(feats)
        if   score >= 0.55: raw[i] =  1
        elif score <= 0.45: raw[i] = -1
    return raw
```

The look-ahead rule is the same as for any other strategy: features for
bar `i` may only use data from bars `i-1` and earlier. The example
ships a tiny hand-coded linear model so it runs without sklearn / torch
installed; swap it for any object exposing `.predict(features) -> float`.

## Plugging in a custom regime detector

Regime segmentation has two pluggable seams:

```python
bt.REGIME_LABELS  = ['Calm', 'Volatile']            # 2..5 labels
bt.detect_regimes = my_detector                     # (df) -> pd.Series[label]
```

`my_detector` returns one label per bar drawn from `REGIME_LABELS`. The
optimiser then runs one look-back search per label, the WFO loop rotates
the active LB bar-by-bar in OOS, and `evaluate_filters` /
`backtest_continuous_regime` work transparently with whatever set you
provide.

Constraints:

* `len(REGIME_LABELS)` must be in `{2, 3, 4, 5}`.
* `detect_regimes(df)` must be free of look-ahead — only use information
  available at bar `i-1` or earlier when labelling bar `i`.
* The series your detector returns must be indexed the same as `df`
  (default `RangeIndex` from `load_ohlc`).

`examples/regime_custom/regime_custom.py` shows three demos:

1. **2-regime volatility detector** — Calm vs Volatile by 50-bar realised
   vol vs its 250-bar median.
2. **4-regime trend × volatility** — CalmUp / CalmDown / VolUp / VolDown.
3. **5-regime ML-style detector** — quantile bucketing of (return, vol),
   structured to be drop-in-replaceable with a fitted scikit-learn
   `KMeans` / `GaussianMixture` / HMM.

Pick one by editing the `DEMO` constant at the top of the file. The same
pattern works for any ML detector: load weights once at import time, do
inference inside `detect_regimes(df)`, return labels.

## A Rust port exists too

If you want the same pipeline but ~24× faster, see
[`quant-research-framework-rs`](https://github.com/DaruFinance/quant-research-framework-rs) — it
ships the same IS/OOS + WFO + robustness loop as `backtester.py` and has
a parallel `examples/atr_cross.rs` that produces bit-identical IS/OOS
numbers to this example when run on the same CSV.
