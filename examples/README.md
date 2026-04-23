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

## A Rust port exists too

If you want the same pipeline but ~24× faster, see
[`quant-research-framework-rs`](https://github.com/DaruFinance/quant-research-framework-rs) — it
ships the same IS/OOS + WFO + robustness loop as `backtester.py` and has
a parallel `examples/atr_cross.rs` that produces bit-identical IS/OOS
numbers to this example when run on the same CSV.
