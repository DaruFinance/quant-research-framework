# Bar-permutation Monte Carlo

Completed-trade `permutation` shuffles without replacement; `resampling` samples with replacement. Both default to 1,000 runs and accept a custom count. Bar permutation defaults to 500 runs because every iteration rebuilds OHLCV and reruns the frozen strategy.

Each invocation tests a frozen strategy specification against reconstructed OHLCV
paths. Each iteration independently shuffles close log returns and complete
intrabar templates without replacement, keeps timestamps fixed, moves volume
with its template, regenerates strategy signals and runs the full backtest
without re-optimizing parameters.

Bar permutation is expensive because every iteration executes another full
backtest. The default is 500 iterations. Start with one full-history run to
measure local cost, then choose a worker count from 1 to 4:

```bash
python mc_bar_permutation/run.py \
  --mode bar-permutation \
  --input data/SOLUSDT_1h.csv \
  --output results/mc-bars \
  --runs 500 \
  --workers 1
```

Completed-trade Monte Carlo uses the same dispatcher:
`python mc_bar_permutation/run.py --mode resampling --returns returns.json --output results/mc-trades`.
Use `--mode permutation` as the alternative. Trade modes default to 1,000 runs.

For the built-in EMA strategy, the script builds a repository-local Rust worker.
Bundled Rust engine source provides the frozen backtest without a separate checkout. Build files stay under the output directory unless
`--barperm-bin` is provided.

Each run writes a native binary trade ledger, metrics and a status record.
`manifest.json` records source and parameter hashes, requested and completed
counts, seeds, failures and output hashes.

Completed runs with the same source, specification and seed are reused on
restart. A failed run makes the command fail. The runner does not report a
p-value from an incomplete queue.

`spec.example.json` freezes an EMA-crossover lookback, disables SL/TP and keeps
fees, slippage and funding enabled. Set `strategy.kind` to `python-callable` and add a `package.module:function` value to use custom code. The callable receives
`(dataframe, lookback, parameters)` and must return one causal raw position
signal per bar using `-1`, `0` or `1`; zero means hold the current target.

This explicit fallback reruns the Python engine and writes a compressed NumPy
ledger. Unsupported strategies and malformed signals fail before a result is
recorded.

The logarithmic reconstruction requires positive finite OHLC. Negative or zero
prices, including negative-price WTI histories, are rejected rather than
dropped or converted to missing values. This mode is a no-replacement bar
permutation, not the stationary bar-return bootstrap used in the later gold
Monte Carlo paper.
