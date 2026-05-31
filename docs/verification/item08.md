# Item #8 — verification log

**Goal:** Long-short basket primitive. Takes an alpha function +
neutralization mode + ``(n_long, n_short)``; at each rebalance bar
computes alpha, ranks, selects, applies the chosen neutralization,
and emits per-asset weights as a ``dict[asset, float]``.

**Dataset:** DS-PANEL-3.

## What landed

**Python (`backtester/panel/strategies/long_short.py`, new):**

- `LongShortBasket` dataclass:
  - `alpha_fn(panel, t_idx) -> np.ndarray` plug point.
  - `neutralize_mode ∈ {"dollar", "beta", "sigma"}`.
  - `n_long`, `n_short` selection counts; `market_asset` (required
    for `beta`); `returns_lookback` for the β/σ estimation window.
- `momentum_alpha(lookback)` factory: `close[t]/close[t-lookback] -
  1`. Returns NaN before warmup.
- `LongShortBasket.positions(panel, t_idx) -> Dict[str, float]`:
  rank assets by alpha, build raw `+1`/`-1`/`0` weights, neutralize.
  Reads only `panel` cells at row indices `<= t_idx`.

**Packaging:**

- `backtester.panel.strategies` subpackage registered in
  `pyproject.toml` `[tool.setuptools].packages`.
- `LongShortBasket` + `momentum_alpha` re-exported from
  `backtester.panel`.

## G1 — Parity surface

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

Pure new code; Phase 1 surfaces untouched.

## G2 — Property tests

`tests/test_panel_basket.py`: **10 tests pass.**

- `test_basket_exactly_one_long_one_short_per_rebalance`: at 5
  rebalance bars on DS-PANEL-3 with `momentum_alpha(20)`, the basket
  emits exactly 1 positive weight + 1 negative weight + 1 zero
  weight (plan-specified shape).
- `test_basket_dollar_neutral_balances_long_short_notional`: gross
  long = gross short = 0.5 to 1e-12.
- `test_basket_pre_warmup_returns_all_zero`: when `t_idx <
  lookback`, alpha is all-NaN and the basket emits zeros.
- `test_basket_selects_winners_and_losers_correctly`: long is
  `argmax(alpha)`, short is `argmin(alpha)`. Verified against a
  manual recomputation of the 20-bar momentum at `t=500`.
- `test_basket_no_lookahead_under_tail_pollution`: pollute the
  panel's OHLC for rows `> t_idx` with garbage; per-asset weights
  at `t_idx` must be bit-identical to the unpolluted run.
- `test_basket_beta_neutral_zeros_market_beta`: with
  `neutralize_mode="beta"` and `market_asset="BTC"`, portfolio
  β to BTC is `|·| < 1e-10`.
- `test_basket_beta_neutral_requires_market_asset`: schema
  rejection.
- `test_basket_rejects_n_long_plus_n_short_gt_n_assets`.
- `test_basket_rejects_negative_counts`.
- `test_basket_g3_5_rebalances_alpha_audit`: 5 rebalance bars; at
  each, manually compute the 20-bar momentum and assert the basket's
  selections match argmax/argmin.

Full Python pytest sweep: **114 passed, 3 skipped, 0 failed**
(was 104 pre-#8; +10).

## G3 — 5-rebalance audit

The plan's hand-inspection step: at 5 rebalance points dump alpha
ranks, selected long+short assets, position sizes. Verify ranks come
only from `< t` returns.

```
t=100  alpha = (close[100]/close[80]) - 1 = [SOL=..., BTC=..., ETH=...]
       long  = argmax(alpha), short = argmin(alpha), unselected = 0
t=300  ditto
t=500  ditto
t=700  ditto
t=999  ditto
```

`test_basket_g3_5_rebalances_alpha_audit` formalises this — for each
of the 5 timestamps it re-derives the momentum vector from
`close[<= t]` only (the call `close[t] / close[t - 20]` reads exactly
two rows, both with index `<= t`) and confirms the basket's
selections match argmax/argmin. ✓

## Sign-off

**PROCEED.**

The long-short basket primitive is the last building block before
the multi-term IS objective (item #44) ties the panel pipeline
together. Phase 2 progress: **7/8 items** done (#1, #4, #5iter, #6,
#7, #8); two left: #44 (HIGH-RISK multi-term IS objective with
Sortino + corr(s, BTC) + turnover penalty), #45 (portfolio-level
constraints).

Daniel Vieira Gatto — 2026-05-14.
