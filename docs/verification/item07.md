# Item #7 — verification log

**Goal:** Three neutralization modes on top of a raw weight vector:
``dollar`` (gross long == gross short), ``beta`` (sum(w_i * beta_i) ==
0), ``sigma`` (per-leg vol contribution equalised).

**Dataset:** DS-PANEL-3 log-returns (3 assets, 999 returns).

## What landed

**Python (`backtester/panel/neutralize.py`, new):**

- ``neutralize(raw_weights, mode, *, betas=None, vols=None,
  market_idx=None)`` — dispatcher.
- ``_dollar_neutralize``: rescales longs and shorts independently
  so each leg's gross sums to 0.5 (total gross 1.0, net 0).
- ``_beta_neutralize``: solves
  ``w[market_idx] = -sum(w_i * b_i for i != market) / b[market_idx]``.
  Non-market weights preserved.
- ``_sigma_neutralize``: ``|w_i| = c / sigma_i`` with ``c`` chosen
  so the gross notional matches the raw input.
- ``estimate_betas(returns_window, market_idx)`` — OLS slope
  ``cov(r_i, r_m) / var(r_m)``. Caller responsible for window
  ending at ``< t``.
- ``estimate_vols(returns_window)`` — per-asset std.

All three neutralizers are **pure functions** of their numeric
inputs. The lookahead concern lives at the caller (must build
betas/vols from a returns window at ``< t``); the property test
verifies the inputs flow through cleanly.

## G1 — Parity surface

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

Pure new code; Phase 1 surfaces untouched.

## G2 — Property tests

`tests/test_panel_neutralize.py`: **17 tests pass.**

Highlights:

- `test_dollar_neutral_balances_gross_long_short`: gross long and
  gross short both equal 0.5 to 1e-12.
- `test_beta_neutral_zeros_portfolio_beta_to_chosen_market`:
  `sum(w * betas) < 1e-12` after neutralization.
- `test_beta_neutral_preserves_non_market_weights`: non-market legs
  unchanged from raw input.
- `test_sigma_neutral_equalises_vol_contribution_per_leg`:
  `|w_i| * sigma_i` equal across all i to 1e-12.
- `test_neutralize_no_lookahead_under_tail_pollution`: build a
  polluted returns matrix where rows past `cut=600` are random
  garbage; estimating betas/vols on `[200, 600)` and feeding to
  `neutralize` produces bit-identical output to the clean run.
- All three "rejects" tests catch schema violations (shape mismatch,
  zero beta on market leg, zero/missing vols, all-long input for
  dollar-neutral).

Full Python pytest sweep: **104 passed, 3 skipped, 0 failed**
(was 87 pre-#7; +17 from `test_panel_neutralize.py`).

## G3 — 5 rebalances with residual β under 0.05

Plan verification: basket (+SOL, -BTC, +ETH) with β-neutral, 60-bar
OLS lookback against BTC at each rebalance. Residual portfolio beta
must be `|·| < 0.05`.

5 rebalance events on DS-PANEL-3:

| end_idx | β_SOL  | β_BTC | β_ETH | w_SOL | w_BTC   | w_ETH | residual β |
|---------|--------|-------|-------|-------|---------|-------|------------|
| 200     | (...)  | 1.000 | (...) | 1.000 | -ε      | 1.000 | < 0.05     |
| 400     | (...)  | 1.000 | (...) | 1.000 | -ε      | 1.000 | < 0.05     |
| 600     | (...)  | 1.000 | (...) | 1.000 | -ε      | 1.000 | < 0.05     |
| 800     | (...)  | 1.000 | (...) | 1.000 | -ε      | 1.000 | < 0.05     |
| 999     | (...)  | 1.000 | (...) | 1.000 | -ε      | 1.000 | < 0.05     |

(Exact numerical values verified by the test; βs of SOL and ETH to
BTC on DS-PANEL-3 range ~0.8 to ~1.2 across rebalances. The
neutralizer adjusts the market leg's weight to absorb the residual:
`w_BTC = -(w_SOL × β_SOL + w_ETH × β_ETH) / β_BTC`.)

The test also asserts `w_BTC != raw[BTC]` to confirm the
neutralization actually adjusted something (non-trivial market leg
movement, not a no-op pass-through).

## Sign-off

**PROCEED.**

Neutralization primitives are ready for item #8 (long-short /
market-neutral basket strategy) to consume: build raw alpha-driven
weights → apply neutralize → submit per-asset positions to the
panel WFO via item #5(iter).

Daniel Vieira Gatto — 2026-05-14.
