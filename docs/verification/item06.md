# Item #6 — verification log

**Goal:** Equal-Risk-Contribution sizer. Given a return window or
covariance matrix, return weights `w` such that each asset
contributes equally to portfolio variance: `w_i * (Σw)_i == V/n` for
all `i`, where `V = w'Σw`.

**Dataset:** DS-PANEL-3 log-returns (3 assets, 999 returns from
1000-bar OHLC).

## What landed

**Python (`backtester/panel/sizing.py`, new):**

- `equal_weights(n)`: trivial 1/N baseline.
- `erc_weights(returns_window=None, *, cov=None, max_iter=1000,
  tol=1e-10)`: ERC weights from either a returns matrix (bars ×
  assets) or a precomputed covariance. **Rescales the covariance to
  unit trace** before solving so SLSQP convergence is consistent
  regardless of the absolute return magnitude (raw hourly-return
  covariances ~1e-6 are ill-conditioned for SLSQP's default ftol; the
  rescale fixes it). Includes an iterative fixed-point fallback for
  pathological inputs where SLSQP reports failure.
- `risk_contributions(weights, cov)`: convenience accessor for the
  per-asset `w_i * (Σw)_i` vector — used by tests / verification logs
  to check the ERC invariant without re-implementing the math.
- `_cov_from_returns(returns_window)`: NaN-tolerant sample
  covariance helper.

**pyproject.toml:** `panel = ["xarray>=2024.0", "pyarrow>=14.0",
"scipy>=1.10"]` — adds scipy as a panel-extras dep.

**Rust port deferred** to the per-asset Rust panel WFO integration
later in Phase 2 (along with #7 and #8). The Python side does all
the heavy weight-construction work; the Rust simulator only needs the
weights vector as input, which can be passed in directly when the
basket primitive in #8 wires per-asset sizing into the kernel.

## G1 — Parity surface

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

The sizing module is pure new code with no kernel interaction; Phase
1 surfaces untouched.

## G2 — Property tests

`tests/test_panel_sizing.py`: **11 tests pass**.

- `test_equal_weights_sum_to_one` / `test_equal_weights_rejects_zero_n`.
- `test_erc_weights_sum_to_one_on_panel_returns`: weights are
  non-negative, sum to 1 to 1e-6.
- `test_erc_equal_risk_contribution_property`: relative dispersion of
  per-asset RCs around their target `V/n` < 1%.
- `test_erc_weights_bounded_by_plan_expectation`: plan-specified
  `[0.1, 0.6]` bound holds for every asset on the DS-PANEL-3 100-bar
  window.
- `test_erc_weights_rolling_rebalance_no_lookahead`: build a polluted
  parallel returns matrix where rows after `cut_idx` are random
  garbage; ERC weights computed on the same `[500, cut_idx)` window
  on both must be bit-identical. **Pure-function lookahead-freeness.**
- `test_erc_weights_consistent_across_calls`: determinism.
- `test_erc_weights_n1_is_trivial`: single-asset case returns `[1.0]`.
- `test_erc_weights_rejects_non_square_cov` and `_insufficient_clean_rows`.
- `test_erc_weekly_rebalance_on_ds_panel_3_50_windows`: roll a 100-
  bar window across DS-PANEL-3 at 168-bar (weekly) cadence; every
  rebalance must satisfy weight-sum + RC-equality.

Full Python pytest sweep: **87 passed, 3 skipped, 0 failed**
(was 76 pre-#6; +11 from `test_panel_sizing.py`).

## G3 — Hand-inspected rebalance events

DS-PANEL-3, 100-bar log-return windows, weekly rebalance cadence
(168 1h bars between rebalances). 5 representative events:

| end_idx | weights (SOL, BTC, ETH)            | port_var   | RC dispersion / target |
|---------|------------------------------------|-----------|------------------------|
| 100     | (0.30, 0.41, 0.29)                 | 2.79e-05  | < 1%                   |
| 268     | (0.31, 0.39, 0.30)                 | 3.21e-05  | < 1%                   |
| 436     | (0.29, 0.40, 0.31)                 | 4.05e-05  | < 1%                   |
| 604     | (0.32, 0.38, 0.30)                 | 3.78e-05  | < 1%                   |
| 772     | (0.30, 0.41, 0.29)                 | 3.55e-05  | < 1%                   |

Each rebalance: weights sum to 1.000000 ± 1e-7; every asset weight
lies in `[0.29, 0.41]` (well inside the plan's `[0.1, 0.6]` guard);
RC dispersion `std(w_i × (Σw)_i − V/n)` is < 1% of the per-asset
target across all rebalances.

Lookahead audit: each rebalance reads exactly the `returns_window`
of shape `(100, 3)` ending at `end_idx`. `_cov_from_returns` operates
purely on its input; `erc_weights` operates on the resulting
covariance. No global state, no series-indexing, no time-dependent
constants. Pure function from a finite input to a fixed-shape output.

## Sign-off

**PROCEED.**

The ERC sizer is the first portfolio-level primitive on top of the
panel substrate. Item #8 (long-short / market-neutral basket) wires
sizing into the basket weight vector; item #44 (multi-term IS
objective) consumes per-asset weights when scoring panel WFO
windows.

Daniel Vieira Gatto — 2026-05-14.
