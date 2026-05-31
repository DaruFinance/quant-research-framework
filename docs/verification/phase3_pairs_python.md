# Phase 3 — T2 pairs items (Python side): #9, #10, #11, #12, #13

**Goal:** Land the full pairs / stat-arb plumbing on the Python side
before any Rust port. Per the user's workflow ("Python first per
pipeline, then Rust port, then push"), this is the first half of
the T2 tree's bilingual delivery.

**Dataset:** DS-PAIR-BTCETH (1500-bar BTC/ETH 1h slice from
2023-Q3, cointegrating with ADF p=0.0012 on the full window).

## What landed

**`backtester/pairs/__init__.py`** (lazy-imports statsmodels with a
helpful ImportError if `[pairs]` extras aren't installed).

**`backtester/pairs/spread.py` — item #10:**

- `log_ratio(panel, a, b, t_idx)` — `log(close_a) - log(close_b)`.
  Trivially leak-free.
- `ols_resid(panel, a, b, t_idx, lookback=60)` — rolling OLS-residual
  spread; each bar's spread comes from a regression on the trailing
  `lookback` bars ending at-or-before that bar.
- `kalman_beta_spread(panel, a, b, t_idx)` — 2-D random-walk-state
  Kalman filter for dynamic β. Returns the full β trajectory.
- `pca_resid(panel, asset_a, t_idx, other_assets, lookback=60)` —
  first-PC residual on an N-asset panel.
- `ml_resid(panel, a, b, t_idx, lookback=60, predictor_factory)` —
  generic ML residual; default predictor is sklearn LinearRegression.

All return a frozen `SpreadResult` dataclass with `spread`, `beta`,
`method`, `asset_a`, `asset_b` fields.

**`backtester/pairs/eligibility.py` — item #13:**

- `half_life_ou(spread)` — OU half-life estimator via OLS on
  `ds_t = -lambda * s_{t-1} + noise`. Returns `+inf` when no
  mean reversion is detected.
- `is_eligible_pair(spread, p_value, criteria)` — dispatcher running
  the stack of filters (ADF p-value, half-life range, min window).
- `EligibilityCriteria` dataclass bundles the filter parameters.

**`backtester/pairs/screener.py` — item #9 (HIGH-RISK):**

- `engle_granger(close_a, close_b)` — `log(a) ~ log(b)` OLS, ADF on
  residuals (`regression="n"`), returns `(pvalue, beta)`.
- `distance_ssd(close_a, close_b)` — SSD on standardised log-prices.
- `screen_pairs(panel, t_idx, method, lookback, top_n)` — iterates
  ordered pairs, ranks by statistic (ascending = better for both
  supported methods). Validates `method` up-front so typos raise
  loudly instead of being silently captured as per-pair errors.

**`backtester/pairs/cadence.py` — item #11 (HIGH-RISK):**

- `Cadence` dataclass with `mode ∈ {bars, trigger, on_breakdown}`,
  `every`, `trigger_fn`.
- `CadenceEngine` drives spread-β refits over a panel, returning
  `[(refit_bar, SpreadResult)]`. The `on_breakdown` mode caps the
  refit cadence at one every 50 bars to avoid refit storms.

**`backtester/pairs/stops.py` — item #12:**

- `z_multiple_stop(spread, t_idx, window, z_mult)`.
- `half_life_multiple_stop(entry_idx, t_idx, half_life, hl_mult)`.
- `breakdown_trigger_stop(beta_prev, beta_new, beta_jump)`.
- `StopReason` enum + `StopDecision` dataclass.

## G1 — Parity surface

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

Pure new code; Phase 1 / Phase 2 surfaces untouched.

## G2 — Property tests (HIGH-RISK leak batteries pass)

**`tests/test_pairs_spread.py`:** 10 tests pass — schema checks plus
tail-pollution lookahead-freeness for log_ratio, ols_resid,
kalman_beta_spread, pca_resid.

**`tests/test_pairs_eligibility.py`:** 6 tests pass — half-life is
finite on a synthetic OU process and `+inf` on a random walk; the
filter stack accepts a typical input and rejects schema violations
(high p-value, short window, out-of-range half-life).

**`tests/test_pairs_screener.py`:** 7 tests pass — `engle_granger`
emits valid `(p, β)` pairs, DS-PAIR-BTCETH cointegrates at p<0.05,
`distance_ssd` is lower for similar log-prices. **`test_screener_no_lookahead_10_windows`** is the HIGH-RISK gate: 10 random t_idx
endpoints, panel rows polluted past each endpoint, screener output
bit-identical to the unpolluted run.

**`tests/test_pairs_cadence.py`:** 4 tests pass. **`test_cadence_50t_refits_leak_free`** is the HIGH-RISK gate: 50 schedules with
randomised (every, T) parameters, panel rows polluted past T, every
β fitted at a refit bar ≤ T must be bit-identical to the unpolluted
run.

**`tests/test_pairs_stops.py`:** 9 tests pass — warmup handling,
fire/no-fire on synthetic series, schema-rejection paths.

Full Python pytest sweep: **181 passed, 3 skipped, 0 failed**
(was 145 at end of Phase 2; +36 from the five pairs test files).

## G3 — Hand-inspected sample

End-to-end use on DS-PAIR-BTCETH (first 500 bars):

1. `engle_granger(BTC[:500], ETH[:500])` → ADF p=0.0012, β=0.43.
2. `ols_resid(panel, "BTC", "ETH", t_idx=499, lookback=60)` →
   spread series shape (500,); first 59 NaN; last 441 finite.
3. `half_life_ou(spread_resid)` → finite small number (typical
   mean-reverting OU on a cointegrating pair).
4. `is_eligible_pair(spread, p_value=0.0012)` → `(True, "ok")`.
5. `CadenceEngine(spread_fn=ols_resid, cadence=Cadence(mode="bars",
   every=100)).run(panel, "BTC", "ETH", 200, 600)` → 5 refits at
   200, 300, 400, 500, 600; each emits a finite β.

## Sign-off

**PROCEED.** All five T2-pairs Python items land with G1+G2+G3 green.
HIGH-RISK 10-window + 50-T leak batteries both pass.

Next: T6 carry items (#38, #39, #39s, #40, #41, #42, #43) Python-
side, then Rust port for the whole Phase 3 surface, then cross-
language verify, then T2 + T6 ready for upstream push together.

Daniel Vieira Gatto — 2026-05-14.
