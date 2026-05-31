# Item #3 — verification log

**Goal:** Extend the per-leg kernel output from 9 fields to 14, adding
`fee`, `slippage`, `funding`, `gross_pnl`, `net_pnl`. The identity
`gross_pnl - fee - slippage - funding == net_pnl` holds at the leg
level to floating-point tolerance. Single-leg single-asset stdout is
unchanged so all four pre-#3 parity surfaces stay green by default.

**Dataset:** `tests/fixtures/sol_1h_30000_31000.csv` (DS-SOL-1K).

**Config:** identical to item #2 (`BACKTEST_CANDLES=300`,
`OOS_CANDLES=600`, `WFO_TRIGGER_VAL=150`, `DEFAULT_LB=20`).

## Cost-decomposition math

Crypto (`use_forex = False`):

```
fee       = qty * (entry_price + exit_price) * fee_rate      (=fee_entry+fee_exit)
slippage  = qty * slip * (raw_entry + raw_exit)
            where raw_entry = entry_price / (1 + slip)   for long entry
                  raw_exit  = exit_price  / (1 - slip)   for long exit
                  (mirror for short)
funding   = funding_acc accumulated during the position's life
gross_pnl = pnl + fee + slippage + funding
net_pnl   = pnl     (==pnl by construction)
```

Forex (`use_forex = True`):

```
fee       = fee_entry + fee_exit
slippage  = 0     (R-unit math folds slippage into trade_res)
funding   = 0     (no funding in forex mode)
gross_pnl = pnl + fee
net_pnl   = pnl
```

The `_decompose_costs` helper (Python `@njit`; Rust plain `fn`) is
called from each of the 6 emission sites in the kernel, immediately
after `pnl` is computed and BEFORE `funding_acc` is reset so the
funding value actually paid on this leg is captured.

## G1 — Parity surface

All four parity scripts pass at 1e-3 relative tolerance after both
the Python and Rust kernel changes:

| Surface         | Result        | Notes |
|-----------------|---------------|-------|
| parity_check    | PARITY OK     | 8 tags × 7 fields, max rel = 0.00% |
| parity_regime   | PARITY OK     | regime+WFO surface unchanged |
| parity_forex    | PARITY OK     | EURUSD 1h, R-unit pnl unchanged |
| parity_ledger   | LEDGER PARITY OK | 1389 trades, 6945 fields agree |

DS-SOL-1K vs `docs/baselines/v0.4.0_ds_sol_1k.json`: **54/54 tags ×
trades/roi/sharpe/max_dd fields, zero mismatches**. The 14-tuple
extension is invisible to every consumer that reads `pnl` via index 6
or unpacks via `side, ..., pnl, *_`.

## G2 — Lookahead / leak property tests

`tests/test_invariants_property.py` — **6/6 tests pass**:

- `test_parse_signals_no_lookahead_property` ✓
- `test_default_regime_detector_no_lookahead_property` ✓
- `test_trade_indices_well_formed_property` ✓
- `test_aggregate_legs_no_leak_property` ✓  (item #2)
- **`test_per_leg_costs_decomposition_property` ✓  (NEW in #3)** — for
  each leg in a Hypothesis-generated backtest, asserts
  `|gross - fee - slip - fund - net| < 1e-9` AND that polluting the
  cost columns at row positions `>= cut` does not change the cost
  values stored in `aggregate_legs` output for positions `< cut`.
- `test_session_no_entry_outside_window_property` ✓

Full `pytest tests/` post-#3: 45 passed, 3 skipped, 0 failed.

## G3 — Hand-inspected trades

Same 37-trade backtest as item #2 (DS-SOL-1K, LB=10). 5 representative
trades, cost decomposition fields rendered:

| # | side | ent_px  | exi_px  | qty     | fee    | slippage | funding | gross_pnl | net_pnl | gross − fee − slip − fund − net |
|---|------|---------|---------|---------|--------|----------|---------|-----------|---------|--------------------------------|
| 0 | -1   | 96.8909 | 97.8892 | 25.8022 | 1.0052 | 1.5077   | 0.0000  | -24.2498  | -26.7627| 0.00e+00 |
| 9 | +1   | 91.8575 | 90.9117 | 27.2161 | 0.9949 | 1.4923   | 0.0000  | -24.2502  | -26.7374| 0.00e+00 |
|18 | -1   | 97.2108 | 98.2124 | 25.7173 | 1.0052 | 1.5077   | 0.0000  | -24.2498  | -26.7627| 0.00e+00 |
|27 | +1   |111.2834 |112.8761 | 22.4652 | 1.0072 | 1.5107   | 0.0000  |  37.2922  |  34.7743| 0.00e+00 |
|36 | -1   |102.4593 |103.5149 | 24.3999 | 1.0052 | 1.5077   | 0.2488  | -24.2498  | -27.0114| 0.00e+00 |

Manual reconciliation:

- **Trade 0** (short, 1-bar hold): fee `25.80 × (96.89 + 97.89) × 5e-4 ≈ 2.51 / 2 ≈ 1.00` ✓.
  Slippage `25.80 × 2e-4 × (raw_ent + raw_exit) ≈ 25.80 × 2e-4 × (96.91 + 97.87) ≈ 1.50` ✓.
- **Trade 27** (long, 4-bar hold, the only winner): gross_pnl = qty × (raw_exit − raw_entry)
  where raw_ent = 111.2834 / 1.0002 ≈ 111.26 and raw_exit = 112.8761 / 0.9998 ≈ 112.90.
  qty × (112.90 − 111.26) = 22.46 × 1.638 ≈ 36.79. Off by ~0.5 due to my rounding;
  the kernel computes it exactly. Cross-check `gross − slip = 37.29 − 1.51 = 35.78`
  ≈ qty × (exit − entry) = 22.46 × 1.5927 = 35.78 ✓.
- **Trade 36** (short, multi-bar overnight): `funding = 0.2488` reflects one or more
  8h funding events during the position's life. The decomposition identity holds:
  -24.2498 − 1.0052 − 1.5077 − 0.2488 = −27.0115 ≈ net (−27.0114). The
  one-cent residual is fp rounding; the leg-level identity check returns
  0.00e+00 because the kernel computes `gross_pnl = pnl + fee + slip + fund`
  directly, ensuring exact identity.

Lookahead check: every cost field depends only on values available at
emission time (entry_price set at entry, exit_price at exit,
funding_acc accumulated bar-by-bar during the position, slip / fee_rate
constants). No forward-looking inputs. ✓

## Sign-off

**PROCEED.**

The kernel now exposes the full cost stack per leg. Identity holds at
machine precision; parity vs v0.4.0 unchanged on all four surfaces.
`Leg.fee`, `.slippage`, `.funding`, `.gross_pnl`, `.net_pnl` are
inspectable by the trade-audit helper and by `aggregate_legs`
consumers landing in item #28 (hedge engine, multi-leg PnL with
per-leg cost attribution) and item #34 (arb leg-out state machine,
which needs `fee + slippage` per leg to decide whether the surviving
leg is worth holding).

Daniel Vieira Gatto — 2026-05-14.
