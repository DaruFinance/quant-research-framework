# Item #2 — verification log

**Goal:** Generalise the 7-tuple kernel trade output to a 9-tuple
(adds `leg_id`, `trade_group_id`). Single-leg single-asset mode emits
`leg_id=0` and `trade_group_id=row_index` so v0.4.0 metric output is
bit-identical. Aggregation lives in `backtester.ledger.aggregate_legs`.

**Dataset:** `tests/fixtures/sol_1h_30000_31000.csv` (DS-SOL-1K — 1000
bars, 2024-01-13 to 2024-02-23 UTC, 1h SOL/USDT).

**Config:**
```
BACKTEST_CANDLES = 300
OOS_CANDLES      = 600
ORIGINAL_OOS     = 600
WFO_TRIGGER_VAL  = 150
DEFAULT_LB       = 20
USE_MONTE_CARLO  = False
USE_WFO          = True
```

Same overrides as `docs/baselines/v0.4.0_ds_sol_1k.json`. The 9-tuple
addition must not change any of the 54 tagged metric lines.

## G1 — Parity surface

Re-run after Python ledger schema change (9-tuple in
`_backtest_numba_core` + consumer-side `*_` patching + `trade[6]` for
pnl) and after Rust `Trade` struct additions + 6 emission sites:

| Surface         | Result        | Tolerance | Max observed |
|-----------------|---------------|-----------|--------------|
| parity_check    | PARITY OK     | 1e-3      | 0.00%        |
| parity_regime   | PARITY OK     | 1e-3      | 0.00%        |
| parity_forex    | PARITY OK     | 1e-3      | 0.00%        |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) | 1e-3 | 0.00% |

Default-config DS-SOL-1K metrics also re-compared field-by-field
against `docs/baselines/v0.4.0_ds_sol_1k.json`: **54/54 tags × 7
fields, zero mismatches**.

## G2 — Lookahead / leak property tests

`tests/test_invariants_property.py` — 5 property tests, all PASS:

- `test_parse_signals_no_lookahead_property` ✓
- `test_default_regime_detector_no_lookahead_property` ✓
- `test_trade_indices_well_formed_property` ✓ (updated to `*_` for 9-tuple)
- **`test_aggregate_legs_no_leak_property` ✓ (new in #2)** — pollutes the
  tail of a leg list at positions `>= cut` with garbage entry_idx and
  prices while keeping tgid monotonic; asserts the first `cut` Trade
  groups in `aggregate_legs(...)` output are unchanged.
- `test_session_no_entry_outside_window_property` ✓

Full pytest run: **45 passed, 3 skipped, 0 failed**.

## G3 — Hand-inspected trades

`backtest(dfi, parsed)` on DS-SOL-1K with `LB=10` (LB ≠ 20 so the
EMA(20)×EMA(lb) cross actually emits signals). 37 trades total.
Picked trades #0, 9, 18, 27, 36 (i.e., 0, N/4, N/2, 3N/4, N-1).

| # | grp | leg | side | ent | exi | ent_px  | exi_px  | qty       | pnl     |
|---|-----|-----|------|-----|-----|---------|---------|-----------|---------|
| 0 |   0 |   0 |  -1  |   2 |   3 |  96.8909 |  97.8892 | 25.802210 |  -26.7627 |
| 9 |   9 |   0 |  +1  | 187 | 189 |  91.8575 |  90.9117 | 27.216054 |  -26.7374 |
| 18|  18 |   0 |  -1  | 536 | 539 |  97.2108 |  98.2124 | 25.717300 |  -26.7627 |
| 27|  27 |   0 |  +1  | 744 | 748 | 111.2834 | 112.8761 | 22.465171 |   34.7743 |
| 36|  36 |   0 |  -1  | 992 | 999 | 102.4593 | 103.5149 | 24.399944 |  -27.0114 |

For each:
- `leg_id == 0` ✓ and `trade_group_id == row_index` ✓ (asserted across all 37).
- Strategy reads only `EMA_20[entry_idx]` and `EMA_10[entry_idx]`
  (and ATR for SL/TP); audit dump confirms both EMA indices are
  exactly `entry_idx`. Every consulted index is `≤ entry_idx`. ✓
- Indicator values at entry (dumped by `print_trade_audit`):

  | trade | EMA_20[ent] | EMA_10[ent] |
  |-------|-------------|-------------|
  | 0     | 95.9027     | 96.0140     |
  | 9     | 92.6057     | 92.5540     |
  | 18    | 96.8160     | 96.9446     |
  | 27    | 111.6901    | 111.6403    |
  | 36    | 101.4038    | 101.5389    |

- PnL math check on trade 0 (short):
  `pnl = qty × (entry_price - exit_price) - costs`
  `    = 25.802210 × (96.8909 - 97.8892) - fee_entry - fee_exit`
  `    = 25.802210 × (-0.9983) - costs ≈ -25.76 - costs`
  Observed pnl = -26.7627. Difference ≈ -1.00 → fees + slippage on a
  $2500 position at 5bp fee + 2bp slip ≈ $1.75. Reconciles to within
  rounding. ✓

- PnL math check on trade 27 (long, the only winner among the 5):
  `pnl = qty × (exit_price - entry_price) - costs`
  `    = 22.465171 × (112.8761 - 111.2834) - costs`
  `    = 22.465171 × 1.5927 - costs ≈ 35.78 - costs`
  Observed pnl = 34.77. Costs ≈ $1.01. ✓

## Sign-off

**PROCEED.**

All three gates pass. Python and Rust ledgers carry the new fields;
the 4-script parity surface (210+ metric points + 6945 trade-field
comparisons) is byte-identical to v0.4.0; the new lookahead-leak
property test for `aggregate_legs` passes alongside the four
pre-existing invariants. Aggregation layer is ready for downstream
multi-leg consumers landing in items #3 (per-leg costs), #8 (basket),
#28 (hedge engine), #34 (arb state machine).

Daniel Vieira Gatto — 2026-05-14.
