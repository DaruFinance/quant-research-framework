# Item #5(iter) — verification log

**Goal:** Add the panel route to the orchestrator dispatch table from
Phase 1 #5. ``walk_forward_panel(panel)`` iterates each asset's bar
slice through the existing single-asset WFO; result is a dict of
per-asset return tuples. Per-asset ledger is bit-identical to running
the single-asset WFO on the same asset alone (the equivalence
contract that future basket primitives in #6-#8 layer on top of).

**Dataset:** DS-PANEL-3 with Phase 0's fixture-tuned config
(`BACKTEST_CANDLES=300`, `OOS_CANDLES=600`, `WFO_TRIGGER_VAL=150`,
`DEFAULT_LB=20`).

## What landed

**Python (`backtester/panel/orchestrator.py`, new):**

- `walk_forward_panel(panel)` — iterates `panel.assets`, converts
  each asset's slice to the single-asset DataFrame shape (UNIX seconds
  → tz-aware NY datetime, OHLC columns), runs the existing
  `bt.walk_forward(df, ...)` entry, returns
  `{asset: (oos_rets, eq_wfo, rb_eq_curves, split_wfo_is)}`.
- `_walk_forward_panel_path(df_or_panel, ...)` — dispatch table entry.
  Asserts the input is a `PanelData` (not a DataFrame) and routes to
  `walk_forward_panel`. Phase 2 only registers `(regime=False,
  multi_asset=True)`; regime + panel composition lands at #44.
- Module-import side effect: registers the panel route in
  `backtester.orchestrator.DISPATCH` so the Phase 1 dispatcher knows
  about it. The `backtester.panel.__init__` does the eager import so
  the route is wired the moment a user touches the panel package.

**Rust (`src/panel/orchestrator.rs`, new behind `feature = "panel"`):**

- `pub fn bars_for_asset(panel, asset)` — extracts one asset's bar
  series as `Vec<Bar>`. Phase 2 minimum-viable contribution from the
  Rust side: the data-shape mirror that future Rust panel WFO will
  consume. The full per-asset orchestrator unification is gated on
  making `walk_forward` / `walk_forward_regime` `pub(crate)`, which
  is a follow-up when the Rust side moves panel-first.

## G1 — Parity surface

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

Phase 1 single-asset surfaces untouched (the panel route is dispatched
only when a `PanelData` is passed in).

## G2 — Property tests

**Python `pytest tests/test_panel_orchestrator.py`: 4 passed.**

- `test_panel_route_registered_under_multi_asset_true` — the Phase 1
  registry knows about `RouteKey(multi_asset=True)` after importing
  `backtester.panel`. ✓
- `test_panel_route_rejects_dataframe_input` — feeding a single-
  asset DataFrame to `_walk_forward_panel_path` raises `TypeError`
  naming the expected type. ✓
- `test_per_asset_run_matches_single_asset_run` — **the equivalence
  contract**. For each asset in DS-PANEL-3, the panel-WFO returns
  are bit-identical to the single-asset WFO returns on the same
  asset's slice. ✓ for SOL (18 oos rets), BTC (29), ETH (12).
- `test_panel_wfo_no_leak_under_tail_pollution` — truncate the panel
  at T=800; build a polluted parallel panel where one asset's
  OHLC after T is NaN; truncate that to T as well. Per-asset oos
  rets must be bit-identical across the two runs. ✓ (the pollution
  rows past T aren't read in the truncated window).

**Rust `cargo test --release --features panel`: 13 panel tests pass**
(was 11 pre-#5 iter; +2 from `src/panel/orchestrator.rs`):

- `bars_for_asset_matches_panel_close` — each extracted bar's
  time/close matches the panel.data cube cell-for-cell.
- `bars_for_asset_rejects_unknown_asset` — `DOGE` not in panel
  returns `Err(msg)` containing the asset name.

Full Python pytest sweep: **76 passed, 3 skipped, 0 failed** (was
72 pre-#5 iter; +4 from `test_panel_orchestrator.py`).

## G3 — Per-asset equivalence

End-to-end verification of the central claim:

For each asset in DS-PANEL-3 with the Phase 0 fixture-tuned config,
`walk_forward_panel(panel)[asset]` matches
`walk_forward(per_asset_df)` bit-identically on:

- `oos_rets` (NumPy array, exact comparison)
- `eq_wfo` (equity curve array, exact)

SOL: 18 oos rets, eq[-1] = 1.002250.
BTC: 29 oos rets, eq[-1] = 0.999443.
ETH: 12 oos rets, eq[-1] = 0.999566.

No metric drift; no off-by-one in the time-axis conversion; no
spurious cross-asset state because each asset is processed through
its own DataFrame.

## Sign-off

**PROCEED.**

The panel orchestrator route is live. Items #6 (ERC sizing), #7
(β-/$-/σ-neutral construction), #8 (long-short basket), and #44
(multi-term IS objective) now have a stable per-asset substrate to
build portfolio-level state on top of without touching the
single-asset hot loop.

Daniel Vieira Gatto — 2026-05-14.
