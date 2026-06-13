# Item #1 — verification log

**Goal:** Load N per-asset OHLC[+V] CSVs into a unified `time × asset
× field` panel. Strict inner-join on timestamps; reject if the joined
grid is non-uniform. Optional dep via `[panel]` extras / cargo feature.

**Dataset:** DS-PANEL-3 sources at
`tests/fixtures/sources/{SOL,BTC,ETH}USDT_*.csv` (1033/1033/1000 rows
each; intersection = 1000 rows).

## What landed

**Python (`backtester/panel/`, new package):**

- `backtester/panel/__init__.py`: lazy-import guard that raises
  `ImportError` with the install hint if `xarray` is missing.
- `backtester/panel/loader.py`:
  - `load_panel(paths: Mapping[str, Path]) -> PanelData`. Reads each
    CSV, validates schema, inner-joins on `time`, infers the modal
    inter-bar interval, and verifies no gaps. Returns an
    `xarray.Dataset` wrapped in a `PanelData` handle.
  - `PanelData` dataclass: `assets`, `times`, `fields`, `__len__`
    accessors. Holds the `xarray.Dataset` to keep the public surface
    decoupled from a specific xarray version.
  - `PanelGapError` / `PanelSchemaError` exception classes; the gap
    error carries the offending timestamp on `.ts`.
- `pyproject.toml`: `panel = ["xarray>=2024.0", "pyarrow>=14.0"]`
  optional extras; `packages = ["backtester", "backtester.panel"]`.

**Rust (`src/panel/`, new module behind `feature = "panel"`):**

- `Cargo.toml`: `[features] panel = ["dep:ndarray", "dep:csv"]`;
  `ndarray = "0.16"` and `csv = "1.3"` as optional deps.
- `src/panel/mod.rs`: re-exports `PanelData`, `PanelError`,
  `load_panel`.
- `src/panel/loader.rs`: mirrors the Python contract using
  `ndarray::Array3<f64>` for the `(time, asset, field)` cube. Same
  schema checks, same gap detection, same modal-interval inference.
- `src/lib.rs`: `#[cfg(feature = "panel")] pub mod panel;`.

## G1 — Parity surface

The panel plugin is pure new code; the default cargo build does not
even compile it. All four Phase 1 parity scripts remain green at 1e-3:

| Surface         | Result        |
|-----------------|---------------|
| parity_check    | PARITY OK     |
| parity_regime   | PARITY OK     |
| parity_forex    | PARITY OK     |
| parity_ledger   | LEDGER PARITY OK (1389 trades, 6945 fields) |

## G2 — Property tests

**Python `pytest tests/test_panel_loader.py`: 9 passed.**

- `test_load_panel_ds_panel_3_shape` — Dataset is 1000 × 3 × 4 OHLC,
  inferred interval = 3600s.
- `test_load_panel_5_random_cells_match_sources` — 5 random
  `(t, asset)` cells × 4 fields = 20 cell reconciliations, all
  bit-identical to the raw CSVs.
- `test_idempotent_loads_produce_identical_data` — loading the same
  paths twice yields element-wise-equal data arrays.
- `test_inner_join_drops_unmatched_timestamps` — when BTC is
  truncated to 500 rows, the intersection is ≤500.
- `test_gap_in_one_asset_raises_panel_gap_error` — dropping one BTC
  row from the middle raises `PanelGapError` whose `.ts` points to
  the bar AFTER the dropped one (post-join diff > 3600s there).
- `test_missing_required_column_raises_schema_error` — drop `high`
  from the SOL CSV → `PanelSchemaError("...high...")`.
- `test_duplicate_timestamps_raise_schema_error` — duplicate a row
  in the SOL CSV → `PanelSchemaError("...duplicate...")`.
- `test_empty_paths_dict_raises` — `load_panel({})` raises.
- `test_loader_no_lookahead_under_tail_pollution` — appending 50
  garbage rows to BTC at timestamps AFTER its natural end produces
  the same panel as the unpolluted load (inner-join semantics).

**Rust `cargo test --release --features panel`: 7 panel tests pass**
(was 4 pre-#1, +3 from `src/panel/loader.rs` inline tests).

- `ds_panel_3_shape` — same as Python.
- `idempotent_loads` — same as Python.
- `empty_paths_errors` — same as Python.

All pre-existing Rust suites unchanged (invariants 14, orchestrator
4, contract_v0_2, behavioural).

Full Python pytest sweep: **67 passed, 3 skipped, 0 failed** (was 58
pre-#1, +9 from `test_panel_loader.py`).

## G3 — Cell reconciliation against source CSVs

5 random cells were sampled deterministically with `np.random.default_rng(seed=42)`:

| ts          | asset | panel.close | src.close  | match |
|-------------|-------|-------------|------------|-------|
| 1707552000  | SOL   | 108.19      | 108.19     | OK    |
| 1705287600  | ETH   | 2503.32     | 2503.32    | OK    |
| 1706209200  | SOL   | 87.75       | 87.75      | OK    |
| 1706018400  | SOL   | 81.07       | 81.07      | OK    |
| 1707912000  | SOL   | 115.44      | 115.44     | OK    |

The test extends this to all 4 fields × 5 cells = 20 cell
comparisons; every one is bit-identical (`abs(panel - src) < 1e-12`).

## Sign-off

**PROCEED.**

The panel data plumbing is in place. Item #4 (cross-asset regime
detector with `data_kind="panel"`) plugs into `load_panel`'s output
directly; item #5(iter) registers the basket orchestrator route that
iterates per-asset over the panel's time axis using the same global
WFO window grid.

Daniel Vieira Gatto — 2026-05-14.
