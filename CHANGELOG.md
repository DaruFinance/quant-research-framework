# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] — 2026-04-26

### Fixed
- **Session-end force-close now actually fires.** Two underlying bugs in
  `_backtest_numba_core` / `_prepare_backtest_inputs`:
  1. `session_end_mask` was set to `times.dt.time == t_end` — i.e. only
     bars whose NY local time *exactly* equalled SESSION_END
     ("16:50") qualified. Bars at any other minute (e.g. crypto bars
     at HH:00, or the synthetic HH:26:40 fixture) silently never
     triggered the force-close, so positions could carry across
     out-of-session windows. Replaced with the standard "last
     in-session bar of each contiguous in-session run" detection.
  2. The force-close branch was guarded on `code != 0`, meaning even
     when the mask was correct the force-close required a strategy
     signal on the closing bar to fire. Removed that guard.
- The previously-xfail test `test_session_blocks_overnight_carry_known_quirk`
  is now a regular passing test
  (`test_session_force_close_prevents_overnight_carry`).

### Notes
- These changes touch behaviour ONLY when `TRADE_SESSIONS = True`. None
  of the currently published research uses session mode, so v0.2.0–0.2.2
  numbers from default-config (sessions off) runs are unaffected. The
  v0.1.0 parity harness still reports 56/56 byte-identical.

## [0.2.2] — 2026-04-26

### Added
- **Property-check suite** (`tests/test_invariants.py`, 9 tests + 1
  documented xfail). Checks the engine's actual outputs rather than
  just whether flags wire through:
  - No trade entry on a bar whose NY local time is outside the session
    window
  - `create_regime_signals` uses the slow-EMA selected by `best_lbs[regime[i]]`
    on every bar (validated against a hand-computed reference with
    deliberately distinct per-regime LBs)
  - Forex per-trade PnL clamped to `[-1R, +RRR×1R]` band (plus fee slack)
  - OOS2 invariants: `OOS_CANDLES = 2 × ORIGINAL_OOS` when the flag is on
  - Trade tuples are well-formed (sides ∈ {-1,1}, indices in range,
    entry ≤ exit, prices > 0, PnL finite)
  - parse_signals is look-ahead-clean (mutating raw[cut:] doesn't change sig[:cut])
  - Default regime detector is look-ahead-clean
  - WFO+regime cadence: optimize_regimes_sequential always called with
    BACKTEST_CANDLES-sized IS slices, never with regime-stretch sizes
- **Documented Python design quirk** (xfail test): `_backtest_numba_core`
  guards session-end force-close on `code != 0` (line 950), so when no
  signal lands on the last in-session bar, the position carries across
  out-of-session windows. Removing the guard would change every
  published research number, so it stays documented. The Rust port
  mirrors it for parity.

### Verified
- v0.1.0 parity still 56/56 byte-identical at 0.1% tol.
- 24 tests passing total + 1 documented xfail.

## [0.2.1] — 2026-04-25

### Added
- **Behavioural test suite** (`tests/test_behavioural.py`) covering forex
  mode (PnL semantics actually change), session mode (out-of-session
  entries are masked), OOS2 (split-point preservation), regime LB
  rotation (per-regime dict keys come from `REGIME_LABELS`), and the
  news-injection robustness scenario (bar series is actually perturbed).
- **Cross-language parity harness** at the sibling Rust repo's
  `tools/parity_check.py`. Runs both engines on the same dataset with
  matching defaults and asserts agreement on baseline + optimised + WFO
  metrics. Verified at byte-identity (0% relative diff) across 8 tag
  groups × 7 metrics on the bundled SOLUSDT_1h dataset.

### Notes
- This release adds verification, not behaviour. The engine code paths
  for forex / session / OOS2 / regime did not change relative to v0.2.0;
  what changed is that those paths now have explicit tests and a
  cross-language oracle backing the parity claim.

## [0.2.0] — 2026-04-25

### Added
- **Pluggable regime-detector contract.** New `REGIME_LABELS` (length 2..5)
  and `detect_regimes(df) -> pd.Series` module-level seam. The default
  detector wraps the existing EMA-200 / 8-bar 3-regime logic. Users can
  supply their own analytic or ML-based detector by reassigning these two
  symbols. `optimize_regimes_sequential`, `walk_forward`,
  `evaluate_filters` and `backtest_continuous_regime` all flow through the
  new contract automatically.
- **ML signal hooks.** Two new examples mirror the two recommended
  patterns:
  - `examples/ml_precomputed/` — train offline, attach a `pred` column
    to the OHLC frame, threshold inside the strategy function. Fastest
    path; framework-agnostic.
  - `examples/ml_callback/` — keep a model in memory and call
    `predict(features)` per bar; supports online/stateful inference.
- **Custom regime detector example.** `examples/regime_custom/` ships
  three demos (2-regime volatility, 4-regime trend×vol, 5-regime
  ML-style) showing the contract in action with REGIME_LABELS sets of
  different lengths.
- `pyproject.toml` (PEP 621), `__version__` constant, `CHANGELOG.md`.
- `examples/README.md` — new sections covering the ML and custom-regime
  contracts.

### Changed
- **WFO + regime segmentation no longer re-anchors the IS window on every
  regime change.** Previously the OOS was sliced by regime boundaries and
  each contiguous regime stretch became its own WFO window with its own
  IS history. The fixed cadence now matches the no-regime path
  (`WFO_TRIGGER_MODE`, `WFO_TRIGGER_VAL`); regime segmentation only
  controls which per-regime LB is active for each OOS bar inside a
  window. This is a correctness fix — backtest values for runs with
  `USE_WFO = True` *and* `USE_REGIME_SEG = True` will differ from
  `0.1.0`. Runs with either flag off are unaffected.
- `_run_wfo_window` now accepts an optional `best_lbs` dict; when set,
  the per-bar active LB rotates with the regime label. Robustness
  overlays inside WFO windows respect the rotation as well.

### Deprecated / Known limitations
- The `make_codes` confluence mask still applies only to entry codes
  (1, 3) and not to exits (2, 4). Adding an opt-in `MASK_EXITS` flag is
  tracked for `0.3.0`.
- The integer-vs-string `side` comparison bug at lines 1260, 1599, 1719,
  1837 (RRR optimisation paths) is unchanged in `0.2.0` to preserve
  existing published research numbers. The fix is tracked for `0.3.0`
  alongside an explicit migration note.

## [0.1.0] — 2026-03 (backfilled)

Initial public release. Contained:

- IS/OOS baseline + walk-forward optimisation (anchored or sliding
  cadence by candle count or trade count).
- Optimiser with coarse/fine search and optional RRR optimisation.
- Robustness suite: fee shock, slippage shock, news-candle injection,
  entry drift, indicator-variance perturbation, plus a queued
  multi-scenario runner.
- Monte Carlo (bootstrap + permutation) for IS validation.
- Forex mode (pip-based risk units, JPY-aware pip size, no funding
  fees), session mode (NY hours with force-close on session end), OOS2
  split, regime segmentation (EMA-200 / 8-bar 3-regime).
- ATR-cross example, synthetic OHLC generator, Binance downloader.
