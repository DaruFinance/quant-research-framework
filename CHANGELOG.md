# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
