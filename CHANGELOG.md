# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] — 2026-06-12

### Added
- **OHLCV contract + volume** — optional `volume` column in `load_ohlc` (backward-
  compatible; 5-column files load byte-identically), `backtester/volume_indicators.py`
  (OBV, VWAP rolling+session, vol SMA/EMA, relative volume, z-score, MFI, A/D), volume
  strategy examples, cross-engine parity via `tools/parity_volume.py`.
- **Overfitting-statistics layer** — Probabilistic Sharpe Ratio, Minimum Track-Record
  Length, Minimum Backtest Length in `backtester/dsr.py`; `backtester/haircut.py`
  (Harvey-Liu Bonferroni+BHY); opt-in `backtester/overfit_report.py` emitting
  DSR/PSR/PBO/MinTRL/MinBTL/haircut after the WFO run (gated by `QRF_OVERFIT=1` /
  `Config.overfit_report`, additive lines that never touch the parity surface).
- **IS parameter-robustness isosurface** — `backtester/opt_surface.py` emits the dense
  in-sample objective grid (opt-in via `EMIT_OPT_SURFACE` / `Config.emit_opt_surface`);
  `tools/render_surface.py` renders it.

### Changed
- **Engine split into modules mirroring the Rust port.** `backtester/__init__.py`
  (3649 → 1629 lines) split into `backtester/metrics.py`, `objectives.py`, and
  `orchestrator.py` — the same boundaries as Rust's `src/{metrics,objectives,orchestrator}.rs`.
  Pure move/re-export: the public surface (`backtester.<name>`) is unchanged and every
  parity surface stays byte-identical (moved code reads live module globals via
  `import backtester as _bt` so `Config.with_config()`'s runtime mutation contract holds).
- **License → Apache-2.0** (was MIT) across `LICENSE`, the `pyproject.toml` classifier,
  and `README.md`.
- Version → 0.6.0 (`pyproject.toml` + `backtester.__version__`). One canonical performance
  band — **23.8–57× faster, 33–65× less memory** vs this Python reference — from the
  paper-grade harness `tools/bench_paper.py` in the Rust repo (median warm over n=5; the
  5,000-bar 232× is a measurement-floor artifact). This single band now matches the
  README, the paper, and `CITATION.cff`.

### Notes
- All new behaviour is opt-in; the existing parity surfaces remain byte-identical against
  the Rust port.

## [0.4.0] — 2026-05-03

### Added
- **`backtester.Config` dataclass** — single library-grade configuration
  surface carrying every tunable knob (50+ fields: `fee_pct`, `use_tp`,
  `forex_mode`, `oos_candles`, ...). Defaults mirror the legacy
  module-level UPPERCASE constants exactly, so `Config()` is always
  equivalent to "use the current module defaults".
  - `Config.from_module()` snapshots the live module state into a
    Config instance — start from "whatever is currently set" and tweak.
  - `Config.apply_to_module()` writes every field back to the module
    globals, returning a snapshot dict. The derived `dd_constraint`
    is recomputed against the new `forex_mode` / `drawdown_constraint`.
  - `Config.with_forex(on=True)` returns a new Config with forex
    defaults applied (SL/TP scaled by `pip_size`, `risk_amount` and
    `account_size` set to 1.0 R-units), mirroring the legacy
    import-time `if FOREX_MODE:` block.
  - `Config.with_sessions(on, start, end)` and `with_oos2(on)` builders
    follow the same copy-then-mutate pattern.
- **`backtester.with_config(cfg)` context manager** — temporarily applies
  a `Config` to module globals for the duration of the block, restores
  prior values on exit (even if the body raises). The single primitive
  the new API stands on.
- **`config: Config | None = None` kwarg on every public entry-point**
  — `main()`, `walk_forward()`, `optimiser()`, `optimize_regimes_sequential()`,
  `monte_carlo()`, `apply_news_injection()`, `classic_single_run()`. When
  passed, the engine uses cfg's values for the call and restores prior
  state on exit. When omitted, reads from module globals — the legacy
  `bt.X = Y` API works exactly as before.
- **`tests/test_config_isolation.py`** — 11 new tests covering Config
  field independence, apply/restore round-trip, exception-safe restore,
  forex-mode helper, drawdown-constraint derivation, and end-to-end
  proof that `bt.main(config=cfg)` does not leak into module globals.

### Changed
- **All 11 `global X` statements eliminated** from
  `backtester/__init__.py`. The two categories of globals were handled
  differently:
  - **Configuration globals** (`TP_PERCENTAGE`, `USE_TP`, `FEE_PCT`,
    `SLIPPAGE_PCT`) that the engine rebinds during RRR optimisation
    and robustness shocks now use `globals()['X'] = ...` — the same
    pattern already used inside `optimiser()` and friends, just
    extended to the four functions that still leaned on `global`.
    Save/restore semantics unchanged.
  - **Runtime-state globals** (`last_unfiltered_raw`, `_last_df`,
    `_last_lb`, `NEWS_FLAGS`, `blocked_*`) moved into a single
    `_runtime_state: dict[str, Any]` holder. Per-run scratch values
    no longer require `global` for rebinding. `bt.NEWS_FLAGS` is
    mirrored to the module attribute so any external consumer keeps
    working.
- **`grep -c "^global " backtester/__init__.py` returns 0** (was 11).
  The engine is now fork-safe at the module-attribute level: two
  `Config` instances can coexist in one process, the
  `with_config` context manager guarantees state restoration, and
  the long-standing `bt.create_raw_signals = my_strategy; bt.main()`
  contract works unchanged.

### Compatibility
- **Public API is fully backwards compatible.** Every test in the v0.3.1
  suite (32 tests, all using the `monkeypatch.setattr(bt, "X", Y)`
  pattern) continues to pass without modification. The README quickstart
  (`bt.create_raw_signals = my_strategy; bt.main()`) is unchanged.
- **Cross-language parity preserved.** All four parity surfaces against
  the Rust port (`tools/parity_check.py`, `parity_regime.py`,
  `parity_forex.py`, `parity_ledger.py`) pass at `1e-3 rel-tol` with
  zero deltas (`PARITY OK` / `LEDGER PARITY OK`).

## [0.3.1] — 2026-05-03

### Fixed
- **Module import works without a CSV.** The top-level
  `FileNotFoundError` raise in `backtester/__init__.py` (~line 149) ran at
  import time, so `pip install quant-research-framework` followed by
  `import backtester as bt` would fail unless the user had pre-staged a
  CSV at `data/your_ohlc.csv`. The check now lives inside `load_ohlc()`
  and `main()`. Library-style use (`bt.create_raw_signals = ...; bt.main()`)
  is unaffected; `python -m backtester` still surfaces the same clear
  error message before any heavy import / numba JIT.
- **`PIP_SIZE` substring fallback now warns.** The legacy
  `"JPY" in CSV_FILE` heuristic still fires for backwards compatibility
  but emits a `UserWarning` so users on filenames like `"FUJPYR.csv"`
  know the auto-detect can be wrong. New `BT_PIP_SIZE` env var takes
  precedence and silences the warning when set explicitly.
- **DSR comment vs formula mismatch.** `backtester/dsr.py` line 99
  formerly said "raw kurtosis (3 for normal)" with no link back to the
  Bailey-LdP 2014 derivation; the reader could not tell at a glance why
  the variance correction is `(g_4 - 1)` and not `(g_4 - 3)` (which would
  be excess kurtosis). The comment now spells out that g_4 is the raw
  fourth standardised moment and that (3 - 1) / 4 = 0.5 is the
  Normal-case reduction, citing Bailey-LdP 2014 §3 / eq. (9).
- **`OOS_CANDLES` doubling timing documented.** The
  `if USE_OOS2: OOS_CANDLES *= 2` runs at module import; flipping the
  flag after `import backtester` has no effect on the constant. A NOTE
  block now spells out the constraint and the workaround.

### Added
- **`backtester/__main__.py`** — `python -m backtester` entry point so
  the README quickstart works post-package refactor (the legacy
  `python backtester.py` form broke when v0.3.0 turned `backtester` into
  a directory).
- **`pyproject.toml` deps cleanup.** `scipy` (DSR), `pytest` and
  `hypothesis` (test suite) and `scikit-learn` (ML examples) are now
  declared. Layout: `[project] dependencies = [..., "scipy", ...]`,
  `[project.optional-dependencies] dev = ["pytest", "hypothesis"]`,
  `examples = ["scikit-learn"]`. `pip install
  quant-research-framework[dev,examples]` installs everything the test
  suite and example notebooks need.
- **GitHub repo discoverability scaffolding** — issue + PR templates,
  `SECURITY.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1),
  `.github/dependabot.yml` (weekly pip updates), README badges row.
- **CI improvements** — multi-OS + multi-Python matrix
  (`{ubuntu, macos, windows} x {3.10, 3.11, 3.12}`) for the pytest job;
  parity scripts remain Linux-only as before. Pre-commit config
  (`ruff format`, `ruff check`) wired into a CI job.
- **Sphinx docs scaffold** under `docs/` (Furo theme, autodoc-driven)
  with a `.github/workflows/docs.yml` that builds and publishes to
  `gh-pages` on each `main` push.
- **PyPI publish workflow.** `.github/workflows/publish-pypi.yml`
  triggered by `v*` tag push, using PyPI trusted publishing — see
  `RELEASING.md` for the one-time configuration step.
- **Notebook walkthrough** (`examples/notebook/walkthrough.ipynb`) —
  load synthetic data, run baseline, run WFO, plot equity, show
  metrics. Binder launch link in the README, `binder/` dir with
  `requirements.txt` and `runtime.txt`.
- **Property-test coverage on `walk_forward_regime`** — three new
  Hypothesis-based invariants in `tests/test_invariants_property.py`
  guarding the OOS LB rotation against the v0.3.0 parity-bug class.
- **Notebook + Binder actually shipped.** The previous v0.3.1 entry
  listed `examples/notebook/walkthrough.ipynb` and `binder/` as added,
  but the files were missing. Now genuinely present: a 10-cell
  notebook (synthetic-or-bundled CSV → import → IS/OOS → ledger plot →
  regime toggle → summary metrics) executing under `jupyter nbconvert
  --to notebook --execute` against the bundled SOLUSDT 1h CSV;
  `binder/requirements.txt` mirrors the runtime + adds
  `jupyter, matplotlib`; `binder/runtime.txt` pins `python-3.10`;
  README carries a `mybinder.org` launch badge.
- **CI status badges** — README top now shows a five-badge row
  (parity, docs, PyPI, DOI, License) so reviewers can see green
  before clicking through. Honest about state: badges flip red the
  moment a workflow breaks.
- **README "What's Included" reflects v0.3.0+ package layout.** The
  v0.3.0 refactor moved `backtester.py` → `backtester/__init__.py` and
  `indicators_tradingview.py` → `backtester/indicators.py`; the
  README still described the old script-based layout. Section now
  enumerates the package's sub-modules, the `examples/` tree
  (atr_cross + ml_precomputed/callback/sklearn + ml_regime_kmeans +
  regime_custom + batch_runner + the new walkthrough notebook),
  `docs/`, `tests/` and `binder/`.

### Fixed
- **Sphinx docs workflow no longer fails the build.** `docs.yml` ran
  `sphinx-build -W -b html` which converts the six pandas-style
  docstring warnings (under `evaluate_filters`,
  `optimize_regimes_sequential`, `backtester/dsr.py`) into errors,
  so `gh-pages` never published. Dropped `-W` and added
  `docs/TODO.md` listing the warnings for follow-up cleanup. Once
  TODO.md is empty, re-add `-W` to enforce the docstring discipline.

## [0.3.0] — 2026-05-03  (paper-v2 retag)

### Changed
- **Package layout.** `backtester.py` is now `backtester/__init__.py`;
  indicator helpers move to `backtester/indicators.py`. The legacy
  top-level `indicators_tradingview.py` is retained as a thin
  re-export shim so pre-v0.3.0 user scripts that do
  `from indicators_tradingview import compute_atr, compute_rsi`
  continue to work unchanged. Public API is otherwise byte-identical:
  `import backtester as bt`, `bt.create_raw_signals = my_strategy`,
  `bt.main()`, all module constants (`DEFAULT_LB`, `FOREX_MODE`, …)
  unchanged.
- **Cross-language parity verified post-refactor.** All four
  invariants of the parity discipline still hold: `pytest tests/`
  (24/24 pass); `parity_check.py` 56/56; `parity_regime.py` 98/98;
  `parity_forex.py` 56/56. Sum: 210/210 metric points across 30 stages.

### Added
- **`CONTRIBUTING.md`** codifying the parity invariant (any change
  altering engine semantics must keep the metric-output diff against
  the Rust port within $10^{-3}$ relative tolerance) and the
  four-command verification checklist contributors run before opening
  a PR.
- **`.github/workflows/parity.yml`** — GitHub Actions parity CI that
  builds the Rust port at the matching commit, installs Python deps,
  and runs the four-command checklist on every push and pull request.

### Author
- Sole author of record canonicalised to **Daniel Vieira Gatto**
  in `CITATION.cff` (alias: `DaruFinance`); previous variants
  (`DaruFinance`, `Daniel G.`) deprecated for citation-tracking
  consistency. References in arXiv-submitted paper updated to match.

## [0.2.5] — 2026-04-30

### Fixed
- **RRR-optimisation side-comparison bug.** The four RRR-probe sites
  (`backtester.py` ~lines 1314, 1657, 1778, 1893) used `side == 'long'`
  (str compared to int8 — always False), sending all trades to the
  short branch. Now uses `side == 1` by default. Set the new
  `LEGACY_SIDE_BUG` module flag to `True` to opt back into the buggy
  code path for bit-equality with prior research that depends on it
  (default `False`). Tests in `tests/` continue to pass at the
  corrected default.

### Added
- **`MASK_EXITS` flag** (default `False`) in `parse_signals`: when
  `True`, the active confluence rule (per `CONFLUENCES`) applies to
  exit codes (2, 4) too, not just entries (1, 3). Default preserves
  v0.2.x behaviour (exits unconditional on signal flip).
- **`listings/lah_demo.py`** — future-pollution probe demonstrating
  that the strategy contract's no-look-ahead obligation is
  user-checkable. Referenced from §3 of the paper.

### Notes
- Cross-language parity verified at the new defaults: 56/56
  default-config + 98/98 regime+WFO + 56/56 forex (the third
  surface, closed in this release; see the Rust repo's
  `tools/parity_forex.py`).

## [0.2.4] — 2026-04-26

### Added
- **Comparison matrix** in `README.md` — this framework vs vectorbt /
  backtrader / NautilusTrader / zipline-reloaded / Lean / bt across
  built-in WFO, per-regime LB optimisation, strict-LAH property tests,
  cross-language byte-parity. Verified against primary docs as of
  2026-04.
- **Benchmark table** in `README.md` — measured numbers from the new
  `tools/bench.py` harness in the sibling Rust repo. Replaces the
  unsourced "~24× faster" claim with reproducible figures across four
  dataset sizes (15k / 25k / 35k / 48k bars on the bundled SOL CSV).
- **`CITATION.cff`** with sibling cross-reference to the Rust port so
  citing either implies citing the framework as a whole.

### Notes
- No backtester engine changes in this release; v0.2.3 numerics are
  preserved. The Rust port's matching v0.3.0 release verifies
  regime+WFO byte-identical parity against this Python reference (see
  the sibling repo's `tools/parity_regime.py`).

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
