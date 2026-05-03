# Contributing to quant-research-framework

This is a research-grade backtester whose central design discipline is
**parity with the Rust port** at
[`quant-research-framework-rs`](https://github.com/DaruFinance/quant-research-framework-rs).
That parity is what gives every other claim — performance, walk-forward
fit, robustness — its credibility. Contributing here means preserving
the discipline.

## The parity invariant

> Any change that alters engine semantics must keep the metric-output
> diff against the Rust port within $10^{-3}$ relative tolerance on
> the three published surfaces (default-config, regime+WFO, forex).

If your change is intentional and changes outputs, the Rust port must
land a matching change in the same release cycle. That is the only path
to merge.

If your change is **not** intended to alter outputs (refactor, docs,
type hints, a faster but equivalent indicator implementation), the
parity diff must still report metric agreement within the $10^{-3}$
relative tolerance band on the surfaces it currently passes. Run the
four-command checklist below before opening a PR.

## Four-command verification checklist

From the repo root, with `quant-research-framework-rs` checked out as a
sibling directory at the matching tag:

```sh
# 1. Property tests on the strategy contract.
pytest tests/

# 2. Default-config parity (56 metric points, 8 stages, SOLUSDT 1h).
python ../quant-research-framework-rs/tools/parity_check.py \
    --csv ../quant-research-framework-rs/data/SOLUSDT_1h.csv --tol 0.001

# 3. Regime + WFO parity (98 metric points, 14 stages, SOLUSDT 1h).
python ../quant-research-framework-rs/tools/parity_regime.py \
    --csv ../quant-research-framework-rs/data/SOLUSDT_1h.csv --tol 0.001

# 4. Forex-mode parity (56 metric points, 8 stages, EURUSD 1h).
python ../quant-research-framework-rs/tools/parity_forex.py \
    --csv ../quant-research-framework-rs/data/EURUSD_1h.csv --tol 0.001
```

All four must pass. The Rust binary must be built from the matching
commit (`cargo build --release` in the sibling repo); CI does this
automatically on push.

## What kinds of changes are welcome

- **Strategy plugins** under `examples/` that use only the documented
  contract from Appendix B of the paper. No engine internals.
- **Indicators** added to `indicators_tradingview.py` (or
  `backtester/indicators.py` after the v0.3.0 package decomposition),
  matching TradingView's reference output where possible.
- **Performance** improvements to the Numba core that are
  semantically identical (parity diff still passes within $10^{-3}$).
- **Tests** under `tests/`, especially new property-based invariants on
  the strategy contract.
- **Documentation** of any kind. The framework is under-documented
  relative to its surface area.

## What requires extra care

- **Engine internals** (`_backtest_numba_core`, `parse_signals`,
  `_backtest_numba_core` helpers): a change here usually shifts metric
  outputs and requires a co-ordinated change in the Rust port + a paper
  CHANGELOG entry. Open an issue first.
- **The strategy contract** (`create_raw_signals` signature, the
  $\le i-1$ obligation): changing it is a major-version bump. Open an
  issue first; the parity scripts and the contract appendix in the
  paper both depend on the current shape.
- **Default constants** (FEE_PCT, FUNDING_FEE, SL_PERC, TP_PERC,
  …): these change every metric in every test. Coordinate with the
  Rust port's `FEE_PCT_DEFAULT` etc.

## Code style

- `ruff format` for formatting; `ruff check` for linting.
- Type hints on new public functions; existing untyped surface is being
  migrated incrementally.
- Docstrings on new public functions, especially anything a user might
  monkey-patch from outside.
- No trailing comments that explain *what* the code does ("# loop over
  bars"); reserve comments for *why* (hidden constraint, citation,
  workaround).

## Releases

- Update `CHANGELOG.md` for any user-visible behaviour change.
- Bump the version in `pyproject.toml` and the `CITATION.cff` block.
- Tag the release; GitHub Actions drives the Zenodo archive on tag
  creation.
- Tag the matching commit on `quant-research-framework-rs` with the
  same version suffix (e.g. Python `v0.4.0` ↔ Rust `v0.3.3`).

## Reporting bugs

Look-ahead leaks and parity-diff failures are the highest-priority bug
class. If you find one, the property tests in
`tests/test_invariants.py` and the four-command checklist above are
the reproducer template. Open an issue with the failing checklist
output attached.
