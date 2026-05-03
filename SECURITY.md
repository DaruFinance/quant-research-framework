# Security policy

This is a research framework that processes user-supplied OHLC CSV
data and runs deterministic numerical computations. It is not an order
router, broker, or live-trading system, and it does not handle
credentials. The realistic threat surface is therefore narrow:

- malformed input CSVs causing crashes or incorrect numerics that go
  undetected (i.e. numerical-correctness bugs that survive the parity
  checks);
- pickled-numpy or pickled-pandas serialisation in any future
  caching/persistence layer (none today);
- a third-party dependency with a known CVE flowing in transitively
  via `pyproject.toml`.

If you find any of the above, please report it privately so we can
ship a fix before disclosure.

## Supported versions

| Version line | Supported          |
|--------------|:------------------:|
| 0.3.x        | :white_check_mark: |
| 0.2.x        | :x: (please upgrade) |
| 0.1.x        | :x:                |

## How to report

Email **77agdg@gmail.com** with subject prefix `[security] qrf:`.

Include:

- the framework version (from `pyproject.toml` or `bt.__version__`);
- a minimal reproduction (a small CSV plus the command you ran);
- the observed wrong behaviour and the expected behaviour;
- if you believe the issue propagates into the Rust port, please copy
  the report to the same address and prefix the subject `[security] qrf-rs:`.

A first response targets within 7 days. Please do not open a public
GitHub issue for security reports until a coordinated disclosure
window has passed.
