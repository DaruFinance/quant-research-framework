"""Entry point for `python -m backtester`.

Runs the framework's main pipeline using the CSV at $BT_CSV (or
`data/your_ohlc.csv` if unset). All other configuration lives at the
top of `backtester/__init__.py` as module-level constants.
"""
from __future__ import annotations

from . import main

if __name__ == "__main__":
    main()
