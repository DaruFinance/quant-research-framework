"""Multi-asset panel plugin (Phase 2 items #1, #4, #5(iter), #6-#8, #44-#45).

Optional package. Pulled in via ``pip install quant-research-framework[panel]``.
The plugin enables cross-sectional / basket strategies and the
multi-asset orchestrator routes registered by item #5(iter) (Phase 2).

Importing this package without the ``panel`` extras installed raises a
clear ``ImportError`` with the install hint, rather than the usual
``ModuleNotFoundError: No module named 'xarray'`` chain.
"""
from __future__ import annotations

try:  # noqa: SIM105 - explicit messaging on missing dep
    import xarray as _xr  # noqa: F401
except ImportError as e:
    raise ImportError(
        "backtester.panel requires the 'panel' extras. Install with:\n"
        "    pip install quant-research-framework[panel]\n"
        f"(underlying error: {e})"
    ) from e

from .loader import load_panel, PanelGapError, PanelSchemaError, PanelData  # noqa: E402,F401
from .regime import (  # noqa: E402,F401
    detect_regimes_panel,
    detect_regimes_panel_per_asset,
    detect_regimes_panel_market,
)
# The orchestrator submodule's import has a side effect: registering
# the multi_asset=True route in the central dispatch table. Import it
# eagerly so the route is wired before any user calls walk_forward_panel.
from .orchestrator import walk_forward_panel  # noqa: E402,F401
from .strategies import LongShortBasket, momentum_alpha  # noqa: E402,F401

__all__ = [
    "load_panel",
    "PanelGapError",
    "PanelSchemaError",
    "PanelData",
    "detect_regimes_panel",
    "detect_regimes_panel_per_asset",
    "detect_regimes_panel_market",
    "walk_forward_panel",
]
