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

__all__ = ["load_panel", "PanelGapError", "PanelSchemaError", "PanelData"]
