"""Pairs / stat-arb plugin (Phase 3 items #9, #10, #11, #12, #13).

Optional package. Pulled via ``pip install quant-research-framework[pairs]``.
Builds on the panel substrate (item #1) to expose:

- Spread-definition primitives (log-ratio, OLS-residual, Kalman β,
  PCA-residual, ML-residual) — item #10.
- Spread screener (Engle-Granger, Johansen, distance/SSD, ...) — #9.
- Spread re-estimation cadence — #11 (HIGH-RISK).
- Spread-aware SL families — #12.
- Pre-screening eligibility filters — #13.
"""
from __future__ import annotations

try:
    import statsmodels  # noqa: F401
except ImportError as e:
    raise ImportError(
        "backtester.pairs requires the 'pairs' extras. Install with:\n"
        "    pip install quant-research-framework[pairs]\n"
        f"(underlying error: {e})"
    ) from e

from .spread import (  # noqa: E402,F401
    log_ratio,
    ols_resid,
    kalman_beta_spread,
    pca_resid,
    ml_resid,
    SpreadResult,
)
from .eligibility import (  # noqa: E402,F401
    half_life_ou,
    is_eligible_pair,
    EligibilityCriteria,
)
from .screener import (  # noqa: E402,F401
    screen_pairs,
    engle_granger,
    distance_ssd,
    ScreenedPair,
)
from .cadence import (  # noqa: E402,F401
    CadenceEngine,
    Cadence,
)
from .stops import (  # noqa: E402,F401
    z_multiple_stop,
    half_life_multiple_stop,
    breakdown_trigger_stop,
    StopReason,
)

__all__ = [
    "log_ratio", "ols_resid", "kalman_beta_spread", "pca_resid", "ml_resid",
    "SpreadResult",
    "half_life_ou", "is_eligible_pair", "EligibilityCriteria",
    "screen_pairs", "engle_granger", "distance_ssd", "ScreenedPair",
    "CadenceEngine", "Cadence",
    "z_multiple_stop", "half_life_multiple_stop", "breakdown_trigger_stop",
    "StopReason",
]
