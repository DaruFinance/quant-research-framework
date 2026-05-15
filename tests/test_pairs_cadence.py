"""Tests for the spread re-estimation cadence engine (item #11,
Phase 3 — HIGH-RISK)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
sm = pytest.importorskip("statsmodels")

from backtester.panel import PanelData, load_panel
from backtester.pairs import (
    Cadence,
    CadenceEngine,
    ols_resid,
)


HERE = Path(__file__).resolve().parent
FIXTURE_DIR = HERE / "fixtures" / "sources"
PAIR_PATHS = {
    "BTC": FIXTURE_DIR / "BTCUSDT_1h_pair_q3_2023.csv",
    "ETH": FIXTURE_DIR / "ETHUSDT_1h_pair_q3_2023.csv",
}


@pytest.fixture(scope="module")
def panel():
    return load_panel(PAIR_PATHS)


def test_bars_cadence_refits_every_n(panel):
    spread_fn = lambda p, a, b, t: ols_resid(p, a, b, t, lookback=60)
    engine = CadenceEngine(spread_fn=spread_fn,
                            cadence=Cadence(mode="bars", every=100))
    refits = engine.run(panel, "BTC", "ETH", t_start=200, t_end=600)
    refit_indices = [t for t, _ in refits]
    # First refit at t_start, then every 100 bars: 200, 300, 400, 500, 600.
    assert refit_indices == [200, 300, 400, 500, 600]
    for _, res in refits:
        assert np.isfinite(res.beta)


def test_trigger_cadence_refits_on_predicate(panel):
    spread_fn = lambda p, a, b, t: ols_resid(p, a, b, t, lookback=60)
    fire_at = [350, 450]
    def trig(spread, t):
        return t in fire_at
    engine = CadenceEngine(spread_fn=spread_fn,
                            cadence=Cadence(mode="trigger", trigger_fn=trig))
    refits = engine.run(panel, "BTC", "ETH", t_start=300, t_end=500)
    refit_indices = [t for t, _ in refits]
    assert refit_indices == [300, 350, 450]


def test_breakdown_cadence_does_not_storm(panel):
    spread_fn = lambda p, a, b, t: ols_resid(p, a, b, t, lookback=60)
    engine = CadenceEngine(spread_fn=spread_fn,
                            cadence=Cadence(mode="on_breakdown"))
    refits = engine.run(panel, "BTC", "ETH", t_start=200, t_end=1000)
    refit_indices = [t for t, _ in refits]
    # No two consecutive refits within 50 bars.
    for prev, cur in zip(refit_indices, refit_indices[1:]):
        assert cur - prev >= 50, f"refit storm: {prev} -> {cur}"


# ---------------------------------------------------------------------------
# HIGH-RISK: 50-T pollute battery for the β-refit cadence.
# ---------------------------------------------------------------------------
def test_cadence_50t_refits_leak_free(panel):
    """50 different schedules; for each, pollute panel rows past the
    last refit bar T. Every β fitted at a refit bar ≤ T must be
    bit-identical to the unpolluted run."""
    rng = np.random.default_rng(0xCA1F)
    spread_fn = lambda p, a, b, t: ols_resid(p, a, b, t, lookback=60)
    n = len(panel)

    for k in range(50):
        # Sample a refit schedule that ends at a random T.
        T = int(rng.integers(300, n - 1))
        engine = CadenceEngine(spread_fn=spread_fn,
                                cadence=Cadence(mode="bars",
                                                  every=int(rng.integers(50, 150))))
        clean = engine.run(panel, "BTC", "ETH", t_start=200, t_end=T)
        # Pollute rows past T.
        polluted_ds = panel.ds.copy(deep=True)
        for field in panel.fields:
            arr = polluted_ds[field].values
            arr[T + 1:] = rng.normal(100, 10, size=arr[T + 1:].shape)
            polluted_ds[field].values[...] = arr
        polluted_panel = PanelData(ds=polluted_ds)
        polluted = engine.run(polluted_panel, "BTC", "ETH", t_start=200, t_end=T)
        assert len(clean) == len(polluted), f"k={k} T={T}"
        for (ta, ca), (tb, cb) in zip(clean, polluted):
            assert ta == tb, f"k={k} T={T}: refit times differ"
            assert ca.beta == cb.beta, (
                f"k={k} T={T} refit@{ta}: β differs {ca.beta} vs {cb.beta}"
            )


def test_trigger_cadence_requires_fn():
    engine = CadenceEngine(
        spread_fn=lambda *args, **kw: None,
        cadence=Cadence(mode="trigger", trigger_fn=None),
    )
    with pytest.raises(ValueError, match="requires trigger_fn"):
        engine.run(None, "BTC", "ETH", 200, 500)  # type: ignore[arg-type]
