"""Invariant-test framework for lookahead-free declarations.

Item #14 of the v2 extension plan: a registry-driven pollute-and-verify
harness so every new state-bearing function (regime detectors, spread
estimators, screeners, quoters in items #4, #9, #10, #11, #21, …)
gets a no-look-ahead property test automatically.

Usage
-----

::

    from backtester.invariants import registers_invariant

    @registers_invariant(name="my_regime_detector", data_kind="ohlc_df")
    def my_detector(df):
        return pd.Series(...)   # one label per bar

The decorator registers the function in a global list. The pytest case
in ``tests/test_invariants_property.py`` iterates the registry and for
each function generates random data, pollutes a suffix, and asserts
that the output for positions strictly before the cut is bit-identical
to the un-polluted call.

Pollution kinds
---------------

- ``"ohlc_df"`` — OHLC DataFrame; pollutes ``close/open/high/low`` at
  rows ``>= cut`` with NaN.
- ``"panel"`` — long-format Parquet-style panel
  (``time, asset, open, high, low, close``); pollutes rows where
  ``time >= cut_time`` (caller passes a time index, not a row index).
- ``"series"`` — 1-D Series of floats; pollutes positions ``>= cut``
  with NaN.

Adding a new data kind is a one-line addition to ``_default_pollute``.

The framework is **pure test infrastructure**: it does not touch the
kernel, does not change any metric, and the four parity surfaces stay
green by definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class InvariantSpec:
    name: str
    func: Callable[..., Any]
    data_kind: str = "ohlc_df"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"InvariantSpec(name={self.name!r}, kind={self.data_kind!r})"


_REGISTRY: List[InvariantSpec] = []


def registers_invariant(
    name: str,
    data_kind: str = "ohlc_df",
    **extra_kwargs: Any,
) -> Callable[[Callable], Callable]:
    """Mark a function as claiming lookahead-freeness.

    The function must accept the data argument as its first positional
    parameter (a DataFrame for ``ohlc_df``/``panel``, an array-like for
    ``series``). Additional kwargs declared here are forwarded to every
    call, allowing fixed-parameter functions like
    ``f(df, lb=20)`` to be registered with ``lb=20`` baked in.
    """
    def decorator(func: Callable) -> Callable:
        _REGISTRY.append(
            InvariantSpec(
                name=name,
                func=func,
                data_kind=data_kind,
                extra_kwargs=dict(extra_kwargs),
            )
        )
        return func

    return decorator


def list_invariants() -> List[InvariantSpec]:
    """Snapshot of currently-registered invariants. Tests parametrize
    over this."""
    return list(_REGISTRY)


def _clear_registry() -> None:
    """Test-only escape hatch for self-tests that register transient
    leakers. Production code never calls this."""
    _REGISTRY.clear()


def _default_pollute(data: Any, cut: int, data_kind: str) -> Any:
    """Return a copy of ``data`` with rows ``>= cut`` corrupted.

    Pollution NaNs all numeric columns relevant to the data kind. The
    point is that a lookahead-free function's output for indices
    ``< cut`` must not see NaN propagation from rows ``>= cut``.
    """
    if data_kind == "ohlc_df":
        polluted = data.copy()
        for col in ("close", "open", "high", "low"):
            if col in polluted.columns:
                polluted.loc[cut:, col] = np.nan
        if "EMA_200" in polluted.columns:
            polluted.loc[cut:, "EMA_200"] = np.nan
        return polluted
    if data_kind == "panel":
        polluted = data.copy()
        # cut here is interpreted as a row index over the long-form
        # panel rather than a timestamp; callers in pairs/panel land
        # convert from time → row count up-front.
        for col in ("open", "high", "low", "close"):
            if col in polluted.columns:
                polluted.loc[cut:, col] = np.nan
        return polluted
    if data_kind == "series":
        polluted = np.array(data, dtype=float).copy()
        polluted[cut:] = np.nan
        return polluted
    raise ValueError(f"unknown data_kind={data_kind!r}")


def _output_head(output: Any, cut: int) -> Any:
    """Normalise heterogeneous outputs (Series / np.ndarray / list /
    Dict[str, Series]) into a comparable head segment of length ``cut``."""
    if isinstance(output, pd.Series):
        return output.iloc[:cut].astype(str).reset_index(drop=True)
    if isinstance(output, pd.DataFrame):
        return output.iloc[:cut].astype(str).reset_index(drop=True)
    if isinstance(output, np.ndarray):
        return output[:cut]
    if isinstance(output, list):
        return output[:cut]
    if isinstance(output, dict):
        return {k: _output_head(v, cut) for k, v in output.items()}
    raise TypeError(f"cannot head-slice output of type {type(output)}")


def _heads_equal(a: Any, b: Any) -> bool:
    """Equality with NaN-tolerant comparison for floats."""
    if isinstance(a, pd.Series) or isinstance(a, pd.DataFrame):
        return bool((a == b).all().all() if isinstance(a, pd.DataFrame) else (a == b).all())
    if isinstance(a, np.ndarray):
        try:
            return bool(np.array_equal(a, b, equal_nan=True))
        except TypeError:  # non-numeric dtype
            return bool((a == b).all())
    if isinstance(a, list):
        return a == b
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_heads_equal(a[k], b[k]) for k in a)
    raise TypeError(f"cannot compare heads of type {type(a)}")


def assert_no_lookahead(
    spec: InvariantSpec,
    data: Any,
    cut: int,
    *,
    pollute: Optional[Callable[[Any, int], Any]] = None,
) -> None:
    """Run the registered function on ``data`` and a polluted copy;
    assert the first ``cut`` rows of the output agree.

    ``pollute`` overrides the default NaN-suffix corruption. Custom
    polluters are useful when the function reads non-standard columns
    or you want to assert tolerance against a specific kind of garbage.
    """
    polluted = (pollute or _default_pollute)(data, cut, spec.data_kind) \
        if pollute is None else pollute(data, cut)
    out_clean = spec.func(data, **spec.extra_kwargs)
    out_poll = spec.func(polluted, **spec.extra_kwargs)
    head_clean = _output_head(out_clean, cut)
    head_poll = _output_head(out_poll, cut)
    if not _heads_equal(head_clean, head_poll):
        raise AssertionError(
            f"invariant {spec.name!r} ({spec.data_kind}) leaked future data "
            f"into output[:{cut}]"
        )
