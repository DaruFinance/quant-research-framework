"""Combinatorial-Symmetric Cross-Validation (CSCV) and the Probability of
Backtest Overfitting (PBO).

Bailey, Borwein, López de Prado, and Zhu (2014), "Pseudo-Mathematics and
Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample
Performance", Notices of the AMS 61(5):458--471. DOI 10.1090/noti1105.

Given a per-bar equity-curve matrix ``M`` of shape ``(T, N)`` for ``N``
strategies (or ``N`` parameter trials of the same strategy) over ``T`` bars,
CSCV partitions the time axis into ``S`` consecutive folds, takes every
``S/2``-sized subset ``C`` of those folds as an in-sample, and uses the
complement as out-of-sample. For each split:

  * pick the in-sample-best strategy ``n* = argmax_n SR_in(n)``;
  * compute its out-of-sample rank ``r_out(n*) ∈ {1, ..., N}``;
  * record the logit-relative rank
    ``lambda = log( r_out(n*) / (N + 1 - r_out(n*)) )``.

PBO is the fraction of splits with ``lambda <= 0`` --- i.e., the in-sample
maximiser ranks below the median out-of-sample. A PBO close to 0.5 indicates
the optimiser is selecting essentially at random; PBO close to 0 indicates
the optimiser's in-sample winners genuinely tend to win out-of-sample.

This is a *reference* implementation aligned with §3 of the Bailey--Borwein--
López de Prado--Zhu paper. The split count ``S`` defaults to 8 (so
``binom(8, 4) = 70`` evaluations); ``S = 16`` (``binom(16, 8) = 12,870``) is
the value used in the original paper, which is a few seconds of work for
``N`` in the low thousands and ``T`` in the tens of thousands.

Public API:

    pbo(equity_matrix, S=8) -> dict with 'pbo', 'lambdas', 'n_splits', ...

This module is post-processing only --- it does not run inside the engine
and does not affect cross-language parity. The Rust mirror is on the v0.5.x
roadmap.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np


def _sharpe(returns: np.ndarray) -> float:
    """Per-bar Sharpe with safe handling of degenerate inputs."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(r.mean() / sd) * math.sqrt(r.size)


def _warn_small_S(S: int, N: int, T: int) -> None:
    """RuntimeWarning if (S, N, T) is small enough to mis-classify
    one-shot ex-post-best overfitting (the documented "miss" mode of
    CSCV at default S=8)."""
    import warnings
    if S < 16 and (N < 50 or T < 10000):
        warnings.warn(
            f"CSCV with S={S}, N={N}, T={T} may under-detect one-shot "
            f"ex-post selection (the documented miss mode of small-S "
            f"CSCV; see Bailey-Borwein-LdP-Zhu 2014 §3). Consider "
            f"S=16 (the value used in the original paper) for a more "
            f"reliable test on small panels.",
            RuntimeWarning,
            stacklevel=2,
        )


def cscv(equity_matrix: np.ndarray, S: int = 16) -> dict:
    """Run CSCV on an equity matrix and return the logit-rank distribution.

    Parameters
    ----------
    equity_matrix : array-like of shape (T, N)
        ``T`` time-bars, ``N`` strategies. Each column is one strategy's
        equity curve (cumulative PnL); the per-bar return is the first
        difference.
    S : int, default 16
        Number of equal-length time folds. Must be even; the in-sample is
        any ``S/2``-sized subset of folds and the out-of-sample is the
        complement. Larger ``S`` produces more splits (``binom(S, S/2)``)
        but each fold is shorter. Default raised from 8 to 16 in commit
        37f84ab+: 16 is the value used in the original Bailey-Borwein-
        LdP-Zhu (2014) paper, reduces under-detection of one-shot
        ex-post selection on small panels, and remains fast
        (``binom(16, 8) = 12,870`` evaluations).

    Returns
    -------
    dict with keys:
        'pbo'       : float, the Probability of Backtest Overfitting
        'lambdas'   : ndarray of shape (n_splits,), the logit-rank values
        'n_splits'  : int, ``binom(S, S/2)``
        'S'         : int, the fold count
        'N'         : int, the number of strategies
        'T'         : int, the number of bars
    """
    M = np.asarray(equity_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("equity_matrix must be 2-D (T, N)")
    if S % 2 != 0:
        raise ValueError(f"S must be even, got {S}")

    T, N = M.shape
    _warn_small_S(S, N, T)
    if N < 2:
        raise ValueError(f"need at least 2 strategies, got N={N}")
    if T < S:
        raise ValueError(f"need at least S={S} bars, got T={T}")
    if N < 4:
        # CSCV's logit transform can hit infinity at the extremes when N is
        # tiny; warn the caller, but proceed.
        pass

    # Per-bar returns (first difference, last bar matches).
    rets = np.diff(M, axis=0, prepend=M[0:1])  # shape (T, N)

    # Partition T bars into S consecutive folds of (roughly) equal length.
    edges = np.linspace(0, T, S + 1, dtype=int)
    folds = [(edges[k], edges[k + 1]) for k in range(S)]

    half = S // 2
    fold_indices = list(range(S))

    lambdas = []
    for c in combinations(fold_indices, half):
        mask_in = np.zeros(T, dtype=bool)
        for k in c:
            a, b = folds[k]
            mask_in[a:b] = True
        mask_out = ~mask_in

        sr_in  = np.array([_sharpe(rets[mask_in, n])  for n in range(N)])
        sr_out = np.array([_sharpe(rets[mask_out, n]) for n in range(N)])

        n_star = int(np.argmax(sr_in))
        # OOS rank of n*: 1 = lowest, N = highest. ties broken by index.
        order = np.argsort(sr_out, kind="stable")
        rank = np.empty(N, dtype=int)
        rank[order] = np.arange(1, N + 1)
        r_out = int(rank[n_star])

        # logit transform. Clip to avoid divide-by-zero at the extremes.
        denom = N + 1 - r_out
        if denom <= 0 or r_out <= 0:
            lam = math.copysign(20.0, r_out - (N + 1) / 2.0)
        else:
            lam = math.log(r_out / denom)
        lambdas.append(lam)

    lambdas = np.asarray(lambdas)
    pbo = float((lambdas <= 0.0).mean())
    return {
        "pbo": pbo,
        "lambdas": lambdas,
        "n_splits": len(lambdas),
        "S": S,
        "N": N,
        "T": T,
    }


def pbo(equity_matrix: np.ndarray, S: int = 16) -> float:
    """Convenience wrapper returning just the scalar PBO."""
    return cscv(equity_matrix, S=S)["pbo"]


def report(equity_matrix: np.ndarray, S: int = 16) -> str:
    """One-line PBO summary suitable for appending to the engine's stdout."""
    out = cscv(equity_matrix, S=S)
    return (f"  PBO  | S={out['S']}  splits={out['n_splits']}  "
            f"N={out['N']}  T={out['T']}  PBO={out['pbo']:.3f}")
