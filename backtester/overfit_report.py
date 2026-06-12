"""Opt-in overfitting-statistics report block (item #3, part C).

Emits additive stdout lines: Deflated Sharpe (DSR), Probabilistic Sharpe
(PSR), Probability of Backtest Overfitting (PBO), Minimum Track-Record
Length (MTRL), Minimum Backtest Length (MBTL), Harvey-Liu haircut.

Every line begins with a two-space indent and a distinct prefix
(``  DSR |``, ``  PSR |``, ``  PBO |``, ``  HCUT |``, ``  MTRL |``,
``  MBTL |``, ``  INFO |``, ``  WARN |``, ``  ----``). NONE carry the
``| Trades: ... ROI: ... PF: ... Shp: ... Win: ...% Exp: ... MaxDD:``
metric body that ``parity_common.LINE_RE`` matches, so ``parse_metrics``
ignores them and the existing parity harnesses stay byte-identical. (The
indent alone does NOT protect them — LINE_RE starts with ^\\s* — the
absence of the Trades: body does. Enforced by tools/parity_overfit_lines.py.)
This module is invoked only when the ``OVERFIT_REPORT`` flag is on.

CRITICAL — effective trials. ``trial_sharpes`` MUST be the distinct
strategies tried in ONE in-sample optimisation (the distinct lookbacks
evaluated by ``_optimiser_impl``), NOT a concatenation across WFO
windows. ``len(trial_sharpes)`` is the trial count N for DSR/PSR/MinTRL.

CRITICAL — Sharpe convention (Lens B D2/D6). DSR/PSR/MinTRL require the
non-annualised per-observation estimator sqrt(T)*mean/std. The CHOSEN
Sharpe is recomputed HERE from ``oos_returns`` in that convention; it is
NOT taken from the engine's met_is['Sharpe'] (in-sample, possibly
annualised). The haircut and MinBTL use the per-period SR = mean/std of
the same OOS returns.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from backtester import dsr as _dsr
from backtester import pbo as _pbo
from backtester import haircut as _haircut


def _scaled_sharpe(rets: np.ndarray) -> float:
    """Non-annualised per-observation Sharpe sqrt(T)*mean/std(ddof=1),
    matching the Bailey-LdP estimator (and multitest.py:103-105). 0.0 if
    fewer than 2 finite obs or std<=0."""
    r = rets[np.isfinite(rets)]
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(math.sqrt(r.size) * r.mean() / sd)


def _per_period_sr(rets: np.ndarray) -> float:
    """Dimensionless per-bar SR = mean/std(ddof=1); 0.0 on degenerate."""
    r = rets[np.isfinite(rets)]
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(r.mean() / sd)


def emit(
    trial_sharpes: Sequence[float],
    oos_returns: Sequence[float],
    sharpe_mode: str = "trade",
    equity_matrix: Optional[np.ndarray] = None,
    pbo_S: int = 16,
    sr_benchmark: float = 0.0,
    prob: float = 0.95,
    haircut_freq: float = 252.0,
    haircut_method: str = "bhy",
) -> None:
    """Print the overfitting report block to stdout.

    Parameters
    ----------
    trial_sharpes : distinct in-sample trial Sharpes (effective N).
    oos_returns   : OOS per-bar return series of the chosen strategy.
                    The chosen Sharpe is recomputed FROM THIS series.
    sharpe_mode   : the engine's SHARPE_MODE that produced trial_sharpes;
                    used only to WARN when the trial scale is annualised.
    equity_matrix : (T, N) OOS-equity matrix for PBO/CSCV; None -> skip.
    pbo_S         : CSCV fold count (default 16, the paper value).
    sr_benchmark  : benchmark Sharpe for PSR/MinTRL (sqrt(T)-scale).
    prob          : target confidence for MinTRL.
    haircut_freq  : periods/year for the haircut annualisation.
    haircut_method: {'bonferroni','bhy'} (default 'bhy').
    """
    trials = [float(s) for s in trial_sharpes if math.isfinite(float(s))]
    n_trials = len(trials)
    rets = np.asarray(oos_returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    T = int(rets.size)

    # Recompute the chosen Sharpe FROM the OOS returns in the Bailey-LdP
    # convention so the statistic and the sample agree (Lens B D2).
    sr_chosen = _scaled_sharpe(rets)
    sr_pp = _per_period_sr(rets)

    print("  ---- Overfitting diagnostics (opt-in; non-parity lines) ----")
    print(f"  INFO | effective trials N={n_trials} (distinct strategies, "
          f"NOT windows*combos)  |  OOS bars T={T}  |  "
          f"SR_chosen(sqrtT)={sr_chosen:.4f}")
    if sharpe_mode != "trade":
        print(f"  WARN | SHARPE_MODE={sharpe_mode!r}: trial Sharpes are "
              f"annualised; DSR/PSR scale differs from the Bailey-LdP "
              f"per-observation estimator. Read DSR/PSR with care.")

    # DSR
    try:
        print(_dsr.report(sr_chosen, trials, rets))
    except Exception as e:                                    # pragma: no cover
        print(f"  DSR  | unavailable: {e}")

    # PSR
    try:
        psr = _dsr.probabilistic_sharpe_ratio(sr_chosen, rets, sr_benchmark)
        print(f"  PSR  | SR_chosen:{sr_chosen:6.2f}  "
              f"SR*:{sr_benchmark:5.2f}  P(SR>SR*):{psr:5.3f}")
    except Exception as e:                                    # pragma: no cover
        print(f"  PSR  | unavailable: {e}")

    # MTRL (single-trial)
    try:
        mtrl = _dsr.min_track_record_length(sr_chosen, rets, sr_benchmark, prob)
        mtrl_s = f"{mtrl:.1f}" if math.isfinite(mtrl) else "inf"
        print(f"  MTRL | target_conf={prob:.2f}  SR*:{sr_benchmark:5.2f}  "
              f"min_obs={mtrl_s}  (have {T})")
    except Exception as e:                                    # pragma: no cover
        print(f"  MTRL | unavailable: {e}")

    # MBTL (N-trial). sr_target MUST be the per-period SR so the result is
    # in observations, comparable to T (Lens C D2 / Lens B D5).
    if n_trials >= 1:
        try:
            mbtl = _dsr.min_backtest_length(n_trials, abs(sr_pp))
            mbtl_s = f"{mbtl:.1f}" if math.isfinite(mbtl) else "inf"
            print(f"  MBTL | N={n_trials}  SR_target(per-bar):{abs(sr_pp):6.4f}  "
                  f"min_obs={mbtl_s}  (have {T})")
        except Exception as e:                                # pragma: no cover
            print(f"  MBTL | unavailable: {e}")

    # PBO / CSCV
    if equity_matrix is not None:
        M = np.asarray(equity_matrix, dtype=float)
        if M.ndim == 2 and M.shape[1] >= 2 and M.shape[0] >= pbo_S:
            try:
                print(_pbo.report(M, S=pbo_S))
            except Exception as e:                            # pragma: no cover
                print(f"  PBO  | unavailable: {e}")
        else:
            print(f"  PBO  | skipped (need (T>=S,N>=2); got shape {M.shape}, "
                  f"S={pbo_S})")
    else:
        print("  PBO  | skipped (no equity matrix supplied)")

    # Harvey-Liu haircut. Annualise the per-period SR, then haircut. The
    # freq cancels inside haircut_sharpe_ratio (t_obs = sr_pp*sqrt(T)); the
    # annualise/de-annualise is presentation-only (Lens C D13).
    if n_trials >= 1 and T >= 2 and math.isfinite(sr_pp):
        try:
            sr_ann = sr_pp * math.sqrt(haircut_freq)
            print(_haircut.report(sr_ann, T, n_trials, haircut_method, haircut_freq))
        except Exception as e:                                # pragma: no cover
            print(f"  HCUT | unavailable: {e}")
    print("  -------------------------------------------------------------")
