"""Per-leg trade ledger aggregation.

Item #2 of the v2 extension plan generalises the kernel's 7-tuple trade
output to a 9-tuple that carries `leg_id` and `trade_group_id`. In
single-leg single-asset mode each kernel-emitted leg IS a logical trade
(one leg per group, `leg_id == 0`, `trade_group_id == row_index`), so
the metric output is bit-identical to v0.4.0. In future multi-leg modes
(pair trades, basket trades, hedged MM positions) the kernel emits one
leg row per leg and `aggregate_legs` groups them into logical Trade
records.

This module is **pure data-only**: every function reads only its input
tuples, never the surrounding bar series. That keeps `aggregate_legs`
trivially lookahead-free under the property-test framework introduced
in item #14.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Leg:
    """A single leg of a logical trade. Mirrors the kernel's 9-tuple."""
    side: int
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    leg_id: int = 0
    trade_group_id: int = 0


@dataclass
class Trade:
    """A logical trade composed of one or more legs sharing a group id."""
    group_id: int
    legs: List[Leg] = field(default_factory=list)

    @property
    def net_pnl(self) -> float:
        return sum(L.pnl for L in self.legs)

    def __repr__(self) -> str:
        # Short form when there is a single leg, so v0.4.0-era output
        # remains visually identical.
        if len(self.legs) == 1:
            L = self.legs[0]
            return (
                f"Trade(side={L.side}, ent={L.entry_idx}, exi={L.exit_idx}, "
                f"ep={L.entry_price:.4f}, xp={L.exit_price:.4f}, "
                f"qty={L.qty:.6f}, pnl={L.pnl:.4f})"
            )
        return f"Trade(group={self.group_id}, n_legs={len(self.legs)})"


def aggregate_legs(legs: Iterable) -> List[Trade]:
    """Group per-leg tuples by `trade_group_id`.

    Accepts the kernel's 9-tuple (preferred) or the legacy 7-tuple
    (treated as a single-leg trade per row, with synthetic `leg_id=0`
    and `trade_group_id=row_index`).

    Returns `List[Trade]` sorted by `group_id`, each Trade's `legs`
    sorted by `leg_id`. Pure data-only — no series indexing — so the
    function is trivially lookahead-free.
    """
    groups: defaultdict = defaultdict(list)
    for row_idx, t in enumerate(legs):
        if len(t) == 7:
            side, ent, exi, ep, xp, qty, pnl = t
            leg_id, tgid = 0, row_idx
        elif len(t) == 9:
            side, ent, exi, ep, xp, qty, pnl, leg_id, tgid = t
        else:
            raise ValueError(
                f"aggregate_legs: tuple width {len(t)} not in (7, 9)"
            )
        groups[int(tgid)].append(
            Leg(
                side=int(side),
                entry_idx=int(ent),
                exit_idx=int(exi),
                entry_price=float(ep),
                exit_price=float(xp),
                qty=float(qty),
                pnl=float(pnl),
                leg_id=int(leg_id),
                trade_group_id=int(tgid),
            )
        )
    trades = []
    for gid in sorted(groups.keys()):
        legs_in_order = sorted(groups[gid], key=lambda L: L.leg_id)
        trades.append(Trade(group_id=gid, legs=legs_in_order))
    return trades


def print_trade_audit(
    trades_or_legs: Iterable,
    df=None,
    n: int = 5,
    indicators: Sequence[str] = (),
) -> None:
    """Dump n representative trades for hand-inspection (G3 gate).

    Picks indices {0, N/4, N/2, 3N/4, N-1} (deduplicated for small N)
    and prints each trade's legs with the indicator values at
    `entry_idx`. The indicator dump is the cornerstone of the
    lookahead audit — every value listed has an index `<= entry_idx`.
    """
    legs_input = list(trades_or_legs)
    if not legs_input:
        print("  (no trades)")
        return
    if isinstance(legs_input[0], Trade):
        trades = legs_input
    else:
        trades = aggregate_legs(legs_input)

    N = len(trades)
    if N <= n:
        picks = list(range(N))
    else:
        picks = sorted({0, N // 4, N // 2, 3 * N // 4, N - 1})

    header = (
        "  # | grp | leg | side |   ent |   exi |     ent_px |     "
        "exi_px |       qty |       pnl"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in picks:
        T = trades[i]
        for L in T.legs:
            side_str = "+1" if L.side == 1 else "-1"
            print(
                f"  {i:>2d} | {T.group_id:>3d} | {L.leg_id:>3d} | "
                f"{side_str:>4s} | "
                f"{L.entry_idx:>5d} | {L.exit_idx:>5d} | "
                f"{L.entry_price:>10.4f} | {L.exit_price:>10.4f} | "
                f"{L.qty:>9.6f} | {L.pnl:>9.4f}"
            )
        if df is not None and indicators:
            ei = T.legs[0].entry_idx
            ind_str = ", ".join(
                f"{name}[{ei}]={df[name].iloc[ei]:.4f}"
                for name in indicators
                if name in df.columns
            )
            if ind_str:
                print(f"       at ent={ei}: {ind_str}")


# Forward-looking. Currently unused but documents intent; flipping it
# does not change kernel behaviour. It will gate downstream analytics
# in items #28 (hedge engine) and beyond.
MULTI_LEG: bool = False
