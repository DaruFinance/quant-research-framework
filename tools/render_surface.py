#!/usr/bin/env python3
"""Item #1 renderer — interactive 3D iso-Sharpe isosurface (3-axis) OR lookback
robustness ridge (2-axis) + 2D heatmap slices + per-window robustness score for
the IN-SAMPLE objective landscape. Engine-agnostic; reads opt_surface.{csv,
parquet} (sibling of trade_list.csv).

EVERYTHING is labelled "in-sample objective landscape"; the engine's chosen pick
is paired with its ACTUAL OOS metric (from trade_list.csv OOS rows) for honest
contrast. The surface NEVER implies OOS performance.

Interactive HTML is self-contained (plotly.js inlined) so it opens offline.

Usage:
    python tools/render_surface.py [RUN_DIR]               # interactive HTML
    python tools/render_surface.py [RUN_DIR] --metric pf
    python tools/render_surface.py [RUN_DIR] --png         # + static PNG
    python tools/render_surface.py [RUN_DIR] --tol 0.10    # plateau within 10% of peak
    python tools/render_surface.py [RUN_DIR] --cdn         # link plotly.js via CDN
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

METRIC_COLS = ("sharpe", "pf", "roi", "mdd")


def discover_surface(run_dir: Path) -> Path:
    for name in ("opt_surface.parquet", "opt_surface.csv"):
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No opt_surface.{{parquet,csv}} in {run_dir}. Re-run the engine with "
        f"EMIT_OPT_SURFACE=1 (and EMIT_OPT_SURFACE_SL=1 for the 3-axis volume).")


def load_surface(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["window_idx"] = df["window_idx"].astype(str)
    df["regime"] = df["regime"].fillna("").astype(str)
    for c in ("roi", "pf", "sharpe", "mdd", "sl"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_oos_pick(run_dir: Path) -> Optional[dict]:
    """Pair the IS landscape with realised OOS from trade_list.csv (cols
    'sample','pnl' — verified against both engines). Loud skip if absent."""
    tl = run_dir / "trade_list.csv"
    if not tl.exists():
        return None
    try:
        t = pd.read_csv(tl)
    except Exception as e:
        sys.stderr.write(f"trade_list.csv unreadable ({e}); OOS pairing skipped\n")
        return None
    if "sample" not in t.columns or "pnl" not in t.columns:
        sys.stderr.write(f"trade_list.csv columns {list(t.columns)} lack "
                         f"'sample'/'pnl'; OOS pairing skipped\n")
        return None
    oos = t[t["sample"].astype(str).str.startswith("OOS")]
    if oos.empty:
        return None
    pnl = oos["pnl"].to_numpy(dtype=float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {"oos_trades": int(len(oos)), "oos_total_pnl": float(pnl.sum()),
            "oos_pf": float(wins / losses) if losses > 0 else float("inf"),
            "oos_win_rate": float((pnl > 0).mean())}


def robustness_score(g: pd.DataFrame, metric: str, tol: float) -> dict:
    """Per-(window, regime) robustness on the IS landscape.

    plateau_frac   = fraction of cells within `tol` of the peak (high => broad
                     plateau => robust). Only meaningful for a positive peak; a
                     plateau over an all-negative landscape is "uniformly bad",
                     not robust, so we flag negative-peak windows.
    peak_sharpness = PF(peak cell) / min(PF of lb-neighbours). This APPROXIMATES
                     (does not replay) SMART_OPTIMIZATION's `pf_cand > 1.10 *
                     pf_neigh for EITHER neighbour` guard — `> either` == `>
                     1.10 * min(neighbours)`. >1.10 flags a fragile needle.
                     Computed on PF (the engine's spike test is on PF regardless
                     of OPT_METRIC) at the PF-argmax cell.
    """
    vals = g[metric].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return {"peak": float("nan"), "plateau_frac": float("nan"),
                "peak_sharpness": float("nan"), "neg_peak": False,
                "n_cells": int(len(g))}
    if metric == "mdd":
        peak = float(np.nanmin(finite))           # least drawdown = best
        plateau = float(np.mean(finite <= peak * (1.0 + tol)))
        neg_peak = False
    else:
        peak = float(np.nanmax(finite))
        thresh = peak * (1.0 - tol) if peak >= 0 else peak * (1.0 + tol)
        plateau = float(np.mean(finite >= thresh))
        neg_peak = peak <= 0

    pf_ratio = float("nan")
    try:
        gi = g.reset_index(drop=True)
        idx = gi["pf"].idxmax()                    # PF-argmax, engine's spike basis
        prow = gi.loc[idx]
        slab = gi[np.isclose(gi["sl"], prow["sl"])].sort_values("lb")
        pf_by_lb = slab.set_index("lb")["pf"]
        peak_lb = int(prow["lb"])
        neigh = [pf_by_lb.get(peak_lb - 1, np.nan), pf_by_lb.get(peak_lb + 1, np.nan)]
        neigh = [x for x in neigh if np.isfinite(x) and x != 0]
        if neigh:
            pf_ratio = float(pf_by_lb.get(peak_lb, np.nan) / min(neigh))
    except Exception:
        pass
    return {"peak": peak, "plateau_frac": plateau, "peak_sharpness": pf_ratio,
            "neg_peak": neg_peak, "n_cells": int(len(g))}


def _colorscale(metric: str) -> str:
    return "Viridis_r" if metric == "mdd" else "Viridis"   # mdd: lower is better


def _fig_ridge_2axis(g: pd.DataFrame, metric: str, title: str):
    """2-axis: rrr is a function of lb, so render metric-vs-lb as a ridge with
    the chosen rrr encoded as marker colour (NOT a fake sparse surface)."""
    import plotly.graph_objects as go
    gg = g.sort_values("lb")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gg["lb"], y=gg[metric], mode="lines+markers",
        marker=dict(color=gg["rrr"], colorscale="Plasma", showscale=True,
                    colorbar=dict(title="chosen RRR")),
        line=dict(color="rgba(120,120,120,0.5)"),
        hovertext=[f"lb={l}, rrr={r}" for l, r in zip(gg["lb"], gg["rrr"])]))
    fig.update_layout(title=title, xaxis_title="lookback (lb)",
                      yaxis_title=f"IS {metric}", margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _fig_isosurface_3axis(g: pd.DataFrame, metric: str, title: str):
    import plotly.graph_objects as go
    val = g[metric].to_numpy(dtype=float)
    vmin = float(np.nanpercentile(val, 5))
    vmax = float(np.nanmax(val))
    fig = go.Figure(data=go.Isosurface(
        x=g["lb"].to_numpy(float), y=g["rrr"].to_numpy(float),
        z=g["sl"].to_numpy(float), value=val,
        isomin=vmin, isomax=vmax, surface_count=4, opacity=0.5,
        colorscale=_colorscale(metric),
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title=f"IS {metric}")))
    steps = []
    for frac in np.linspace(0.0, 0.95, 12):
        lvl = vmin + frac * (vmax - vmin)
        steps.append(dict(method="restyle", args=[{"isomin": lvl}],
                          label=f"{lvl:.2f}"))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="lookback (lb)", yaxis_title="RRR", zaxis_title="SL %"),
        sliders=[dict(active=0, currentvalue={"prefix": "iso-level: "}, steps=steps)],
        margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _fig_heatmaps(g: pd.DataFrame, metric: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    sls = sorted(g["sl"].unique())
    fig = make_subplots(rows=1, cols=len(sls),
                        subplot_titles=[f"SL={s:.3g}" for s in sls])
    for j, s in enumerate(sls, start=1):
        piv = g[np.isclose(g["sl"], s)].pivot_table(
            index="rrr", columns="lb", values=metric, aggfunc="mean")
        fig.add_trace(go.Heatmap(z=piv.values, x=list(piv.columns), y=list(piv.index),
                                 colorscale=_colorscale(metric),
                                 showscale=(j == len(sls)),
                                 colorbar=dict(title=metric)), row=1, col=j)
        fig.update_xaxes(title_text="lb", row=1, col=j)
        if j == 1:
            fig.update_yaxes(title_text="RRR", row=1, col=j)
    fig.update_layout(
        title=f"IS objective landscape — {metric} heatmap slices (in-sample)",
        margin=dict(l=0, r=0, t=60, b=0))
    return fig


def render_png(g: pd.DataFrame, metric: str, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sls = sorted(g["sl"].unique())
    fig, axes = plt.subplots(1, len(sls), figsize=(4 * len(sls), 4), squeeze=False)
    cmap = "viridis_r" if metric == "mdd" else "viridis"
    for j, s in enumerate(sls):
        piv = g[np.isclose(g["sl"], s)].pivot_table(
            index="rrr", columns="lb", values=metric, aggfunc="mean")
        ax = axes[0][j]
        im = ax.imshow(piv.values, aspect="auto", origin="lower",
                       extent=[piv.columns.min(), piv.columns.max(),
                               piv.index.min(), piv.index.max()], cmap=cmap)
        ax.set_title(f"SL={s:.3g}")
        ax.set_xlabel("lookback (lb)")
        if j == 0:
            ax.set_ylabel("RRR")
        fig.colorbar(im, ax=ax, label=f"IS {metric}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=".", type=Path)
    ap.add_argument("--metric", default="sharpe", choices=METRIC_COLS)
    ap.add_argument("--tol", type=float, default=0.10,
                    help="plateau tolerance: cells within this fraction of peak")
    ap.add_argument("--window", default=None, help="restrict to one window_idx")
    ap.add_argument("--png", action="store_true", help="also emit static PNG")
    ap.add_argument("--cdn", action="store_true",
                    help="link plotly.js via CDN (smaller HTML, needs network)")
    ap.add_argument("--out", default=None, help="output HTML path")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    df = load_surface(discover_surface(run_dir))
    if args.window is not None:
        df = df[df["window_idx"] == str(args.window)]
    if df.empty:
        sys.stderr.write("Surface is empty after filtering.\n")
        return 2

    three_axis = df["sl"].nunique() > 1
    smode = df["sharpe_mode"].iloc[0] if "sharpe_mode" in df.columns else "trade"
    oos = load_oos_pick(run_dir)

    print(f"\nIN-SAMPLE OBJECTIVE LANDSCAPE — robustness diagnostics "
          f"(metric={args.metric}, sharpe_mode={smode}, plateau tol={args.tol:.0%})")
    print(f"{'window':>8} {'regime':>10} {'peak':>10} {'plateau%':>9} "
          f"{'pf/min-neigh':>12} {'cells':>6}")
    for (w, reg), gg in df.groupby(["window_idx", "regime"]):
        s = robustness_score(gg, args.metric, args.tol)
        flags = ""
        if np.isfinite(s["peak_sharpness"]) and s["peak_sharpness"] > 1.10:
            flags += " (needle!)"
        if s.get("neg_peak"):
            flags += " (neg-peak: plateau!=robust)"
        print(f"{w:>8} {reg or '-':>10} {s['peak']:>10.3f} "
              f"{s['plateau_frac']*100:>8.1f}% {s['peak_sharpness']:>12.3f}"
              f"  ({s['n_cells']} cells){flags}")
    if oos:
        print(f"\nEngine pick ACTUAL OOS (from trade_list.csv): "
              f"trades={oos['oos_trades']} total_pnl={oos['oos_total_pnl']:.4g} "
              f"PF={oos['oos_pf']:.3f} win_rate={oos['oos_win_rate']:.1%}")
    else:
        print("\n(no trade_list.csv OOS rows found — OOS pairing skipped)")

    try:
        import plotly.graph_objects as go  # noqa: F401
        have_plotly = True
    except ImportError:
        sys.stderr.write("plotly not installed — emitting PNG only. For "
                         "interactive HTML: pip install "
                         "quant-research-framework[surface]\n")
        have_plotly = False
        args.png = True

    out_html = Path(args.out) if args.out else (run_dir / "opt_surface.html")
    kind = "3D iso-Sharpe isosurface" if three_axis else "2-axis lookback ridge"
    title = (f"IN-SAMPLE objective landscape — {kind} ({args.metric}, "
             f"sharpe={smode}) [NOT OOS performance]")

    if have_plotly:
        fig_main = (_fig_isosurface_3axis(df, args.metric, title) if three_axis
                    else _fig_ridge_2axis(df, args.metric, title))
        fig_hm = _fig_heatmaps(df, args.metric) if three_axis else None
        from plotly.io import to_html
        plotlyjs = "cdn" if args.cdn else True   # True => inline => offline-safe
        html = ["<html><head><meta charset='utf-8'></head><body>",
                f"<h2>{title}</h2>",
                to_html(fig_main, include_plotlyjs=plotlyjs, full_html=False)]
        if fig_hm is not None:
            html += ["<hr>", to_html(fig_hm, include_plotlyjs=False, full_html=False)]
        if oos:
            html.append(f"<p><b>Engine pick — ACTUAL OOS</b> (paired, not the "
                        f"surface): trades={oos['oos_trades']}, "
                        f"PF={oos['oos_pf']:.3f}, win_rate={oos['oos_win_rate']:.1%}, "
                        f"total_pnl={oos['oos_total_pnl']:.4g}</p>")
        html.append("</body></html>")
        out_html.write_text("".join(html))
        print(f"\nWrote interactive HTML: {out_html}")

    if args.png:
        out_png = out_html.with_suffix(".png")
        render_png(df, args.metric, out_png, title)
        print(f"Wrote static PNG: {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
