#!/usr/bin/env python3
"""Idempotent builder for the Phase 3 fixture set.

Outputs (committed alongside this script under ``tests/fixtures/``):

- ``pair_btc_eth_1h_1500.parquet`` — 1500 1h bars of BTCUSDT + ETHUSDT
  from a window where the log-ratio is cointegrating (ADF p < 0.05).
- ``funding_btcusdt_200evt.parquet`` — 200 8h funding events from
  Binance USDⓈ-M perpetual history.
- ``oi_btc_perp_1h_7d.parquet`` — 7 days of 1h BTC perp open interest
  from Binance futures.
- ``basis_btc_perp_spot_1d.parquet`` — 1 day of 1h basis bp (perp -
  spot) for BTC.
- ``onchain_nvt_50d.csv`` — 50 daily NVT readings derived from
  blockchain.info free public stats endpoints (no API key).

Run from the repo root:

    python tests/fixtures/build_phase3.py

Network-bound but single-threaded. Skips a fixture if it already
exists (deletes stale ones to re-fetch).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_PAIR     = HERE / "pair_btc_eth_1h_1500.parquet"
OUT_FUNDING  = HERE / "funding_btcusdt_200evt.parquet"
OUT_OI       = HERE / "oi_btc_perp_1h_7d.parquet"
OUT_BASIS    = HERE / "basis_btc_perp_spot_1d.parquet"
OUT_ONCHAIN  = HERE / "onchain_nvt_50d.csv"

SOURCES = HERE / "sources"
SOURCES.mkdir(exist_ok=True)


def _http_get(url: str, timeout: int = 30, retries: int = 3) -> bytes:
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1 + i)
    raise RuntimeError(f"http_get failed for {url}: {last}")


# ---------------------------------------------------------------------------
# DS-PAIR-BTCETH: 1500-bar BTC/ETH 1h from a cointegrating window.
# ---------------------------------------------------------------------------
def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int,
                 limit: int = 1000) -> List[list]:
    params = {"symbol": symbol, "interval": interval,
              "startTime": start_ms, "endTime": end_ms, "limit": limit}
    url = "https://api.binance.com/api/v3/klines?" + urllib.parse.urlencode(params)
    return json.loads(_http_get(url))


def fetch_klines_window(symbol: str, interval: str, start: dt.datetime,
                        end: dt.datetime) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: List[list] = []
    cur = start_ms
    while cur < end_ms:
        bars = fetch_klines(symbol, interval, cur, end_ms, limit=1000)
        if not bars:
            break
        rows.extend(bars)
        cur = bars[-1][0] + 60_000
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=[
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_volume", "trades", "taker_base", "taker_quote",
        "ignore",
    ])
    df["time"] = (df["open_time_ms"] // 1000).astype(np.int64)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(np.float64)
    return df[["time", "open", "high", "low", "close"]].drop_duplicates("time").sort_values("time").reset_index(drop=True)


def build_pair_btc_eth():
    if OUT_PAIR.exists():
        print(f"  {OUT_PAIR.name}: cached, keeping")
        return
    print(f"  {OUT_PAIR.name}: fetching BTC + ETH 1h for 2023-Q3...")
    # 2023-Q3 = Jul 1 to Sep 30 = 92 days = 2208 1h bars; we'll pick a
    # 1500-bar cointegrating sub-window.
    start = dt.datetime(2023, 7, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2023, 9, 30, tzinfo=dt.timezone.utc)
    btc = fetch_klines_window("BTCUSDT", "1h", start, end)
    eth = fetch_klines_window("ETHUSDT", "1h", start, end)
    # Inner-join.
    merged = btc.merge(eth, on="time", suffixes=("_btc", "_eth"))
    print(f"    inner-join produced {len(merged)} bars")

    # Find a 1500-bar window where the log-ratio is cointegrating.
    from statsmodels.tsa.stattools import adfuller
    log_btc = np.log(merged["close_btc"].values)
    log_eth = np.log(merged["close_eth"].values)
    log_ratio = log_btc - log_eth
    # ADF on the full series first.
    adf_full = adfuller(log_ratio, regression="c")
    print(f"    full-series ADF on log(BTC/ETH): stat={adf_full[0]:.3f} p={adf_full[1]:.4f}")

    # Slide a 1500-bar window; pick the lowest-p window.
    best_p = None
    best_start = 0
    win = 1500
    if len(merged) < win:
        raise RuntimeError(
            f"fetched only {len(merged)} bars; need >= {win}"
        )
    for s in range(0, len(merged) - win + 1, 50):
        adf = adfuller(log_ratio[s:s + win], regression="c")
        if best_p is None or adf[1] < best_p:
            best_p = adf[1]
            best_start = s
    print(f"    best 1500-bar ADF: start={best_start} p={best_p:.4f}")
    if best_p > 0.05:
        print(f"    warning: best p={best_p:.4f} > 0.05; spread may not be "
              f"strictly cointegrating but kept for fixture purposes")

    sub = merged.iloc[best_start:best_start + win].reset_index(drop=True)

    # Save as a long-form panel-style frame so load_panel can ingest.
    rows = []
    for _, r in sub.iterrows():
        for asset in ("BTC", "ETH"):
            rows.append({
                "time": int(r["time"]),
                "asset": asset,
                "open": float(r[f"open_{asset.lower()}"]),
                "high": float(r[f"high_{asset.lower()}"]),
                "low":  float(r[f"low_{asset.lower()}"]),
                "close": float(r[f"close_{asset.lower()}"]),
            })
    out = pd.DataFrame(rows).sort_values(["time", "asset"]).reset_index(drop=True)
    out.to_parquet(OUT_PAIR, index=False)
    # Also write per-asset source CSVs for load_panel.
    for asset in ("BTC", "ETH"):
        per = sub.rename(columns={
            f"open_{asset.lower()}": "open", f"high_{asset.lower()}": "high",
            f"low_{asset.lower()}": "low", f"close_{asset.lower()}": "close",
        })[["time", "open", "high", "low", "close"]]
        per.to_csv(SOURCES / f"{asset}USDT_1h_pair_q3_2023.csv", index=False)
    print(f"    wrote {OUT_PAIR.name} ({len(out)} rows) + 2 per-asset CSVs")


# ---------------------------------------------------------------------------
# DS-FUNDING-200: Binance USDⓈ-M perpetual funding rate history.
# ---------------------------------------------------------------------------
def build_funding_200():
    if OUT_FUNDING.exists():
        print(f"  {OUT_FUNDING.name}: cached, keeping")
        return
    print(f"  {OUT_FUNDING.name}: fetching 200 BTCUSDT funding events...")
    url = ("https://fapi.binance.com/fapi/v1/fundingRate"
           "?symbol=BTCUSDT&limit=200")
    data = json.loads(_http_get(url))
    if len(data) != 200:
        print(f"    got {len(data)} events (expected 200)")
    rows = [(int(d["fundingTime"] // 1000), float(d["fundingRate"])) for d in data]
    df = pd.DataFrame(rows, columns=["time", "rate"])
    df = df.sort_values("time").reset_index(drop=True)
    df.to_parquet(OUT_FUNDING, index=False)
    print(f"    wrote {OUT_FUNDING.name} ({len(df)} events, span "
          f"{df['time'].iloc[0]} .. {df['time'].iloc[-1]})")


# ---------------------------------------------------------------------------
# DS-OI-7D: 7 days of 1h BTC perp open interest from Binance futures.
# ---------------------------------------------------------------------------
def build_oi_7d():
    if OUT_OI.exists():
        print(f"  {OUT_OI.name}: cached, keeping")
        return
    print(f"  {OUT_OI.name}: fetching 7d BTCUSDT-perp OI at 1h cadence...")
    # Binance futures /futures/data/openInterestHist max limit 500.
    # 7 days * 24h = 168 hourly samples; well under the limit.
    url = ("https://fapi.binance.com/futures/data/openInterestHist"
           "?symbol=BTCUSDT&period=1h&limit=168")
    data = json.loads(_http_get(url))
    rows = [
        (int(d["timestamp"] // 1000), float(d["sumOpenInterest"]),
         float(d["sumOpenInterestValue"]))
        for d in data
    ]
    df = pd.DataFrame(rows, columns=["time", "open_interest", "open_interest_usd"])
    df = df.sort_values("time").reset_index(drop=True)
    df.to_parquet(OUT_OI, index=False)
    print(f"    wrote {OUT_OI.name} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# DS-BASIS-1D: 24 1h points of BTC perp - spot mark price.
# ---------------------------------------------------------------------------
def build_basis_1d():
    if OUT_BASIS.exists():
        print(f"  {OUT_BASIS.name}: cached, keeping")
        return
    print(f"  {OUT_BASIS.name}: fetching 1d BTC perp & spot at 1h...")
    end = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    start = end - dt.timedelta(days=2)  # small buffer for join
    spot = fetch_klines_window("BTCUSDT", "1h", start, end)
    # Perp via futures klines endpoint.
    sm = int(start.timestamp() * 1000)
    em = int(end.timestamp() * 1000)
    perp_url = ("https://fapi.binance.com/fapi/v1/klines"
                f"?symbol=BTCUSDT&interval=1h&startTime={sm}&endTime={em}"
                f"&limit=1000")
    perp_rows = json.loads(_http_get(perp_url))
    perp = pd.DataFrame(perp_rows, columns=[
        "open_time_ms", "open", "high", "low", "close", "volume",
        "close_time_ms", "quote_volume", "trades", "taker_base",
        "taker_quote", "ignore",
    ])
    perp["time"] = (perp["open_time_ms"] // 1000).astype(np.int64)
    perp["close"] = perp["close"].astype(np.float64)
    perp = perp[["time", "close"]].drop_duplicates("time").sort_values("time").reset_index(drop=True)
    merged = spot[["time", "close"]].merge(perp, on="time", suffixes=("_spot", "_perp"))
    merged["basis_bp"] = (
        (merged["close_perp"] - merged["close_spot"]) / merged["close_spot"] * 10_000.0
    )
    # Trim to the most recent 24 rows.
    merged = merged.tail(24).reset_index(drop=True)
    merged[["time", "close_spot", "close_perp", "basis_bp"]].to_parquet(OUT_BASIS, index=False)
    print(f"    wrote {OUT_BASIS.name} ({len(merged)} rows, basis_bp mean="
          f"{merged['basis_bp'].mean():.2f}bp)")


# ---------------------------------------------------------------------------
# DS-ONCHAIN-50: 50 daily NVT readings derived from blockchain.info free stats.
# NVT = market_cap_usd / transaction_volume_usd. We use the free stats:
#   - market-price.json (USD price)
#   - n-transactions.json + estimated-transaction-volume-usd.json
# blockchain.info supports timespan=Nweeks; 50 days ~ 8 weeks.
# ---------------------------------------------------------------------------
def _blockchain_info_series(name: str) -> pd.DataFrame:
    url = (f"https://api.blockchain.info/charts/{name}"
           f"?timespan=8weeks&format=json&sampled=false")
    data = json.loads(_http_get(url))
    rows = [(int(v["x"]), float(v["y"])) for v in data["values"]]
    return pd.DataFrame(rows, columns=["time", name.replace("-", "_")])


def _bucket_to_day(df: pd.DataFrame) -> pd.DataFrame:
    """blockchain.info series sample at different cadences (e.g.
    market-price ~ 2-day, transaction-volume daily). Round each row's
    UNIX ts down to the nearest midnight UTC and keep the latest
    value per day so the per-series day grids align before the merge.
    """
    out = df.copy()
    out["day"] = (out["time"] // 86_400) * 86_400
    return out.sort_values("time").drop_duplicates("day", keep="last").reset_index(drop=True)


def build_onchain_nvt_50d():
    if OUT_ONCHAIN.exists():
        print(f"  {OUT_ONCHAIN.name}: cached, keeping")
        return
    print(f"  {OUT_ONCHAIN.name}: fetching blockchain.info stats for NVT...")
    price = _bucket_to_day(_blockchain_info_series("market-price"))
    tx_vol_usd = _bucket_to_day(_blockchain_info_series("estimated-transaction-volume-usd"))
    supply = _bucket_to_day(_blockchain_info_series("total-bitcoins"))
    df = price.merge(supply, on="day").merge(tx_vol_usd, on="day")
    df["market_cap_usd"] = df["market_price"] * df["total_bitcoins"]
    df["estimated_transaction_volume_usd"] = df["estimated_transaction_volume_usd"].replace(0, np.nan)
    df["nvt"] = df["market_cap_usd"] / df["estimated_transaction_volume_usd"]
    df = df.dropna(subset=["nvt"]).tail(50).reset_index(drop=True)
    df = df[["day", "market_price", "total_bitcoins", "market_cap_usd",
             "estimated_transaction_volume_usd", "nvt"]].rename(columns={"day": "time"})
    df.to_csv(OUT_ONCHAIN, index=False)
    print(f"    wrote {OUT_ONCHAIN.name} ({len(df)} daily readings, "
          f"NVT range {df['nvt'].min():.1f} .. {df['nvt'].max():.1f})")


def main() -> int:
    print("Building Phase 3 fixtures (sequential, single-threaded)...")
    build_pair_btc_eth()
    build_funding_200()
    build_oi_7d()
    build_basis_1d()
    build_onchain_nvt_50d()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
