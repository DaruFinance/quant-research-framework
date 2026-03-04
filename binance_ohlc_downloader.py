#!/usr/bin/env python3
"""
Binance OHLC Downloader — full-history klines via API and/or Binance Vision archive

What it does
------------
- Downloads OHLCV (klines) for SPOT or FUTURES (USDⓈ-M / COIN-M) pairs.
- Lets you choose the symbol, interval, market, and source (API, archive, or both).
- Gets the ENTIRE available history for your inputs, writing a single CSV (or Parquet).
- Resumable: if you pass --resume and an output file exists, it will continue from last bar.
- Safe ordering: archive mode writes in chronological order; API mode appends strictly forward.

Examples
--------
# Auto: use Binance Vision monthly/daily archives, then top up via API to current minute
python binance_ohlc_downloader.py \
  --symbol BTCUSDT --interval 1m --market spot --source auto \
  --since 2017-01-01 --until now --out data/BTCUSDT_1m.csv

# API only: SPOT ETH 5m since 2020
python binance_ohlc_downloader.py \
  --symbol ETHUSDT --interval 5m --market spot --source api \
  --since 2020-01-01 --out data/ETHUSDT_5m.csv --rpm 900

# FUTURES (USDⓈ-M) SOL 1h, archive only
python binance_ohlc_downloader.py \
  --symbol SOLUSDT --interval 1h --market um --source archive \
  --since 2020-01-01 --until now --out data/SOLUSDT_1h.csv

# Resume an existing file (continues from last open_time)
python binance_ohlc_downloader.py --symbol BTCUSDT --interval 1m \
  --market spot --source api --out data/BTCUSDT_1m.csv --resume

Notes
-----
- No API key is needed for klines. If you add one, it won’t harm, but it’s unused here.
- For Parquet output, install pyarrow: `pip install pyarrow` (optional).
- Intervals supported: 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
- Markets: spot | um (USDⓈ-M futures) | cm (COIN‑M futures)
"""

import argparse
import csv
import datetime as dt
import io
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import requests

# --------------------------- Helpers & Constants --------------------------- #

BINANCE_API_BASE = {
    "spot": "https://api.binance.com",
    "um": "https://fapi.binance.com",   # USDⓈ-M Futures
    "cm": "https://dapi.binance.com",   # COIN-M Futures
}

VISION_BASE = {
    "spot": "https://data.binance.vision/data/spot",
    "um": "https://data.binance.vision/data/futures/um",
    "cm": "https://data.binance.vision/data/futures/cm",
}

VALID_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"
}

@dataclass
class Args:
    symbol: str
    interval: str
    market: str  # spot|um|cm
    source: str  # api|archive|auto
    since: str
    until: str
    out: str
    fmt: str
    resume: bool
    rpm: int
    retries: int
    timeout: int


def ts_ms_now() -> int:
    return int(time.time() * 1000)


def parse_time_ms(s: str) -> int:
    """Parse a time string into milliseconds since epoch (UTC).
    Accepts: 'now', 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM', 'YYYY-MM-DDTHH:MM:SS'.
    """
    s = s.strip().lower()
    if s == "now":
        return ts_ms_now()
    # Be generous with formats
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        except ValueError:
            pass
    raise ValueError(f"Could not parse time: {s}")


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def ms_to_iso(ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000.0).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------- File Writing Utilities ----------------------- #

CSV_HEADER = ["time", "open", "high", "low", "close"]


def init_csv(path: str) -> None:
    ensure_parent_dir(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def last_open_time_from_csv(path: str) -> Optional[int]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    # read last non-header line efficiently
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        if end == 0:
            return None
        # scan backwards for last newline
        size = min(4096, end)
        while size <= end:
            f.seek(-size, os.SEEK_END)
            chunk = f.read(size)
            if b"\n" in chunk:
                last_line = chunk.splitlines()[-1]
                break
            if size == end:
                last_line = chunk
                break
            size = min(size * 2, end)
        else:
            return None
    try:
        row = last_line.decode("utf-8").split(",")
        if row and row[0].isdigit():
            return int(row[0])
    except Exception:
        return None
    return None


def write_kline_rows(path: str, rows: Iterable[List]) -> int:
    """Append rows (raw Binance API row lists), adding ISO columns. Returns count."""
    count = 0
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            if not r:
                continue
            # r = [ open_time, open, high, low, close, volume, close_time,
            #       quote_asset_volume, number_of_trades, taker_buy_base,
            #       taker_buy_quote, ignore ]
            try:
                ot = int(r[0])
                ct = int(r[6])
                w.writerow([
                    int(r[0]) // 1000,  # convert ms to seconds
                    r[1], r[2], r[3], r[4]
                ])
                count += 1
            except Exception as e:
                print(f"[WARN] Skipping malformed row: {e} | {r}")
    return count


# --------------------------- API Downloader ------------------------------- #

class ApiDownloader:
    def __init__(self, market: str, interval: str, rpm: int, retries: int, timeout: int):
        self.market = market
        self.interval = interval
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ohlc-downloader/1.0"})
        self.base = BINANCE_API_BASE[market]
        self.rpm = max(60, rpm)  # cap sane minimum
        self.delay = 60.0 / float(self.rpm)
        self.retries = max(1, retries)
        self.timeout = timeout

    def _endpoint(self) -> str:
        return f"{self.base}/api/v3/klines" if self.market == "spot" else f"{self.base}/fapi/v1/klines" if self.market == "um" else f"{self.base}/dapi/v1/klines"

    def fetch_range(self, symbol: str, start_ms: int, end_ms: int, out_path: str, resume_from_last: bool = False) -> int:
        url = self._endpoint()
        total = 0
        start = start_ms
        # If resuming, jump to last_open_time + 1ms
        if resume_from_last:
            last = last_open_time_from_csv(out_path)
            if last is not None and last >= start:
                start = last + 1
                print(f"[API] Resuming from {ms_to_iso(start)}")
        print(f"[API] Downloading {symbol} {self.interval} from {ms_to_iso(start)} to {ms_to_iso(end_ms)}")
        while True:
            params = {
                "symbol": symbol.upper(),
                "interval": self.interval,
                "startTime": start,
                "endTime": end_ms,
                "limit": 1000,
            }
            data = self._get_with_retries(url, params)
            if not data:
                break
            wrote = write_kline_rows(out_path, data)
            total += wrote
            last_open = int(data[-1][0])
            start = last_open + 1  # avoid overlap regardless of interval size
            if start > end_ms:
                break
            time.sleep(self.delay)
        print(f"[API] Wrote {total} rows")
        return total

    def _get_with_retries(self, url: str, params: dict):
        tries = 0
        while True:
            tries += 1
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (418, 429):  # rate limited
                    retry_after = int(r.headers.get("Retry-After", "1"))
                    sleep_s = max(1, retry_after)
                    print(f"[API] Rate limited, sleeping {sleep_s}s…")
                    time.sleep(sleep_s)
                    continue
                if 500 <= r.status_code < 600:
                    print(f"[API] Server error {r.status_code}, retrying…")
                else:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                print(f"[API] Error: {e} (try {tries}/{self.retries})")
            if tries >= self.retries:
                raise RuntimeError("Too many API errors; aborting.")
            # backoff
            time.sleep(min(5 * tries, 30))


# --------------------------- Archive Downloader --------------------------- #

class ArchiveDownloader:
    def __init__(self, market: str, interval: str, retries: int, timeout: int):
        self.market = market
        self.interval = interval
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ohlc-downloader/1.0"})
        self.vision_base = VISION_BASE[market]
        self.retries = max(1, retries)
        self.timeout = timeout

    def monthly_url(self, symbol: str, year: int, month: int) -> str:
        mm = f"{month:02d}"
        return f"{self.vision_base}/monthly/klines/{symbol.upper()}/{self.interval}/{symbol.upper()}-{self.interval}-{year}-{mm}.zip"

    def daily_url(self, symbol: str, year: int, month: int, day: int) -> str:
        mm = f"{month:02d}"
        dd = f"{day:02d}"
        return f"{self.vision_base}/daily/klines/{symbol.upper()}/{self.interval}/{symbol.upper()}-{self.interval}-{year}-{mm}-{dd}.zip"

    def _get_bytes(self, url: str) -> Optional[bytes]:
        tries = 0
        while True:
            tries += 1
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    return r.content
                if r.status_code == 404:
                    return None
                if 500 <= r.status_code < 600:
                    print(f"[ARCHIVE] Server {r.status_code} for {url}, retrying…")
                else:
                    print(f"[ARCHIVE] HTTP {r.status_code} for {url}, skipping.")
                    return None
            except Exception as e:
                print(f"[ARCHIVE] Error fetching {url}: {e} (try {tries}/{self.retries})")
            if tries >= self.retries:
                print("[ARCHIVE] Too many errors; skipping file.")
                return None
            time.sleep(min(5 * tries, 30))

    def _write_zip_csv_rows(self, zbytes: bytes, out_path: str, start_ms: int, end_ms: int) -> int:
        total = 0
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith('.csv'):
                    continue
                with zf.open(name) as f:
                    reader = csv.reader(io.TextIOWrapper(f, encoding='utf-8'))
                    for r in reader:
                        if not r:
                            continue
                        try:
                            ot = int(r[0])
                        except Exception:
                            continue
                        if ot < start_ms or ot > end_ms:
                            continue
                        total += write_kline_rows(out_path, [r])
        return total

    def fetch(self, symbol: str, start_ms: int, end_ms: int, out_path: str) -> int:
        """Download monthly zips in chronological order.
        If a monthly zip is missing for a month that's within [start, end],
        and that month intersects with the end-month (often current/partial),
        we try daily files for that month.
        """
        start_dt = dt.datetime.utcfromtimestamp(start_ms / 1000.0)
        end_dt = dt.datetime.utcfromtimestamp(end_ms / 1000.0)
        year, month = start_dt.year, start_dt.month
        end_year, end_month = end_dt.year, end_dt.month

        total = 0
        while (year < end_year) or (year == end_year and month <= end_month):
            # Compute month range in ms
            first_day = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
            if month == 12:
                next_month = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)
            else:
                next_month = dt.datetime(year, month + 1, 1, tzinfo=dt.timezone.utc)
            month_start_ms = max(start_ms, int(first_day.timestamp() * 1000))
            month_end_ms = min(end_ms, int(next_month.timestamp() * 1000) - 1)

            m_url = self.monthly_url(symbol, year, month)
            z = self._get_bytes(m_url)
            if z:
                print(f"[ARCHIVE] {symbol} {self.interval} {year}-{month:02d}: using monthly")
                total += self._write_zip_csv_rows(z, out_path, month_start_ms, month_end_ms)
            else:
                # fall back to daily only if the month intersects with end month (often current month)
                # or if explicitly earlier but monthly missing — try dailies anyway
                print(f"[ARCHIVE] {symbol} {self.interval} {year}-{month:02d}: monthly not found, trying daily")
                day = 1
                while True:
                    day_dt = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=day - 1)
                    if day_dt.month != month:
                        break
                    day_start_ms = max(start_ms, int(day_dt.timestamp() * 1000))
                    day_end_ms = min(end_ms, int((day_dt + dt.timedelta(days=1)).timestamp() * 1000) - 1)
                    if day_start_ms > day_end_ms:
                        day += 1
                        continue
                    d_url = self.daily_url(symbol, year, month, day)
                    dz = self._get_bytes(d_url)
                    if dz:
                        print(f"[ARCHIVE] {symbol} {self.interval} {year}-{month:02d}-{day:02d}: daily OK")
                        total += self._write_zip_csv_rows(dz, out_path, day_start_ms, day_end_ms)
                    else:
                        # Missing day is OK; might not exist for new listings
                        pass
                    day += 1

            # next month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
        print(f"[ARCHIVE] Wrote {total} rows")
        return total


# --------------------------- Parquet (optional) --------------------------- #

def maybe_convert_to_parquet(csv_path: str) -> Optional[str]:
    try:
        import pandas as pd
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception:
        print("[PARQUET] Skipping: install pandas+pyarrow for Parquet output.")
        return None
    df = pd.read_csv(csv_path)
    pq_path = os.path.splitext(csv_path)[0] + ".parquet"
    df.to_parquet(pq_path, index=False)
    print(f"[PARQUET] Wrote {pq_path}")
    return pq_path


# --------------------------- Orchestration --------------------------- #

def run(args: Args) -> None:
    if args.interval not in VALID_INTERVALS:
        raise SystemExit(f"Interval {args.interval} not in supported set: {sorted(VALID_INTERVALS)}")
    if args.market not in ("spot", "um", "cm"):
        raise SystemExit("market must be one of: spot|um|cm")
    if args.source not in ("api", "archive", "auto"):
        raise SystemExit("source must be one of: api|archive|auto")

    # Compute time bounds
    start_ms = parse_time_ms(args.since)
    end_ms = parse_time_ms(args.until)
    if start_ms >= end_ms:
        raise SystemExit("since must be < until")

    init_csv(args.out)

    # If resume without archive step, adjust start time from file
    if args.resume and args.source in ("api",):
        last = last_open_time_from_csv(args.out)
        if last is not None and last + 1 > start_ms:
            start_ms = last + 1
            print(f"[RESUME] Adjusted since to {ms_to_iso(start_ms)} based on {args.out}")

    total_rows = 0

    if args.source in ("archive", "auto"):
        arch = ArchiveDownloader(args.market, args.interval, args.retries, args.timeout)
        total_rows += arch.fetch(args.symbol, start_ms, end_ms, args.out)

    if args.source in ("api", "auto"):
        api = ApiDownloader(args.market, args.interval, args.rpm, args.retries, args.timeout)
        # For API topping, start where file ends (even if not resume flag) to avoid dups
        last = last_open_time_from_csv(args.out)
        api_start = max(start_ms, (last + 1) if last is not None else start_ms)
        total_rows += api.fetch_range(args.symbol, api_start, end_ms, args.out, resume_from_last=True)

    print(f"[DONE] Total rows written: {total_rows}")

    if args.fmt.lower() == "parquet":
        maybe_convert_to_parquet(args.out)


# --------------------------- CLI ----------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Download full-history OHLCV (klines) from Binance API and/or Binance Vision archive.")
    p.add_argument("--symbol", required=True, help="Trading pair symbol, e.g., BTCUSDT, ETHUSDT")
    p.add_argument("--interval", required=True, help="Kline interval (e.g., 1m, 5m, 1h, 1d, 1w, 1M)")
    p.add_argument("--market", default="spot", help="spot | um (USDⓈ-M futures) | cm (COIN-M futures)")
    p.add_argument("--source", default="auto", help="api | archive | auto (archive then API top-up)")
    p.add_argument("--since", default="2017-01-01", help="Start time: 'YYYY-MM-DD[THH:MM[:SS]]' or 'now'")
    p.add_argument("--until", default="now", help="End time: 'YYYY-MM-DD[THH:MM[:SS]]' or 'now'")
    p.add_argument("--out", default="./output.csv", help="Output CSV path")
    p.add_argument("--fmt", default="csv", help="csv | parquet (converts CSV to Parquet if pyarrow available)")
    p.add_argument("--resume", action="store_true", help="Resume from existing CSV's last open_time when using --source api")
    p.add_argument("--rpm", type=int, default=900, help="Requests per minute throttle for API (default 900)")
    p.add_argument("--retries", type=int, default=5, help="HTTP retry attempts per request (default 5)")
    p.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds (default 30)")

    ns = p.parse_args()
    a = Args(
        symbol=ns.symbol,
        interval=ns.interval,
        market=ns.market.lower(),
        source=ns.source.lower(),
        since=ns.since,
        until=ns.until,
        out=ns.out,
        fmt=ns.fmt,
        resume=ns.resume,
        rpm=ns.rpm,
        retries=ns.retries,
        timeout=ns.timeout,
    )
    try:
        run(a)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
