# Data Folder

This folder is used to store OHLC market data for the backtester.

CSV files are not included in this repository.

Expected format:

timestamp,open,high,low,close

- timestamp: UNIX seconds
- open/high/low/close: numeric price values

You can generate data using the included downloader:

python binance_ohlc_downloader.py --symbol DOGEUSDT --interval 30m --market spot --source api --since 2017-11-01 --until now --out data/DOGEUSDT_30m.csv
