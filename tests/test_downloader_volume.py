"""Downloader column fidelity and resume checks without network calls."""
import csv
import io
import zipfile

import pytest

import binance_ohlc_downloader as dl


# First SOLUSDT hourly candle from Binance spot, 2020-08-11 06:00 UTC.
ROW = [1597125600000, "2.85000000", "3.47000000", "2.85000000",
       "2.95150000", "20032.26000000", 1597129199999]


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.reader(f))


@pytest.mark.parametrize("volume", [False, True])
@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_api_keeps_requested_columns_and_seconds(tmp_path, monkeypatch, volume, unit):
    path = tmp_path / "api.csv"
    dl.init_csv(str(path), volume)
    row = ROW.copy()
    row[0] = ROW[0] // 1000 if unit == "s" else ROW[0] * (1000 if unit == "us" else 1)
    api = dl.ApiDownloader("spot", "1h", 900, 1, 1, volume=volume)
    monkeypatch.setattr(api, "_get_with_retries", lambda *args: [row])
    assert api.fetch_range("SOLUSDT", ROW[0], ROW[0], str(path)) == 1
    assert read_rows(path) == [dl.CSV_HEADER + (["volume"] if volume else []),
                               [str(ROW[0] // 1000)] + ROW[1:6 if volume else 5]]
    assert dl.last_open_time_from_csv(str(path)) == ROW[0]


@pytest.mark.parametrize("volume", [False, True])
@pytest.mark.parametrize("unit", ["ms", "us"])
def test_archive_filters_and_writes_volume(tmp_path, volume, unit):
    path = tmp_path / "archive.csv"
    dl.init_csv(str(path), volume)
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(["open_time", "open", "high", "low", "close", "volume"])
    for offset in (-3600000, 0, 3600000):
        row = ROW.copy()
        row[0] = (ROW[0] + offset) * (1000 if unit == "us" else 1)
        writer.writerow(row)
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w") as z:
        z.writestr("SOLUSDT.csv", text.getvalue())
    arch = dl.ArchiveDownloader("spot", "1h", 1, 1, volume=volume)
    assert arch._write_zip_csv_rows(zipped.getvalue(), str(path), ROW[0], ROW[0]) == 1
    assert read_rows(path)[1] == [str(ROW[0] // 1000)] + ROW[1:6 if volume else 5]


@pytest.mark.parametrize("existing_volume", [False, True])
def test_resume_schema_mismatch_fails_without_writing(tmp_path, existing_volume):
    path = tmp_path / "resume.csv"
    dl.init_csv(str(path), existing_volume)
    dl.write_kline_rows(str(path), [ROW], existing_volume)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="matching --volume"):
        dl.init_csv(str(path), not existing_volume)
    assert path.read_bytes() == original
    dl.init_csv(str(path), existing_volume)
    assert path.read_bytes() == original


@pytest.mark.parametrize("source", ["api", "archive", "auto"])
def test_resume_converts_saved_seconds_before_fetch(tmp_path, monkeypatch, source):
    path = tmp_path / "resume.csv"
    dl.init_csv(str(path), True)
    dl.write_kline_rows(str(path), [ROW], True)
    starts = []

    def fetch(self, symbol, start_ms, end_ms, out_path, **kwargs):
        assert self.volume
        starts.append(start_ms)
        return 0

    monkeypatch.setattr(dl.ApiDownloader, "fetch_range", fetch)
    monkeypatch.setattr(dl.ArchiveDownloader, "fetch", fetch)
    dl.run(dl.Args("SOLUSDT", "1h", "spot", source, "2020-08-11", "2020-08-12",
                   str(path), "csv", True, 900, 1, 1, volume=True))
    assert starts == [ROW[0] + 1] * (2 if source == "auto" else 1)
    assert len(read_rows(path)) == 2


def test_api_resume_empty_remaining_range_makes_no_request(tmp_path, monkeypatch):
    path = tmp_path / "done.csv"
    dl.init_csv(str(path), True)
    dl.write_kline_rows(str(path), [ROW], True)
    api = dl.ApiDownloader("spot", "1h", 900, 1, 1, volume=True)

    def unexpected(*args):
        pytest.fail("already complete range made a network request")

    monkeypatch.setattr(api, "_get_with_retries", unexpected)
    assert api.fetch_range("SOLUSDT", ROW[0], ROW[0], str(path), True) == 0
