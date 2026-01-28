from __future__ import annotations

from pathlib import Path

import pandas as pd

from seis import ParsedFile, parse_file


DATA_FILE = Path("data/seis_fr/bbz_697145747.mseed")
CSV_FILE = Path("output/csv/seis_fr/bbz_697145747.csv")


def test_parse_file_basic():
    parsed = parse_file(DATA_FILE)
    assert parsed.number_of_records > 0
    assert not parsed.dataframe.empty
    assert parsed.first_timestamp is not None
    assert parsed.last_timestamp is not None
    assert parsed.first_timestamp <= parsed.last_timestamp


def test_csv_roundtrip(tmp_path: Path):
    parsed = parse_file(DATA_FILE)
    out_csv = tmp_path / "roundtrip.csv"
    parsed.to_csv(out_csv)
    parsed_csv = ParsedFile.from_csv(out_csv)

    assert len(parsed_csv.dataframe) == len(parsed.dataframe)
    assert parsed_csv.first_timestamp == parsed.first_timestamp
    assert parsed_csv.last_timestamp == parsed.last_timestamp


def test_parse_csv_matches_sample_file():
    parsed = ParsedFile.from_csv(CSV_FILE)
    assert parsed.number_of_records > 0
    assert parsed.dataframe["encoding"].iloc[0] == 11


def test_miniseed_csv_miniseed_roundtrip_bytes(tmp_path: Path):
    parsed = ParsedFile.from_csv(CSV_FILE)
    subset = parsed.dataframe.head(2000).copy()
    first_miniseed = tmp_path / "first.mseed"
    ParsedFile(
        first_timestamp=subset["timestamp"].iloc[0],
        last_timestamp=subset["timestamp"].iloc[-1],
        number_of_records=int(subset["record_index"].max()) + 1,
        dataframe=subset,
    ).to_miniseed(first_miniseed)

    reparsed = parse_file(first_miniseed)
    csv_path = tmp_path / "roundtrip.csv"
    reparsed.to_csv(csv_path)
    roundtrip = ParsedFile.from_csv(csv_path)
    second_miniseed = tmp_path / "second.mseed"
    roundtrip.to_miniseed(second_miniseed)

    assert first_miniseed.read_bytes() == second_miniseed.read_bytes()
