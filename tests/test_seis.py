from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seis import ParsedFile, parse_file


SAMPLES_ROOT = Path("tests/sample_data")


def _sample_files() -> list[Path]:
    return sorted(SAMPLES_ROOT.rglob("*.mseed3"))


@pytest.mark.parametrize("path", _sample_files())
def test_parse_sample_files(path: Path):
    parsed = parse_file(path)
    assert parsed.number_of_records > 0
    if parsed.dataframe.empty:
        assert parsed.first_timestamp is None
        assert parsed.last_timestamp is None
        return

    assert parsed.first_timestamp is not None
    assert parsed.last_timestamp is not None
    assert parsed.first_timestamp <= parsed.last_timestamp
    assert pd.to_datetime(parsed.dataframe["timestamp"], utc=True).is_monotonic_increasing


@pytest.mark.parametrize("path", _sample_files())
def test_roundtrip_miniseed_csv_miniseed_bytes(path: Path, tmp_path: Path):
    parsed = parse_file(path)
    if parsed.dataframe.empty:
        return

    encoding = int(parsed.dataframe["encoding"].iloc[0])
    if encoding in {4, 5}:
        pytest.skip("float encodings handled in CSV roundtrip test")

    first_miniseed = tmp_path / f"{path.stem}-first.mseed3"
    parsed.to_miniseed(first_miniseed)

    reparsed = parse_file(first_miniseed)
    csv_path = tmp_path / f"{path.stem}-roundtrip.csv"
    reparsed.to_csv(csv_path)
    roundtrip = ParsedFile.from_csv(csv_path)
    second_miniseed = tmp_path / f"{path.stem}-second.mseed3"
    roundtrip.to_miniseed(second_miniseed)

    assert first_miniseed.read_bytes() == second_miniseed.read_bytes()


@pytest.mark.parametrize("path", _sample_files())
def test_roundtrip_csv_equivalence_for_floats(path: Path, tmp_path: Path):
    parsed = parse_file(path)
    if parsed.dataframe.empty:
        return

    encoding = int(parsed.dataframe["encoding"].iloc[0])
    if encoding not in {4, 5}:
        pytest.skip("only float encodings checked for approx CSV equivalence")

    csv_first = tmp_path / f"{path.stem}-first.csv"
    parsed.to_csv(csv_first)
    reparsed = ParsedFile.from_csv(csv_first)
    miniseed_path = tmp_path / f"{path.stem}-rt.mseed3"
    reparsed.to_miniseed(miniseed_path)
    csv_second = tmp_path / f"{path.stem}-second.csv"
    parse_file(miniseed_path).to_csv(csv_second)

    df_a = pd.read_csv(csv_first, parse_dates=["timestamp"])
    df_b = pd.read_csv(csv_second, parse_dates=["timestamp"])
    assert len(df_a) == len(df_b)
    for column in ("timestamp", "source_id", "record_index", "sample_index", "encoding"):
        assert (df_a[column] == df_b[column]).all()
    assert df_a["sample"].to_numpy() == pytest.approx(df_b["sample"].to_numpy(), rel=1e-12, abs=1e-12)
    if "sample_rate" in df_a.columns:
        assert df_a["sample_rate"].to_numpy() == pytest.approx(
            df_b["sample_rate"].to_numpy(), rel=1e-12, abs=1e-12
        )
