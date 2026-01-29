from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from msu_miniseed import (
    MiniseedData,
    MiniseedRecord,
    read_miniseed,
    read_csv,
    read_json,
)


SAMPLES_ROOT = Path("tests/sample_data")


def _sample_files() -> list[Path]:
    return sorted(SAMPLES_ROOT.rglob("*.mseed3"))


def _json_sample_files() -> list[Path]:
    return sorted(SAMPLES_ROOT.rglob("*.json"))


@pytest.mark.parametrize("path", _sample_files(), ids=lambda x: x.stem)
def test_parse_sample_files(path: Path):
    parsed = read_miniseed(path)
    assert parsed.number_of_records > 0
    if parsed.df.empty:
        assert parsed.first_timestamp is None
        assert parsed.last_timestamp is None
        return

    assert parsed.first_timestamp is not None
    assert parsed.last_timestamp is not None
    assert parsed.first_timestamp <= parsed.last_timestamp
    assert pd.to_datetime(parsed.df["timestamp"], utc=True).is_monotonic_increasing


@pytest.mark.parametrize("path", _sample_files(), ids=lambda x: x.stem)
def test_roundtrip_miniseed_csv_miniseed_bytes(path: Path, tmp_path: Path):
    parsed = read_miniseed(path)
    if parsed.df.empty:
        return

    encoding = int(parsed.df["encoding"].iloc[0])
    if encoding in {4, 5}:
        pytest.skip("float encodings handled in CSV roundtrip test")

    first_miniseed = tmp_path / f"{path.stem}-first.mseed3"
    parsed.to_miniseed(first_miniseed)

    reparsed = read_miniseed(first_miniseed)
    csv_path = tmp_path / f"{path.stem}-roundtrip.csv"
    reparsed.to_csv(csv_path)
    roundtrip = read_csv(csv_path)
    second_miniseed = tmp_path / f"{path.stem}-second.mseed3"
    roundtrip.to_miniseed(second_miniseed)

    assert first_miniseed.read_bytes() == second_miniseed.read_bytes()


@pytest.mark.parametrize("path", _sample_files(), ids=lambda x: x.stem)
def test_roundtrip_csv_equivalence_for_floats(path: Path, tmp_path: Path):
    parsed = read_miniseed(path)
    if parsed.df.empty:
        return

    encoding = int(parsed.df["encoding"].iloc[0])
    if encoding not in {4, 5}:
        pytest.skip("only float encodings checked for approx CSV equivalence")

    csv_first = tmp_path / f"{path.stem}-first.csv"
    parsed.to_csv(csv_first)
    reparsed = read_csv(csv_first)
    miniseed_path = tmp_path / f"{path.stem}-rt.mseed3"
    reparsed.to_miniseed(miniseed_path)
    csv_second = tmp_path / f"{path.stem}-second.csv"
    read_miniseed(miniseed_path).to_csv(csv_second)

    df_a = pd.read_csv(csv_first, parse_dates=["timestamp"])
    df_b = pd.read_csv(csv_second, parse_dates=["timestamp"])
    assert len(df_a) == len(df_b)
    for column in (
        "timestamp",
        "source_id",
        "record_index",
        "sample_index",
        "encoding",
    ):
        assert (df_a[column] == df_b[column]).all()
    assert df_a["sample"].to_numpy() == pytest.approx(
        df_b["sample"].to_numpy(), rel=1e-12, abs=1e-12
    )
    if "sample_rate" in df_a.columns:
        assert df_a["sample_rate"].to_numpy() == pytest.approx(
            df_b["sample_rate"].to_numpy(), rel=1e-12, abs=1e-12
        )


@pytest.mark.parametrize("path", _json_sample_files(), ids=lambda x: x.stem)
def test_roundtrip_json_files(path: Path, tmp_path: Path):
    original_blob = json.loads(path.read_text(encoding="utf-8"))
    parsed = read_json(path)
    reconstructed_path = tmp_path / f"{path.stem}-reconstructed.json"
    parsed.to_json(reconstructed_path)

    reconstructed_text = reconstructed_path.read_text(encoding="utf-8")
    assert reconstructed_text == json.dumps(original_blob, indent=4)
    reconstructed_blob = json.loads(reconstructed_text)
    assert original_blob == reconstructed_blob


def _record(**overrides) -> MiniseedRecord:
    base = MiniseedRecord(
        record_index=0,
        source_id="FDSN:XX_TEST__L_H_Z",
        start_timestamp=pd.Timestamp("2022-01-01T00:00:00Z"),
        sample_rate=1.0,
        encoding=11,
        samples=[1, 2, 3],
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"source_id": "FDSN:YY_TEST__L_H_Z"}, False),
        ({"sample_rate": 2.0}, False),
        ({"encoding": 10}, False),
    ],
)
def test_miniseed_record_compatible_with(overrides, expected):
    left = _record()
    right = _record(**overrides)
    assert left.compatible_with(right) is expected


def test_miniseed_record_compatible_with_rejects_other_type():
    record = _record()
    assert record.compatible_with("not-a-record") is False


def test_miniseed_data_all_compatible_empty():
    assert MiniseedData().all_compatible() is True


def test_miniseed_data_all_compatible_single():
    data = MiniseedData(records=[_record()])
    assert data.all_compatible() is True


def test_miniseed_data_all_compatible_all_match():
    records = [_record(samples=[10]), _record(samples=[20, 30], record_index=1)]
    data = MiniseedData(records=records)
    assert data.all_compatible() is True


def test_miniseed_data_all_compatible_detects_mismatch():
    records = [_record(), _record(source_id="FDSN:DIFF__L_H_Z", record_index=1)]
    data = MiniseedData(records=records)
    assert data.all_compatible() is False
