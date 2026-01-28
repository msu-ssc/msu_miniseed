from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import struct
from typing import Iterable

import numpy as np
import pandas as pd

HEADER_FMT = "<2sBBIHHBBBBdIIBBHI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


@dataclass(slots=True)
class ParsedFile:
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None
    number_of_records: int
    dataframe: pd.DataFrame

    @classmethod
    def from_csv(cls, path: str | Path) -> "ParsedFile":
        return parse_csv(path)

    def to_csv(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_csv(path, index=False)

    def to_miniseed(self, path: Path | str) -> None:
        write_miniseed(self.dataframe, path)


def parse_file(path: str | Path) -> ParsedFile:
    data = Path(path).read_bytes()
    frames: list[pd.DataFrame] = []
    record_index = 0
    offset = 0

    while offset + HEADER_SIZE <= len(data):
        header = data[offset : offset + HEADER_SIZE]
        indicator, version, flags, nanosec, year, doy, hour, minute, second, encoding, sample_rate, nsamples, crc, pubver, id_len, extra_len, payload_len = struct.unpack(
            HEADER_FMT, header
        )
        if indicator != b"MS":
            raise ValueError(f"Invalid record indicator at offset {offset}: {indicator!r}")
        if version != 3:
            raise ValueError(f"Unsupported miniSEED version {version} at offset {offset}")

        record_len = HEADER_SIZE + id_len + extra_len + payload_len
        if offset + record_len > len(data):
            raise ValueError("Record length extends beyond file end")

        id_start = offset + HEADER_SIZE
        id_end = id_start + id_len
        source_id = data[id_start:id_end].decode("ascii", errors="replace")

        extra_start = id_end
        extra_end = extra_start + extra_len
        extra_headers = None
        if extra_len:
            extra_raw = data[extra_start:extra_end]
            try:
                extra_headers = json.loads(extra_raw.decode("utf-8"))
            except json.JSONDecodeError:
                extra_headers = {"_raw": extra_raw.decode("utf-8", errors="replace")}

        payload_start = extra_end
        payload_end = payload_start + payload_len
        payload = data[payload_start:payload_end]

        samples = _decode_payload(encoding, payload, nsamples)
        if samples and sample_rate == 0.0:
            raise ValueError("Sample rate is zero but samples are present")

        start_ts = _build_start_timestamp(year, doy, hour, minute, second, nanosec)
        timestamps = _build_timestamps(start_ts, sample_rate, len(samples))

        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "sample": samples,
                "source_id": source_id,
                "record_index": record_index,
                "sample_index": np.arange(len(samples), dtype="int64"),
                "sample_rate": sample_rate,
                "encoding": encoding,
            }
        )
        if extra_headers is not None:
            frame["extra_headers"] = json.dumps(extra_headers)
        frames.append(frame)

        offset += record_len
        record_index += 1

    if offset != len(data):
        raise ValueError("Trailing bytes after final record")

    if frames:
        dataframe = pd.concat(frames, ignore_index=True)
    else:
        dataframe = pd.DataFrame(
            columns=[
                "timestamp",
                "sample",
                "source_id",
                "record_index",
                "sample_index",
                "sample_rate",
                "encoding",
                "extra_headers",
            ]
        )

    first_timestamp = dataframe["timestamp"].iloc[0] if not dataframe.empty else None
    last_timestamp = dataframe["timestamp"].iloc[-1] if not dataframe.empty else None

    return ParsedFile(
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        number_of_records=record_index,
        dataframe=dataframe,
    )


def parse_csv(path: str | Path) -> ParsedFile:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "record_index" in df.columns:
        record_count = int(df["record_index"].max()) + 1 if not df.empty else 0
    else:
        record_count = 1 if not df.empty else 0
        df = df.copy()
        df["record_index"] = 0

    if "sample_index" not in df.columns:
        df = df.copy()
        df["sample_index"] = df.groupby("record_index").cumcount()

    first_timestamp = df["timestamp"].iloc[0] if not df.empty else None
    last_timestamp = df["timestamp"].iloc[-1] if not df.empty else None

    return ParsedFile(
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        number_of_records=record_count,
        dataframe=df,
    )


def write_miniseed(dataframe: pd.DataFrame, path: Path | str) -> None:
    if dataframe.empty:
        Path(path).write_bytes(b"")
        return

    df = dataframe.copy()
    if "timestamp" not in df.columns or "sample" not in df.columns:
        raise ValueError("dataframe must contain timestamp and sample columns")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["record_index"] = df.get("record_index", 0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[bytes] = []

    for record_index, group in df.groupby("record_index", sort=True):
        group = group.sort_values("sample_index") if "sample_index" in group.columns else group
        source_id = str(group["source_id"].iloc[0]) if "source_id" in group.columns else ""
        sample_rate = _infer_sample_rate(group)
        encoding = int(group["encoding"].iloc[0]) if "encoding" in group.columns else 3
        if "encoding" in group.columns and (group["encoding"].nunique() > 1):
            raise ValueError(f"Multiple encodings in record {record_index}")
        samples, payload = _encode_payload(encoding, group["sample"])

        start_ts = pd.Timestamp(group["timestamp"].iloc[0])
        year = start_ts.year
        doy = int(start_ts.dayofyear)
        hour = start_ts.hour
        minute = start_ts.minute
        second = start_ts.second
        nanosec = start_ts.microsecond * 1_000 + start_ts.nanosecond

        id_bytes = source_id.encode("ascii", errors="replace")
        id_len = len(id_bytes)
        extra_len = 0
        payload_len = len(payload)
        header = struct.pack(
            HEADER_FMT,
            b"MS",
            3,
            0,
            nanosec,
            year,
            doy,
            hour,
            minute,
            second,
            encoding,
            float(sample_rate),
            int(len(samples)),
            0,
            0,
            id_len,
            extra_len,
            payload_len,
        )
        parts.append(header + id_bytes + payload)

    path.write_bytes(b"".join(parts))


def _build_start_timestamp(
    year: int, doy: int, hour: int, minute: int, second: int, nanosec: int
) -> pd.Timestamp:
    date = dt.date(year, 1, 1) + dt.timedelta(days=doy - 1)
    leap = 1 if second == 60 else 0
    safe_second = 59 if second == 60 else second
    microsecond = nanosec // 1_000
    nanosecond = nanosec % 1_000
    return pd.Timestamp(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=hour,
        minute=minute,
        second=safe_second,
        microsecond=microsecond,
        nanosecond=nanosecond,
        tz="UTC",
    ) + pd.Timedelta(seconds=leap)


def _build_timestamps(
    start: pd.Timestamp, sample_rate: float, count: int
) -> pd.DatetimeIndex:
    if count == 0:
        return pd.DatetimeIndex([], tz="UTC")
    if sample_rate > 0:
        interval_seconds = 1.0 / sample_rate
    elif sample_rate < 0:
        interval_seconds = abs(sample_rate)
    else:
        interval_seconds = 0.0

    if interval_seconds == 0.0:
        return pd.DatetimeIndex([start] * count, tz="UTC")

    interval_ns = int(round(interval_seconds * 1_000_000_000))
    base_ns = start.value
    offsets = np.arange(count, dtype="int64") * interval_ns
    return pd.to_datetime(base_ns + offsets, unit="ns", utc=True)


def _infer_sample_rate(group: pd.DataFrame) -> float:
    if "sample_rate" in group.columns:
        value = float(group["sample_rate"].iloc[0])
        if value != 0.0:
            return value

    if len(group) < 2:
        return 0.0
    span = (group["timestamp"].iloc[1] - group["timestamp"].iloc[0]).total_seconds()
    if span == 0.0:
        return 0.0
    return 1.0 / span


def _encode_steim1(samples: np.ndarray) -> bytes:
    if len(samples) == 0:
        return b""

    diffs = _differences_from_samples(samples)
    x0 = int(samples[0])
    xn = int(samples[-1])
    data_words: list[int] = []
    data_codes: list[int] = []

    i = 0
    while i < len(diffs):
        if i + 4 <= len(diffs) and all(_fits_bits(d, 8) for d in diffs[i : i + 4]):
            word = 0
            for shift, val in zip((24, 16, 8, 0), diffs[i : i + 4], strict=True):
                word |= (_mask_bits(val, 8) << shift)
            data_words.append(word)
            data_codes.append(1)
            i += 4
        elif i + 2 <= len(diffs) and all(_fits_bits(d, 16) for d in diffs[i : i + 2]):
            word = 0
            for shift, val in zip((16, 0), diffs[i : i + 2], strict=True):
                word |= (_mask_bits(val, 16) << shift)
            data_words.append(word)
            data_codes.append(2)
            i += 2
        else:
            word = _mask_bits(diffs[i], 32)
            data_words.append(word)
            data_codes.append(3)
            i += 1

    frames = _build_steim_frames(data_words, data_codes, x0, xn)
    return _frames_to_bytes(frames)


def _encode_steim2(samples: np.ndarray) -> bytes:
    if len(samples) == 0:
        return b""

    diffs = _differences_from_samples(samples)
    x0 = int(samples[0])
    xn = int(samples[-1])
    data_words: list[int] = []
    data_codes: list[int] = []

    i = 0
    while i < len(diffs):
        if i + 4 <= len(diffs) and all(_fits_bits(d, 8) for d in diffs[i : i + 4]):
            word = 0
            for shift, val in zip((24, 16, 8, 0), diffs[i : i + 4], strict=True):
                word |= (_mask_bits(val, 8) << shift)
            data_words.append(word)
            data_codes.append(1)
            i += 4
            continue

        if _fits_bits(diffs[i], 30):
            word = (1 << 30) | _mask_bits(diffs[i], 30)
            data_words.append(word)
            data_codes.append(2)
            i += 1
            continue

        raise ValueError("Steim2 difference exceeds 30-bit range")

    frames = _build_steim_frames(data_words, data_codes, x0, xn)
    return _frames_to_bytes(frames)


def _differences_from_samples(samples: np.ndarray) -> list[int]:
    x0 = int(samples[0])
    diffs = [x0]
    prev = x0
    for value in samples[1:]:
        curr = int(value)
        diffs.append(curr - prev)
        prev = curr
    return diffs


def _build_steim_frames(
    data_words: list[int], data_codes: list[int], x0: int, xn: int
) -> list[list[int]]:
    frames: list[list[int]] = []
    idx = 0

    frame0 = [0] * 16
    frame0[1] = _mask_bits(x0, 32)
    frame0[2] = _mask_bits(xn, 32)
    codes0 = [0] * 16
    codes0[0] = 0
    codes0[1] = 0
    codes0[2] = 0

    for pos in range(3, 16):
        if idx >= len(data_words):
            break
        frame0[pos] = data_words[idx]
        codes0[pos] = data_codes[idx]
        idx += 1
    frame0[0] = _control_word(codes0)
    frames.append(frame0)

    while idx < len(data_words):
        frame = [0] * 16
        codes = [0] * 16
        codes[0] = 0
        for pos in range(1, 16):
            if idx >= len(data_words):
                break
            frame[pos] = data_words[idx]
            codes[pos] = data_codes[idx]
            idx += 1
        frame[0] = _control_word(codes)
        frames.append(frame)

    return frames


def _control_word(codes: list[int]) -> int:
    word = 0
    for pos, code in enumerate(codes):
        word |= (code & 0b11) << (30 - 2 * pos)
    return word


def _frames_to_bytes(frames: list[list[int]]) -> bytes:
    words: list[int] = []
    for frame in frames:
        words.extend(frame)
    return struct.pack(">" + "I" * len(words), *words)


def _fits_bits(value: int, bits: int) -> bool:
    min_val = -(1 << (bits - 1))
    max_val = (1 << (bits - 1)) - 1
    return min_val <= value <= max_val


def _mask_bits(value: int, bits: int) -> int:
    return value & ((1 << bits) - 1)


def _decode_payload(encoding: int, payload: bytes, nsamples: int) -> list[int | float]:
    if nsamples == 0:
        return []
    if encoding == 1:
        return _decode_int16(payload, nsamples)
    if encoding == 3:
        return _decode_int32(payload, nsamples)
    if encoding == 4:
        return _decode_float32(payload, nsamples)
    if encoding == 5:
        return _decode_float64(payload, nsamples)
    if encoding == 10:
        return _decode_steim1(payload, nsamples)
    if encoding == 11:
        return _decode_steim2(payload, nsamples)
    raise NotImplementedError(f"Unsupported encoding {encoding}")


def _encode_payload(encoding: int, series: pd.Series) -> tuple[np.ndarray, bytes]:
    if encoding == 1:
        samples = series.astype("int16").to_numpy()
        return samples, samples.astype("<i2", copy=False).tobytes()
    if encoding == 3:
        samples = series.astype("int32").to_numpy()
        return samples, samples.astype("<i4", copy=False).tobytes()
    if encoding == 4:
        samples = series.astype("float32").to_numpy()
        return samples, samples.astype("<f4", copy=False).tobytes()
    if encoding == 5:
        samples = series.astype("float64").to_numpy()
        return samples, samples.astype("<f8", copy=False).tobytes()
    if encoding == 10:
        samples = series.astype("int32").to_numpy()
        return samples, _encode_steim1(samples)
    if encoding == 11:
        samples = series.astype("int32").to_numpy()
        return samples, _encode_steim2(samples)
    raise NotImplementedError(f"Unsupported encoding {encoding}")


def _decode_int16(payload: bytes, nsamples: int) -> list[int]:
    count = len(payload) // 2
    values = list(struct.unpack("<" + "h" * count, payload[: count * 2]))
    return values[:nsamples]


def _decode_int32(payload: bytes, nsamples: int) -> list[int]:
    count = len(payload) // 4
    values = list(struct.unpack("<" + "i" * count, payload[: count * 4]))
    return values[:nsamples]


def _decode_float32(payload: bytes, nsamples: int) -> list[float]:
    count = len(payload) // 4
    values = list(struct.unpack("<" + "f" * count, payload[: count * 4]))
    return values[:nsamples]


def _decode_float64(payload: bytes, nsamples: int) -> list[float]:
    count = len(payload) // 8
    values = list(struct.unpack("<" + "d" * count, payload[: count * 8]))
    return values[:nsamples]


def _decode_steim1(payload: bytes, nsamples: int) -> list[int]:
    if len(payload) % 64 != 0:
        raise ValueError("Steim1 payload length must be a multiple of 64 bytes")
    words = struct.unpack(">" + "I" * (len(payload) // 4), payload)
    x0 = None
    diffs: list[int] = []

    for frame_index in range(len(words) // 16):
        base = frame_index * 16
        control = words[base]
        for word_index in range(16):
            code = (control >> (30 - 2 * word_index)) & 0b11
            if word_index == 0:
                continue
            word = words[base + word_index]
            if frame_index == 0 and word_index == 1:
                x0 = _sign_extend(word, 32)
                continue
            if frame_index == 0 and word_index == 2:
                continue
            if code == 0:
                continue
            if code == 1:
                for shift in (24, 16, 8, 0):
                    diffs.append(_sign_extend((word >> shift) & 0xFF, 8))
            elif code == 2:
                for shift in (16, 0):
                    diffs.append(_sign_extend((word >> shift) & 0xFFFF, 16))
            elif code == 3:
                diffs.append(_sign_extend(word, 32))

    if x0 is None:
        return []
    return _integrate_differences(x0, diffs, nsamples)


def _decode_steim2(payload: bytes, nsamples: int) -> list[int]:
    if len(payload) % 64 != 0:
        raise ValueError("Steim2 payload length must be a multiple of 64 bytes")
    words = struct.unpack(">" + "I" * (len(payload) // 4), payload)
    x0 = None
    diffs: list[int] = []

    for frame_index in range(len(words) // 16):
        base = frame_index * 16
        control = words[base]
        for word_index in range(16):
            code = (control >> (30 - 2 * word_index)) & 0b11
            if word_index == 0:
                continue
            word = words[base + word_index]
            if frame_index == 0 and word_index == 1:
                x0 = _sign_extend(word, 32)
                continue
            if frame_index == 0 and word_index == 2:
                continue
            if code == 0:
                continue
            if code == 1:
                for shift in (24, 16, 8, 0):
                    diffs.append(_sign_extend((word >> shift) & 0xFF, 8))
            elif code == 2:
                dnib = (word >> 30) & 0b11
                if dnib == 1:
                    diffs.append(_sign_extend(word & 0x3FFFFFFF, 30))
                elif dnib == 2:
                    for shift in (15, 0):
                        diffs.append(_sign_extend((word >> shift) & 0x7FFF, 15))
                elif dnib == 3:
                    for shift in (20, 10, 0):
                        diffs.append(_sign_extend((word >> shift) & 0x3FF, 10))
            elif code == 3:
                dnib = (word >> 30) & 0b11
                if dnib == 0:
                    for shift in (24, 18, 12, 6, 0):
                        diffs.append(_sign_extend((word >> shift) & 0x3F, 6))
                elif dnib == 1:
                    for shift in (25, 20, 15, 10, 5, 0):
                        diffs.append(_sign_extend((word >> shift) & 0x1F, 5))
                elif dnib == 2:
                    for shift in (24, 20, 16, 12, 8, 4, 0):
                        diffs.append(_sign_extend((word >> shift) & 0x0F, 4))

    if x0 is None:
        return []
    return _integrate_differences(x0, diffs, nsamples)


def _integrate_differences(x0: int, diffs: Iterable[int], nsamples: int) -> list[int]:
    diffs_list = list(diffs)
    if not diffs_list:
        return [x0] if nsamples else []

    samples: list[int] = []
    prev = x0 - diffs_list[0]
    for diff in diffs_list:
        prev += diff
        samples.append(prev)
        if len(samples) >= nsamples:
            break

    if samples and samples[0] != x0:
        samples = [x0]
        prev = x0
        for diff in diffs_list:
            prev += diff
            samples.append(prev)
            if len(samples) >= nsamples:
                break

    return samples[:nsamples]


def _sign_extend(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return value - (1 << bits) if value & sign_bit else value


def main() -> None:
    print("Use parse_file(path) to parse miniSEED data.")
