"""Decode the Galaxy Watch batch contract stored in Neon.

The ingestion table deliberately stores opaque ``bytea`` payloads.  This module
is the read-only boundary between that transport contract and a model-ready
long table.  It does not use ``received_at`` as physiological time; timestamps
inside the signed batch payload remain the source time until clock correction
has been applied.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, cast

import pandas as pd

SENSOR_PATH_TO_KIND: Final[dict[str, str]] = {
    "/kidsignal/v1/sensor-batch": "heart_rate_ibi",
    "/kidsignal/v1/eda-batch": "eda",
    "/kidsignal/v1/accelerometer-batch": "accelerometer",
    "/kidsignal/v1/ppg-batch": "ppg",
    "/kidsignal/v1/skin-temperature-batch": "skin_temperature",
}

_MAX_SAMPLES: Final[int] = 10_000


@dataclass(frozen=True, slots=True)
class DecodedSensorBatch:
    sensor_path: str
    session_id: str
    batch_id: str
    sequence: int
    samples: tuple[dict[str, object], ...]


class _Reader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.position = 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.position + size > len(self.payload):
            raise ValueError("truncated sensor payload")
        start = self.position
        self.position += size
        return self.payload[start : self.position]

    def integer(self, fmt: str) -> int:
        return int(struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0])

    def floating(self) -> float:
        return float(self.integer_as_float(struct.unpack(">f", self.take(4))[0]))

    @staticmethod
    def integer_as_float(value: float) -> float:
        return value

    def utf(self) -> str:
        raw = self.take(self.integer(">H"))
        # java.io.DataOutputStream.writeUTF uses modified UTF-8.  IDs in the
        # current contract are ASCII, but accepting the encoded NUL is cheap.
        raw = raw.replace(b"\xc0\x80", b"\x00")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid UTF-8 batch identifier") from error

    def bounded_count(self, *, allow_empty: bool = False) -> int:
        value = self.integer(">i")
        minimum = 0 if allow_empty else 1
        if value < minimum or value > _MAX_SAMPLES:
            raise ValueError("invalid sample count")
        return value


def _read_common(reader: _Reader) -> tuple[int, str, str, int, int]:
    version = reader.integer(">i")
    if version != 1:
        raise ValueError(f"unsupported sensor payload version: {version}")
    session_id = reader.utf()
    batch_id = reader.utf()
    sequence = reader.integer(">q")
    if sequence < 0:
        raise ValueError("negative sensor batch sequence")
    return version, session_id, batch_id, sequence, reader.bounded_count()


def decode_sensor_payload(sensor_path: str, payload: bytes) -> DecodedSensorBatch:
    """Decode one Kotlin ``ContractCodec`` batch into Python dictionaries."""

    if sensor_path not in SENSOR_PATH_TO_KIND:
        raise ValueError(f"unsupported sensor path: {sensor_path}")
    reader = _Reader(bytes(payload))
    _, session_id, batch_id, sequence, count = _read_common(reader)
    samples: list[dict[str, object]] = []
    if sensor_path.endswith("sensor-batch"):
        for _ in range(count):
            sample: dict[str, object] = {
                "timestamp_ms": reader.integer(">q"),
                "heart_rate_bpm": reader.integer(">i"),
                "heart_rate_status": reader.integer(">i"),
            }
            ibi_count = reader.bounded_count(allow_empty=True)
            sample["ibi_ms"] = [reader.integer(">i") for _ in range(ibi_count)]
            sample["ibi_status"] = [reader.integer(">i") for _ in range(ibi_count)]
            samples.append(sample)
    elif sensor_path.endswith("eda-batch"):
        for _ in range(count):
            samples.append(
                {
                    "timestamp_ms": reader.integer(">q"),
                    "skin_conductance_us": reader.floating(),
                    "status": reader.integer(">i"),
                }
            )
    elif sensor_path.endswith("accelerometer-batch"):
        for _ in range(count):
            samples.append(
                {
                    "timestamp_ms": reader.integer(">q"),
                    "raw_x": reader.integer(">i"),
                    "raw_y": reader.integer(">i"),
                    "raw_z": reader.integer(">i"),
                }
            )
    elif sensor_path.endswith("ppg-batch"):
        for _ in range(count):
            samples.append(
                {
                    "timestamp_ms": reader.integer(">q"),
                    "raw_green": reader.integer(">i"),
                    "green_status": reader.integer(">i"),
                    "raw_ir": reader.integer(">i"),
                    "ir_status": reader.integer(">i"),
                    "raw_red": reader.integer(">i"),
                    "red_status": reader.integer(">i"),
                }
            )
    else:
        for _ in range(count):
            samples.append(
                {
                    "timestamp_ms": reader.integer(">q"),
                    "object_temperature_c": reader.floating(),
                    "ambient_temperature_c": reader.floating(),
                    "status": reader.integer(">i"),
                }
            )
    if reader.position != len(reader.payload):
        raise ValueError("trailing bytes in sensor payload")
    return DecodedSensorBatch(
        sensor_path=sensor_path,
        session_id=session_id,
        batch_id=batch_id,
        sequence=sequence,
        samples=tuple(samples),
    )


def map_sensor_batches(records: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Return a long sample table while retaining transport and source time."""

    rows: list[dict[str, object]] = []
    for record in records:
        path = str(record["sensor_path"])
        payload = record["payload"]
        if not isinstance(payload, bytes | bytearray | memoryview):
            raise TypeError("sensor payload must be bytes-like")
        decoded = decode_sensor_payload(path, bytes(payload))
        received = record.get("received_at")
        received_at = (
            pd.to_datetime(cast(str, received), utc=True)
            if received is not None
            else pd.NaT
        )
        for sample in decoded.samples:
            timestamp_ms = int(cast(int, sample["timestamp_ms"]))
            row: dict[str, object] = {
                "sensor_path": decoded.sensor_path,
                "sensor_kind": SENSOR_PATH_TO_KIND[decoded.sensor_path],
                "session_id": decoded.session_id,
                "batch_id": decoded.batch_id,
                "sequence": decoded.sequence,
                "sensor_timestamp_ms": timestamp_ms,
                "sensor_timestamp_utc": pd.to_datetime(
                    timestamp_ms, unit="ms", utc=True
                ),
                "received_at": received_at,
            }
            row.update({key: value for key, value in sample.items() if key != "timestamp_ms"})
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "sensor_path",
                "sensor_kind",
                "session_id",
                "batch_id",
                "sequence",
                "sensor_timestamp_ms",
                "sensor_timestamp_utc",
                "received_at",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["session_id", "sensor_timestamp_utc", "sensor_kind", "sequence"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_mapped_samples(samples: pd.DataFrame) -> dict[str, object]:
    """Create a non-sensitive mapping summary for an audit report."""

    required = {"sensor_kind", "session_id", "sensor_timestamp_utc"}
    missing = sorted(required.difference(samples.columns))
    if missing:
        raise ValueError(f"mapped samples missing columns: {missing}")
    if samples.empty:
        return {"sample_count": 0, "session_count": 0, "sensor_kinds": {}}
    grouped = samples.groupby("sensor_kind", sort=True)
    kinds: dict[str, object] = {}
    for kind, frame in grouped:
        timestamps = pd.to_datetime(frame["sensor_timestamp_utc"], utc=True)
        kinds[str(kind)] = {
            "sample_count": len(frame),
            "session_count": int(frame["session_id"].nunique()),
            "first_sensor_timestamp_utc": timestamps.min().isoformat(),
            "last_sensor_timestamp_utc": timestamps.max().isoformat(),
        }
    return {
        "sample_count": len(samples),
        "session_count": int(samples["session_id"].nunique()),
        "sensor_kinds": kinds,
    }
