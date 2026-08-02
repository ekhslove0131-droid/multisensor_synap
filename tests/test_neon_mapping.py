from __future__ import annotations

import struct

import pandas as pd
import pytest

from multisensor_ml.neon_mapping import (
    SENSOR_PATH_TO_KIND,
    decode_sensor_payload,
    map_sensor_batches,
)


def _write_utf(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack(">H", len(encoded)) + encoded


def _header(path: str, sequence: int, count: int) -> bytes:
    return (
        struct.pack(">i", 1)
        + _write_utf("session-a")
        + _write_utf("batch-a")
        + struct.pack(">q", sequence)
        + struct.pack(">i", count)
    )


def test_decodes_watch_ppg_payload_using_big_endian_contract() -> None:
    payload = _header("/kidsignal/v1/ppg-batch", 4, 1) + struct.pack(
        ">qiiiiii", 1_725_000_000_040, 10, 0, 20, 0, 30, 0
    )

    decoded = decode_sensor_payload("/kidsignal/v1/ppg-batch", payload)

    assert decoded.sequence == 4
    assert decoded.samples == (
        {
            "timestamp_ms": 1_725_000_000_040,
            "raw_green": 10,
            "green_status": 0,
            "raw_ir": 20,
            "ir_status": 0,
            "raw_red": 30,
            "red_status": 0,
        },
    )


def test_decodes_sensor_batch_ibi_arrays() -> None:
    payload = _header("/kidsignal/v1/sensor-batch", 2, 1) + struct.pack(
        ">qiiiiiii", 1_725_000_000_000, 82, 1, 2, 610, 620, 0, 0
    )

    decoded = decode_sensor_payload("/kidsignal/v1/sensor-batch", payload)

    assert decoded.samples[0]["heart_rate_bpm"] == 82
    assert decoded.samples[0]["heart_rate_status"] == 1
    assert decoded.samples[0]["ibi_ms"] == [610, 620]
    assert decoded.samples[0]["ibi_status"] == [0, 0]


def test_mapping_rejects_unknown_path_and_trailing_bytes() -> None:
    payload = _header("/kidsignal/v1/eda-batch", 0, 1) + struct.pack(
        ">qfi", 1_725_000_000_000, 1.25, 0
    )
    with pytest.raises(ValueError, match="unsupported sensor path"):
        decode_sensor_payload("/unknown", payload)
    with pytest.raises(ValueError, match="trailing bytes"):
        decode_sensor_payload("/kidsignal/v1/eda-batch", payload + b"x")


def test_maps_batches_to_long_form_without_using_received_time_as_sensor_time() -> None:
    payload = _header("/kidsignal/v1/eda-batch", 0, 1) + struct.pack(
        ">qfi", 1_725_000_000_000, 1.25, 0
    )
    rows = map_sensor_batches(
        [
            {
                "sensor_path": "/kidsignal/v1/eda-batch",
                "session_id": "session-a",
                "batch_id": "batch-a",
                "sequence": 0,
                "payload": payload,
                "received_at": "2026-08-02T05:10:00Z",
            }
        ]
    )

    assert isinstance(rows, pd.DataFrame)
    assert rows.loc[0, "sensor_kind"] == SENSOR_PATH_TO_KIND["/kidsignal/v1/eda-batch"]
    assert rows.loc[0, "sensor_timestamp_utc"] == pd.Timestamp(
        "2024-08-30T06:40:00Z"
    )
    assert rows.loc[0, "received_at"] == pd.Timestamp("2026-08-02T05:10:00Z")
