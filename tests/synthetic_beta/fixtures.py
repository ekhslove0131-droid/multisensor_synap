from __future__ import annotations

import hashlib
import io
import zipfile
from uuid import UUID

import numpy as np

RUN_UUID = "10000000-0000-4000-8000-000000000001"
SUBJECT_UUID = "20000000-0000-4000-8000-000000000001"
SOURCE_UUID = "30000000-0000-4000-8000-000000000001"
CATALOG: dict[str, dict[str, tuple[str, float | None]]] = {
    "watch": {
        "ppg_green": ("a.u.", 100.0), "ppg_ir": ("a.u.", 100.0),
        "ppg_red": ("a.u.", 100.0), "eda": ("uS", 1.0),
        "skin_temp": ("degC", 1.0), "hr": ("bpm", 1.0), "ibi": ("ms", None),
        "acc_x": ("g", 25.0), "acc_y": ("g", 25.0), "acc_z": ("g", 25.0),
    },
    "h10": {
        "ecg": ("uV", 130.0), "hr": ("bpm", 1.0), "rr": ("ms", None),
        "acc_x": ("g", 25.0), "acc_y": ("g", 25.0), "acc_z": ("g", 25.0),
    },
    "muse": {
        "eeg_1": ("uV", 256.0), "eeg_2": ("uV", 256.0),
        "eeg_3": ("uV", 256.0), "eeg_4": ("uV", 256.0),
    },
}
AUXILIARY = {
    "acc_x": ("g", 25.0), "acc_y": ("g", 25.0), "acc_z": ("g", 25.0),
    "contact": ("binary", 1.0),
}


def native_payload(
    *, family: str = "watch", hidden: bool = False,
) -> tuple[dict[str, object], bytes]:
    start_ms = 10_000
    arrays: dict[str, np.ndarray] = {}
    signals: dict[str, dict[str, object]] = {}
    for name, (unit, fs) in CATALOG[family].items():
        count = 0 if fs is None else int(fs * 10)
        _arrays(arrays, name, start_ms, fs, count)
        signals[name] = {"unit": unit, "nominal_fs_hz": fs, "count": count}
    auxiliary: dict[str, dict[str, object]] = {}
    if family == "muse":
        for name, (unit, fs) in AUXILIARY.items():
            count = int(fs * 10)
            _arrays(arrays, f"aux_{name}", start_ms, fs, count, binary=name == "contact")
            auxiliary[name] = {"unit": unit, "nominal_fs_hz": fs, "count": count}
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    content = output.getvalue()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        decompressed = sum(item.file_size for item in archive.infolist())
    manifest: dict[str, object] = {
        "schema_version": "synthetic_beta_native_v1",
        "realm": "SYNTHETIC_BETA",
        "beta_run_uuid": RUN_UUID,
        "run_epoch": 1,
        "beta_subject_uuid": SUBJECT_UUID,
        "beta_source_uuid": SOURCE_UUID,
        "family": family,
        "start_ms": start_ms,
        "end_ms": start_ms + 10_000,
        "duration_ms": 10_000,
        "clock": {
            "sample_time_unit": "us",
            "availability_time_unit": "ms",
            "timeline": "SYNTHETIC_LOGICAL",
        },
        "encoding": {
            "format": "npz_compressed",
            "sha256": hashlib.sha256(content).hexdigest(),
            "compressed_bytes": len(content),
            "decompressed_bytes": decompressed,
            "float32_atol": 1e-6,
        },
        "signals": signals,
    }
    if auxiliary:
        manifest["auxiliary"] = auxiliary
    if hidden:
        manifest["event_id"] = "teacher-leak"
    return manifest, content


def _arrays(
    output: dict[str, np.ndarray],
    name: str,
    start_ms: int,
    fs: float | None,
    count: int,
    *,
    binary: bool = False,
) -> None:
    values = (
        np.ones(count, dtype=np.float32)
        if binary
        else np.linspace(0.1, 1.0, count, dtype=np.float32)
    )
    period_us = int(1_000_000 / fs) if fs else 1
    time_us = np.arange(count, dtype=np.int64) * period_us + start_ms * 1_000
    output[f"{name}_values"] = values
    output[f"{name}_time_us"] = time_us
    output[f"{name}_available_ms"] = ((time_us + 999) // 1_000).astype(np.int64)
    output[f"{name}_sequence"] = np.arange(count, dtype=np.int64)
    output[f"{name}_valid"] = np.ones(count, dtype=np.bool_)


def assert_uuid4(value: str) -> None:
    assert UUID(value).version == 4
