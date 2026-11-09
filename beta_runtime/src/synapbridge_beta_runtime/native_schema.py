"""Closed native-part schema constants shared by codec checks."""

from __future__ import annotations

import math
from typing import cast

MANIFEST_FIELDS = {
    "schema_version", "realm", "beta_run_uuid", "run_epoch", "beta_subject_uuid",
    "beta_source_uuid", "family", "start_ms", "end_ms", "duration_ms", "clock",
    "encoding", "signals",
}
SIGNAL_FIELDS = {"unit", "nominal_fs_hz", "count"}
ENCODING_FIELDS = {
    "format", "sha256", "compressed_bytes", "decompressed_bytes", "float32_atol",
}
CLOCK = {
    "sample_time_unit": "us",
    "availability_time_unit": "ms",
    "timeline": "SYNTHETIC_LOGICAL",
}
MAX_DECOMPRESSED = 2 * 1024 * 1024
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
    "muse": {f"eeg_{index}": ("uV", 256.0) for index in range(1, 5)},
}
AUXILIARY = {
    "acc_x": ("g", 25.0), "acc_y": ("g", 25.0), "acc_z": ("g", 25.0),
    "contact": ("binary", 1.0),
}


def nominal_rate_matches(value: object, expected: float | None) -> bool:
    if expected is None:
        return value is None
    if type(value) not in (int, float):
        return False
    number = cast(int | float, value)
    return math.isfinite(float(number)) and float(number) == expected
