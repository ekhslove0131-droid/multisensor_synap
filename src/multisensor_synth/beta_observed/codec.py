"""Create-only local NPZ and JSON part writer."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from synapbridge_beta_runtime.types import NativePart, SignalSamples


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def encode_part(part: NativePart) -> tuple[dict[str, object], bytes]:
    arrays: dict[str, np.ndarray] = {}
    signals = _descriptors(arrays, part.signals, "")
    auxiliary = _descriptors(arrays, part.auxiliary, "aux_")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)  # type: ignore[arg-type]
    content = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        decompressed = sum(item.file_size for item in archive.infolist())
    manifest: dict[str, object] = {
        "schema_version": "synthetic_beta_native_v1",
        "realm": "SYNTHETIC_BETA",
        "beta_run_uuid": part.scope.run_uuid,
        "run_epoch": part.scope.epoch,
        "beta_subject_uuid": part.scope.subject_uuid,
        "beta_source_uuid": part.source_uuid,
        "family": part.family,
        "start_ms": part.start_ms,
        "end_ms": part.end_ms,
        "duration_ms": 10_000,
        "clock": {"timeline": "SYNTHETIC_LOGICAL", "sample_time_unit": "us",
                  "availability_time_unit": "ms"},
        "encoding": {"format": "npz_compressed", "sha256": _sha(content),
                     "compressed_bytes": len(content), "decompressed_bytes": decompressed,
                     "float32_atol": 1e-6},
        "signals": signals,
    }
    if auxiliary:
        manifest["auxiliary"] = auxiliary
    return manifest, content


def write_part(part: NativePart, root: Path, window_no: int) -> dict[str, object]:
    manifest, content = encode_part(part)
    directory = root / part.scope.subject_uuid / part.family
    content_path = directory / f"{window_no:04d}.npz"
    manifest_path = directory / f"{window_no:04d}.json"
    _write(content_path, content)
    manifest_bytes = canonical_bytes(manifest)
    _write(manifest_path, manifest_bytes)
    return {"family": part.family, "subject_uuid": part.scope.subject_uuid,
            "source_uuid": part.source_uuid, "start_ms": part.start_ms,
            "end_ms": part.end_ms, "manifest_uri": manifest_path.resolve().as_uri(),
            "manifest_sha256": _sha(manifest_bytes),
            "content_uri": content_path.resolve().as_uri(), "content_sha256": _sha(content)}


def _descriptors(arrays: dict[str, np.ndarray], values: Mapping[str, SignalSamples],
                 prefix: str) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, samples in values.items():
        if not isinstance(samples, SignalSamples):
            raise ValueError("signal_samples_invalid")
        key = f"{prefix}{name}"
        arrays[f"{key}_values"] = samples.values
        arrays[f"{key}_time_us"] = samples.time_us
        arrays[f"{key}_available_ms"] = samples.available_ms
        arrays[f"{key}_sequence"] = samples.sequence
        arrays[f"{key}_valid"] = samples.valid
        result[name] = {"unit": samples.unit, "nominal_fs_hz": samples.nominal_fs_hz,
                        "count": len(samples.values)}
    return result


def write_json_create_only(path: Path, value: object) -> str:
    content = canonical_bytes(value)
    _write(path, content)
    return _sha(content)


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
