"""Closed local codec for lossless ten-second native signal parts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections.abc import Mapping
from typing import Any, cast
from uuid import UUID

import numpy as np

from .native_schema import (
    AUXILIARY,
    CATALOG,
    CLOCK,
    ENCODING_FIELDS,
    MANIFEST_FIELDS,
    MAX_DECOMPRESSED,
    SIGNAL_FIELDS,
    nominal_rate_matches,
)
from .npy_guard import validate_npz_headers
from .numeric import strict_float
from .types import BetaScope, NativePart, SignalSamples

__all__ = ["strict_float"]

_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("canonical_json_invalid") from error


def validate_scope(payload: Mapping[str, object], expected: BetaScope) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "beta_run_uuid", "run_epoch", "beta_subject_uuid",
    }:
        raise ValueError("beta_scope_invalid")
    epoch = payload["run_epoch"]
    if (not _uuid4(payload["beta_run_uuid"])
            or type(epoch) is not int or epoch < 1
            or not _uuid4(payload["beta_subject_uuid"])):
        raise ValueError("beta_scope_invalid")
    if (payload["beta_run_uuid"], payload["run_epoch"], payload["beta_subject_uuid"]) != (
        expected.run_uuid, expected.epoch, expected.subject_uuid,
    ):
        raise ValueError("beta_scope_mismatch")
    return dict(payload)


def decode_native(manifest: Mapping[str, Any], content: bytes) -> NativePart:
    _validate_manifest(manifest)
    encoding = manifest["encoding"]
    if (encoding["sha256"] != hashlib.sha256(content).hexdigest()
            or encoding["compressed_bytes"] != len(content)):
        raise ValueError("native_content_invalid")
    validate_npz_headers(content, manifest)
    try:
        loaded = np.load(io.BytesIO(content), allow_pickle=False)
    except Exception as error:
        raise ValueError("native_content_invalid") from error
    try:
        family = str(manifest["family"])
        signals, signal_keys = _decode_group(loaded, manifest["signals"], CATALOG[family], "")
        auxiliary_value = manifest.get("auxiliary", {})
        auxiliary, aux_keys = _decode_group(
            loaded, auxiliary_value, AUXILIARY if auxiliary_value else {}, "aux_"
        )
        if set(loaded.files) != signal_keys | aux_keys:
            raise ValueError("native_content_invalid")
    finally:
        loaded.close()
    scope = BetaScope(str(manifest["beta_run_uuid"]), int(manifest["run_epoch"]),
                      str(manifest["beta_subject_uuid"]))
    return NativePart(
        scope, str(manifest["beta_source_uuid"]), family,
        NativePart.freeze_signals(signals), NativePart.freeze_auxiliary(auxiliary),
        int(manifest["start_ms"]), int(manifest["end_ms"]),
    )


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError("native_manifest_invalid")
    fields = set(manifest)
    if fields != MANIFEST_FIELDS and fields != MANIFEST_FIELDS | {"auxiliary"}:
        raise ValueError("native_manifest_invalid")
    family = manifest.get("family")
    start, end = manifest.get("start_ms"), manifest.get("end_ms")
    if (manifest.get("schema_version") != "synthetic_beta_native_v1"
            or manifest.get("realm") != "SYNTHETIC_BETA"
            or not all(_uuid4(manifest.get(key)) for key in (
                "beta_run_uuid", "beta_subject_uuid", "beta_source_uuid"))
            or type(manifest.get("run_epoch")) is not int or manifest["run_epoch"] < 1
            or family not in CATALOG or type(start) is not int or start < 1
            or type(end) is not int or end < 1
            or manifest.get("duration_ms") != 10_000 or end - start != 10_000
            or manifest.get("clock") != CLOCK
            or not _encoding_structure_valid(manifest.get("encoding"))
            or not isinstance(manifest.get("signals"), Mapping)
            or set(manifest["signals"]) != set(CATALOG[family])
            or ("auxiliary" in manifest and not isinstance(manifest["auxiliary"], Mapping))
            or (family != "muse" and "auxiliary" in manifest)
            or (family == "muse" and "auxiliary" in manifest
                and set(manifest["auxiliary"]) != set(AUXILIARY))):
        raise ValueError("native_manifest_invalid")


def _decode_group(loaded: Any, descriptors: Mapping[str, Any],
                  catalog: Mapping[str, tuple[str, float | None]], prefix: str
                  ) -> tuple[dict[str, SignalSamples], set[str]]:
    samples: dict[str, SignalSamples] = {}
    expected: set[str] = set()
    for name, expected_contract in catalog.items():
        descriptor = descriptors[name]
        if not isinstance(descriptor, Mapping) or set(descriptor) != SIGNAL_FIELDS:
            raise ValueError("native_manifest_invalid")
        unit, fs = expected_contract
        count = descriptor.get("count")
        if (descriptor.get("unit") != unit
                or not nominal_rate_matches(descriptor.get("nominal_fs_hz"), fs)):
            raise ValueError("native_manifest_invalid")
        if (type(count) is not int or count < 0
                or (fs is not None and count != int(fs * 10))
                or (fs is None and count > 100)):
            raise ValueError("native_manifest_invalid")
        key = f"{prefix}{name}"
        arrays = {suffix: loaded[f"{key}_{suffix}"] for suffix in (
            "values", "time_us", "available_ms", "sequence", "valid"
        )}
        expected.update(f"{key}_{suffix}" for suffix in arrays)
        _validate_arrays(arrays, count, binary=name == "contact")
        frozen = {}
        for suffix, array in arrays.items():
            copied = np.array(array, copy=True)
            copied.flags.writeable = False
            frozen[suffix] = copied
        samples[name] = SignalSamples(
            frozen["values"], frozen["time_us"], frozen["available_ms"],
            frozen["sequence"], frozen["valid"], unit, fs,
        )
    return samples, expected


def _validate_arrays(arrays: Mapping[str, np.ndarray], count: int, *, binary: bool) -> None:
    if (arrays["values"].dtype != np.float32 or arrays["time_us"].dtype != np.int64
            or arrays["available_ms"].dtype != np.int64
            or arrays["sequence"].dtype != np.int64 or arrays["valid"].dtype != np.bool_
            or any(array.ndim != 1 or len(array) != count for array in arrays.values())
            or not np.isfinite(arrays["values"]).all()
            or (count > 1 and (np.any(np.diff(arrays["time_us"]) <= 0)
                              or np.any(np.diff(arrays["sequence"]) <= 0)))
            or np.any(arrays["available_ms"] * 1_000 < arrays["time_us"])
            or (binary and not np.isin(arrays["values"], (0.0, 1.0)).all())):
        raise ValueError("native_content_invalid")


def _encoding_structure_valid(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != ENCODING_FIELDS:
        return False
    atol = value.get("float32_atol")
    compressed = value.get("compressed_bytes")
    decompressed = value.get("decompressed_bytes")
    if (type(compressed) is not int or type(decompressed) is not int
            or type(atol) not in (int, float)):
        return False
    result = (value.get("format") == "npz_compressed"
            and bool(require_sha256(value.get("sha256")))
            and 0 < compressed <= MAX_DECOMPRESSED
            and 0 < decompressed <= MAX_DECOMPRESSED
            and math.isfinite(float(cast(int | float, atol)))
            and 0 < float(cast(int | float, atol)) <= 1e-5)
    return bool(result)


def _uuid4(value: object) -> bool:
    try:
        parsed = UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return False
    return bool(parsed.version == 4 and str(parsed) == value)


def require_sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("sha256_invalid")
    return value
