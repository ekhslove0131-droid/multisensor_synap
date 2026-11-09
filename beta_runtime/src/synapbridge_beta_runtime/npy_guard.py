"""Allocation-free NPZ/NPY header validation for native beta parts."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from .native_schema import (
    AUXILIARY,
    CATALOG,
    MAX_DECOMPRESSED,
    SIGNAL_FIELDS,
    nominal_rate_matches,
)

_SUFFIX_DTYPES: dict[str, np.dtype[Any]] = {
    "values": np.dtype(np.float32),
    "time_us": np.dtype(np.int64),
    "available_ms": np.dtype(np.int64),
    "sequence": np.dtype(np.int64),
    "valid": np.dtype(np.bool_),
}


def validate_npz_headers(content: bytes, manifest: Mapping[str, object]) -> None:
    expected = _expected_arrays(manifest)
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if (len(names) != len(set(names)) or set(names) != set(expected)
                    or any(item.is_dir() for item in infos)):
                raise ValueError("native_content_invalid")
            total = sum(item.file_size for item in infos)
            encoding = manifest["encoding"]
            if (not isinstance(encoding, Mapping)
                    or total != encoding["decompressed_bytes"]
                    or total > MAX_DECOMPRESSED):
                raise ValueError("native_content_invalid")
            for info in infos:
                _validate_header(archive, info, *expected[info.filename])
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("native_content_invalid") from error


def _expected_arrays(
    manifest: Mapping[str, object],
) -> dict[str, tuple[np.dtype[Any], int]]:
    family = str(manifest["family"])
    groups: list[tuple[object, Mapping[str, tuple[str, float | None]], str]] = [
        (manifest["signals"], CATALOG[family], "")
    ]
    if "auxiliary" in manifest:
        groups.append((manifest["auxiliary"], cast(
            Mapping[str, tuple[str, float | None]], AUXILIARY
        ), "aux_"))
    expected: dict[str, tuple[np.dtype[Any], int]] = {}
    for descriptors, catalog, prefix in groups:
        if not isinstance(descriptors, Mapping):
            raise ValueError("native_manifest_invalid")
        for name, (unit, fs) in catalog.items():
            descriptor = descriptors.get(name)
            if not isinstance(descriptor, Mapping) or set(descriptor) != SIGNAL_FIELDS:
                raise ValueError("native_manifest_invalid")
            count = descriptor.get("count")
            if (descriptor.get("unit") != unit
                    or not nominal_rate_matches(descriptor.get("nominal_fs_hz"), fs)
                    or type(count) is not int or count < 0
                    or (fs is not None and count != int(fs * 10))
                    or (fs is None and count > 100)):
                raise ValueError("native_manifest_invalid")
            for suffix, dtype in _SUFFIX_DTYPES.items():
                expected[f"{prefix}{name}_{suffix}.npy"] = (dtype, count)
    return expected


def _validate_header(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    expected_dtype: np.dtype[Any],
    expected_count: int,
) -> None:
    with archive.open(info) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=4096
            )
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(
                stream, max_header_size=4096
            )
        else:
            raise ValueError("native_content_invalid")
        header_bytes = stream.tell()
    dtype = np.dtype(dtype)
    if (shape != (expected_count,) or fortran or dtype.hasobject
            or dtype != expected_dtype
            or info.file_size != header_bytes + expected_count * dtype.itemsize):
        raise ValueError("native_content_invalid")
