from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
import pytest
import synapbridge_beta_runtime.codec as codec

from .fixtures import native_payload


def test_declared_huge_npy_shape_is_rejected_before_numpy_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, content = native_payload()
    source = zipfile.ZipFile(io.BytesIO(content))
    malicious = io.BytesIO()
    with zipfile.ZipFile(malicious, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for info in source.infolist():
            if info.filename == "ppg_green_values.npy":
                header = io.BytesIO()
                np.lib.format.write_array_header_1_0(
                    header,
                    {"descr": "<f4", "fortran_order": False, "shape": (100_000_000,)},
                )
                target.writestr(info.filename, header.getvalue())
            else:
                target.writestr(info.filename, source.read(info.filename))
    source.close()
    payload = malicious.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        decompressed = sum(item.file_size for item in archive.infolist())
    manifest["encoding"].update(
        sha256=hashlib.sha256(payload).hexdigest(),
        compressed_bytes=len(payload),
        decompressed_bytes=decompressed,
    )
    calls = []

    def forbidden_load(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        raise AssertionError("np.load must not run")

    monkeypatch.setattr(codec.np, "load", forbidden_load)
    with pytest.raises(ValueError, match="native_content_invalid"):
        codec.decode_native(manifest, payload)
    assert calls == []
