from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from synapbridge_beta_runtime.codec import (
    canonical_bytes,
    decode_native,
    strict_float,
    validate_scope,
)
from synapbridge_beta_runtime.types import BetaScope, local_path

from .fixtures import RUN_UUID, SUBJECT_UUID, native_payload


def test_bool_is_not_a_sample() -> None:
    with pytest.raises(ValueError, match="strict_float_invalid"):
        strict_float(True)


def test_nonfinite_is_rejected() -> None:
    with pytest.raises(ValueError, match="strict_float_invalid"):
        strict_float(float("nan"))


def test_bool_epoch_is_not_an_integer_scope() -> None:
    expected = BetaScope(RUN_UUID, 1, SUBJECT_UUID)
    with pytest.raises(ValueError, match="beta_scope_invalid"):
        validate_scope({"beta_run_uuid": RUN_UUID, "run_epoch": True,
                        "beta_subject_uuid": SUBJECT_UUID}, expected)


def test_local_artifact_uri_requires_explicit_file_scheme() -> None:
    with pytest.raises(ValueError, match="local_artifact_uri_invalid"):
        local_path("/private/tmp/model.onnx")
    assert str(local_path("file:///private/tmp/model.onnx")) == "/private/tmp/model.onnx"


def test_native_npz_decodes_exact_float32_microsecond_arrays() -> None:
    manifest, content = native_payload()
    part = decode_native(manifest, content)

    assert part.scope == BetaScope(RUN_UUID, 1, SUBJECT_UUID)
    assert part.source_uuid == "30000000-0000-4000-8000-000000000001"
    assert part.family == "watch"
    assert part.start_ms == 10_000
    assert part.end_ms == 20_000
    assert part.signals["hr"].values.dtype == np.float32
    assert part.signals["hr"].time_us.dtype == np.int64
    assert part.signals["hr"].valid.dtype == np.bool_
    assert part.signals["hr"].values.flags.writeable is False
    with pytest.raises(FrozenInstanceError):
        part.family = "h10"


def test_teacher_metadata_is_rejected() -> None:
    manifest, content = native_payload(hidden=True)
    with pytest.raises(ValueError, match="native_manifest_invalid"):
        decode_native(manifest, content)


def test_wrong_dtype_and_trailing_shape_are_rejected() -> None:
    manifest, content = native_payload()
    changed = bytearray(content)
    with pytest.raises(ValueError, match="native_content_invalid"):
        decode_native(manifest, bytes(changed[:-5]))


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError, match="canonical_json_invalid"):
        canonical_bytes({"value": float("nan")})


def test_zero_count_event_signal_is_preserved_without_invented_interval() -> None:
    manifest, content = native_payload(family="h10")
    part = decode_native(manifest, content)
    assert len(part.signals["rr"].values) == 0
    assert part.signals["rr"].nominal_fs_hz is None


def test_muse_auxiliary_is_separate_from_waveform_signals() -> None:
    manifest, content = native_payload(family="muse")
    part = decode_native(manifest, content)
    assert set(part.signals) == {"eeg_1", "eeg_2", "eeg_3", "eeg_4"}
    assert set(part.auxiliary) == {"acc_x", "acc_y", "acc_z", "contact"}
    assert set(part.signals).isdisjoint(part.auxiliary)


def test_family_catalog_unit_and_cadence_are_closed() -> None:
    manifest, content = native_payload()
    manifest["signals"]["hr"]["unit"] = "Hz"
    with pytest.raises(ValueError, match="native_manifest_invalid"):
        decode_native(manifest, content)


def test_boolean_is_not_accepted_as_numeric_sample_rate() -> None:
    manifest, content = native_payload()
    manifest["signals"]["eda"]["nominal_fs_hz"] = True
    with pytest.raises(ValueError, match="native_manifest_invalid"):
        decode_native(manifest, content)
