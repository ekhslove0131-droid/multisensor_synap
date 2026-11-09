from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.kernel import runtime_kernel_sha256
from synapbridge_beta_runtime.model_contract import validate_manifest
from synapbridge_beta_runtime.model_reader import (
    _personal_references,
    _predict,
    read_beta_runtime,
)
from synapbridge_beta_runtime.types import BetaScope, LocalArtifact
from synapbridge_beta_runtime.wire_contract import WIRE16 as WIRE16_VALUES

WIRE16 = [
    {"kind": kind, "name": name, "sha256": digest}
    for kind, name, digest in WIRE16_VALUES
]


def test_reader_contract_requires_current_sixteen_wire_refs() -> None:
    value, scope = _wire_manifest()
    validate_manifest(value, scope)


def test_reader_contract_rejects_wrong_or_duplicate_wire16_refs() -> None:
    wrong_sha = [dict(item) for item in WIRE16]
    wrong_sha[0]["sha256"] = "0" * 64
    value, scope = _wire_manifest(wrong_sha)
    with pytest.raises(ValueError, match="beta_bundle_manifest_invalid"):
        validate_manifest(value, scope)

    duplicate = [dict(item) for item in WIRE16]
    duplicate[-1] = dict(duplicate[0])
    value, scope = _wire_manifest(duplicate)
    with pytest.raises(ValueError, match="beta_bundle_manifest_invalid"):
        validate_manifest(value, scope)


def _wire_manifest(
    wire_refs: list[dict[str, str]] | None = None,
) -> tuple[dict[str, object], BetaScope]:
    scope = BetaScope(
        "10000000-0000-4000-8000-000000000001",
        1,
        "20000000-0000-4000-8000-000000000001",
    )
    policy = {"candidate_grid": [1.0, 1.5, 2.0]}
    digest = "a" * 64
    value = {
        "schema_version": "synthetic_beta_model_bundle_v1",
        "realm": "SYNTHETIC_BETA",
        "status": "BETA_SELECTED",
        "model_role": "BETA_SELECTED",
        "allowed_use": "BETA_ONLY",
        "real_data_status": "SYNTHETIC_ONLY_NOT_REAL_VERIFIED",
        "stage": None,
        "beta_run_uuid": scope.run_uuid,
        "run_epoch": scope.epoch,
        "dataset_manifest_sha256": digest,
        "training_config_sha256": digest,
        "generation_protocol_sha256": digest,
        "standard_fit_count": 3,
        "personal_calibration_count": 90,
        "calibration_support_kind": "QUALITY_QUALIFIED_FEATURE_STRIDE",
        "locked_personal_calibration_count": 0,
        "m2_fit_attempt_count": 28,
        "standard_refs": [],
        "personal_reference_index": {},
        "pattern_models": [{} for _ in range(14)],
        "selection_failures": [],
        "calibration_bounds_ms": [1_000, 1_801_000],
        "policy": policy,
        "policy_sha256": hashlib.sha256(canonical_bytes(policy)).hexdigest(),
        "wire_contract_refs": wire_refs or WIRE16,
        "kernel_sha256": runtime_kernel_sha256(),
        "runtime_wheel_ref": {},
        "runtime_lock_sha256": digest,
        "locked_test_used": False,
        "content_sha256": digest,
    }
    return value, scope


def test_reader_rejects_manifest_with_hidden_label_reference(tmp_path: Path) -> None:
    value = {
        "schema_version": "synthetic_beta_model_bundle_v1",
        "realm": "SYNTHETIC_BETA",
        "allowed_use": "BETA_ONLY",
        "label_uri": "file:///forbidden",
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path = tmp_path / "manifest.json"
    path.write_bytes(payload)
    ref = LocalArtifact(path.as_uri(), hashlib.sha256(payload).hexdigest())
    scope = BetaScope(
        "10000000-0000-4000-8000-000000000001",
        1,
        "20000000-0000-4000-8000-000000000001",
    )
    with pytest.raises(ValueError, match="beta_bundle_manifest_invalid"):
        read_beta_runtime(ref, scope)


def test_one_ready_personal_source_does_not_require_all_three(tmp_path: Path) -> None:
    subject = "20000000-0000-4000-8000-000000000001"
    value = {subject: {"watch": {
        "centers": {"eda_median_180": 1.0, "hr_median_180": 70.0},
        "scales": {"eda_median_180": 0.5, "hr_median_180": 2.0},
        "feature_order": ["eda_median_180", "hr_median_180"],
        "calibration_sha": "a" * 64,
        "scored_valid_seconds": 1800,
        "support_kind": "QUALITY_QUALIFIED_FEATURE_STRIDE",
    }}}
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path = tmp_path / "refs.json"
    path.write_bytes(payload)
    refs, reasons = _personal_references(
        {"uri": path.as_uri(), "sha256": hashlib.sha256(payload).hexdigest()}, subject
    )
    assert set(refs) == {"watch"}
    assert reasons == {
        "h10": "CALIBRATION_SUPPORT_INSUFFICIENT_OR_MISSING",
        "muse": "CALIBRATION_SUPPORT_INSUFFICIENT_OR_MISSING",
    }


def test_predict_rechecks_model_sha_before_onnxruntime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "changed.onnx"
    path.write_bytes(b"changed-after-load")
    called = False

    def forbidden_session(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        "synapbridge_beta_runtime.model_reader.ort.InferenceSession", forbidden_session,
    )
    with pytest.raises(ValueError, match="beta_artifact_sha_mismatch"):
        _predict(LocalArtifact(path.as_uri(), "0" * 64), np.zeros(2))
    assert called is False
