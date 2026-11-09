"""Fail-closed handoff when a final training round stops before manifest commit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.kernel import runtime_kernel_sha256

SOURCE_SETS = ("watch", "h10", "muse", "watch+h10", "watch+muse",
               "h10+muse", "watch+h10+muse")


def finalize_post_export_failure(
    partial_root: Path, previous_failure: Path, output: Path, *,
    corrected_fit_rounds: int, diagnostic_pre_freeze_fits: int,
    diagnostic_contract_invalid_fits: int, diagnostic_export_debug_fits: int,
    corrected_runtime_wheel: Path,
) -> dict[str, object]:
    if output.exists() or (partial_root / "manifest.json").exists():
        raise FileExistsError("post_export_failure_output_invalid")
    previous_bytes = previous_failure.read_bytes()
    previous = json.loads(previous_bytes)
    expected = {
        partial_root / "models" / source / f"{head}.onnx"
        for source in SOURCE_SETS for head in ("current", "onset60")
    }
    actual = set((partial_root / "models").rglob("*.onnx"))
    counts = (corrected_fit_rounds, diagnostic_pre_freeze_fits,
              diagnostic_contract_invalid_fits, diagnostic_export_debug_fits)
    if (previous.get("realm") != "SYNTHETIC_BETA"
            or previous.get("status") != "NO_QUALIFIED_BETA_MODEL"
            or previous.get("locked_test_use_count") != 0
            or expected != actual or corrected_fit_rounds != 3
            or any(type(value) is not int or value < 0 for value in counts)
            or corrected_runtime_wheel.suffix != ".whl"):
        raise ValueError("post_export_failure_evidence_invalid")
    inventory = [_ref(path, path.read_bytes()) for path in sorted(actual)]
    wheel_bytes = corrected_runtime_wheel.read_bytes()
    wheel_target = output / "runtime" / corrected_runtime_wheel.name
    corrected_attempts = corrected_fit_rounds * 28
    body: dict[str, object] = {
        "schema_version": "synthetic_beta_failure_handoff_v3",
        "realm": "SYNTHETIC_BETA", "status": "NO_QUALIFIED_BETA_MODEL",
        "failure_reason": "ROUND3_IMMUTABLE_MANIFEST_NOT_CREATED",
        "failure_detail": "WIRE_CONTRACT_CARDINALITY_DRIFT_12_TO_14",
        "model_role": None, "allowed_use": "BETA_ONLY",
        "delivery_eligible": False, "promotion_eligible": False,
        "real_data_status": "NOT VERIFIED", "stage": None,
        "previous_failure_ref": _ref(previous_failure, previous_bytes),
        "partial_model_refs": inventory, "partial_model_file_count": len(inventory),
        "partial_models_authorized_for_runtime": False,
        "selected_head_count": 0, "required_head_count": 14,
        "corrected_fit_round_count": corrected_fit_rounds,
        "corrected_m2_fit_attempt_count": corrected_attempts,
        "diagnostic_pre_freeze_fit_attempt_count": diagnostic_pre_freeze_fits,
        "diagnostic_contract_invalid_fit_attempt_count": (
            diagnostic_contract_invalid_fits
        ),
        "diagnostic_export_debug_fit_attempt_count": diagnostic_export_debug_fits,
        "total_recorded_m2_fit_attempt_count": (
            corrected_attempts + diagnostic_pre_freeze_fits
            + diagnostic_contract_invalid_fits + diagnostic_export_debug_fits
        ),
        "locked_test_use_count": 0,
        "synthetic_performance_status": "NOT EVALUABLE",
        "actual_real_performance": "NOT VERIFIED",
        "runtime_kernel_sha256": runtime_kernel_sha256(),
        "runtime_wheel_ref": _ref(wheel_target, wheel_bytes),
    }
    manifest = {**body, "content_sha256": hashlib.sha256(
        canonical_bytes(body)
    ).hexdigest()}
    output.mkdir(parents=True)
    wheel_target.parent.mkdir(parents=True)
    _write(wheel_target, wheel_bytes)
    _write(output / "manifest.json", canonical_bytes(manifest))
    return manifest


def _ref(path: Path, content: bytes) -> dict[str, str]:
    return {"uri": path.resolve().as_uri(),
            "sha256": hashlib.sha256(content).hexdigest()}


def _write(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
