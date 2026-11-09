"""Immutable handoff for a truthful NO_QUALIFIED beta outcome."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.kernel import runtime_kernel_sha256


def finalize_no_qualified(
    round_manifest: Path,
    evaluation: Path,
    output: Path,
    *,
    corrected_fit_rounds: int,
    diagnostic_fit_attempts: int,
    diagnostic_contract_invalid_fit_attempts: int,
    corrected_runtime_wheel: Path,
) -> dict[str, object]:
    if output.exists():
        raise FileExistsError("beta_failure_output_exists")
    round_bytes = round_manifest.read_bytes()
    evaluation_bytes = evaluation.read_bytes()
    round_value = json.loads(round_bytes)
    evaluation_value = json.loads(evaluation_bytes)
    if (round_value.get("realm") != "SYNTHETIC_BETA"
            or round_value.get("status") != "NO_QUALIFIED_BETA_MODEL"
            or evaluation_value.get("status") != "NO_QUALIFIED_BETA_MODEL"
            or evaluation_value.get("locked_use") != {
                "status": "NOT_ACCESSED", "use_count": 0,
            }
            or corrected_fit_rounds < 1 or diagnostic_fit_attempts < 0
            or diagnostic_contract_invalid_fit_attempts < 0
            or corrected_runtime_wheel.suffix != ".whl"):
        raise ValueError("beta_failure_evidence_invalid")
    runtime_bytes = corrected_runtime_wheel.read_bytes()
    runtime_target = output / "runtime" / corrected_runtime_wheel.name
    missing = [
        {"source_set": item["source_set"], "head": item["head"],
         "reason": item["reason"]}
        for item in round_value["selection_failures"]
        if item.get("reason") == "NO_QUALIFIED_BETA_MODEL"
    ]
    if not missing:
        raise ValueError("beta_failure_missing_head_invalid")
    body: dict[str, object] = {
        "schema_version": "synthetic_beta_failure_handoff_v2",
        "realm": "SYNTHETIC_BETA", "status": "NO_QUALIFIED_BETA_MODEL",
        "model_role": None, "allowed_use": "BETA_ONLY",
        "delivery_eligible": False, "promotion_eligible": False,
        "real_data_status": "NOT VERIFIED", "stage": None,
        "round_manifest_ref": _ref(round_manifest, round_bytes),
        "evaluation_ref": _ref(evaluation, evaluation_bytes),
        "dataset_manifest_sha256": round_value["dataset_manifest_sha256"],
        "training_config_sha256": round_value["training_config_sha256"],
        "generation_protocol_sha256": round_value["generation_protocol_sha256"],
        "training_round_runtime_wheel_ref": round_value["runtime_wheel_ref"],
        "runtime_wheel_ref": _ref(runtime_target, runtime_bytes),
        "runtime_kernel_sha256": runtime_kernel_sha256(),
        "wire_contract_refs": round_value["wire_contract_refs"],
        "corrected_fit_round_count": corrected_fit_rounds,
        "corrected_m2_fit_attempt_count": (
            corrected_fit_rounds * int(round_value["m2_fit_attempt_count"])
        ),
        "diagnostic_pre_freeze_fit_attempt_count": diagnostic_fit_attempts,
        "diagnostic_contract_invalid_fit_attempt_count": (
            diagnostic_contract_invalid_fit_attempts
        ),
        "total_recorded_m2_fit_attempt_count": (
            corrected_fit_rounds * int(round_value["m2_fit_attempt_count"])
            + diagnostic_fit_attempts + diagnostic_contract_invalid_fit_attempts
        ),
        "selected_head_count": len(round_value["pattern_models"]),
        "required_head_count": 14, "missing_heads": missing,
        "locked_test_use_count": 0,
        "synthetic_performance_status": "NOT EVALUABLE",
        "actual_real_performance": "NOT VERIFIED",
    }
    manifest = {**body, "content_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest()}
    output.mkdir(parents=True)
    runtime_target.parent.mkdir(parents=True)
    _write(runtime_target, runtime_bytes)
    payload = canonical_bytes(manifest)
    _write(output / "manifest.json", payload)
    return manifest


def _ref(path: Path, content: bytes) -> dict[str, str]:
    return {"uri": path.resolve().as_uri(), "sha256": hashlib.sha256(content).hexdigest()}


def _write(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
