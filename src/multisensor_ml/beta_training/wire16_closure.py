"""Fail-closed wire16 receipt for the incomplete third synthetic beta fit round."""

from __future__ import annotations

import json
from pathlib import Path

import onnx
from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.kernel import runtime_kernel_sha256

from .artifacts import sha256, wire_refs, write_create_only
from .wire16_support import local_path, wheel_kernel_sha, wire_paths, wire_pins

_PAIRS = {
    (source, head)
    for source in (
        "watch", "h10", "muse", "watch+h10", "watch+muse", "h10+muse",
        "watch+h10+muse",
    )
    for head in ("current", "onset60")
}
_BLOCKERS = [
    "ROUND3_IMMUTABLE_MANIFEST_NOT_CREATED",
    "ORIGINAL_PER_MODEL_EXPORT_RECEIPTS_NOT_PERSISTED",
    "ORIGINAL_VALIDATION_SELECTION_EVIDENCE_NOT_PERSISTED",
]


def finalize_wire16_compatibility(
    *,
    round3_root: Path,
    wire_root: Path,
    wire_pin_path: Path,
    failure_manifest: Path,
    wire16_runtime_wheel: Path,
    output: Path,
) -> dict[str, object]:
    """Verify transport drift closure without promoting partial model files."""
    if output.exists():
        raise FileExistsError("wire16_closure_output_exists")
    failure_bytes = failure_manifest.read_bytes()
    failure = json.loads(failure_bytes)
    if (
        failure.get("status") != "NO_QUALIFIED_BETA_MODEL"
        or failure.get("model_role") is not None
        or failure.get("partial_models_authorized_for_runtime") is not False
        or failure.get("partial_model_file_count") != 14
        or failure.get("locked_test_use_count") != 0
    ):
        raise ValueError("round3_failure_contract_invalid")
    if (round3_root / "manifest.json").exists():
        raise ValueError("round3_manifest_unexpected")

    previous = failure.get("previous_failure_ref")
    if not isinstance(previous, dict):
        raise ValueError("previous_failure_ref_invalid")
    previous_path = local_path(str(previous.get("uri")))
    previous_bytes = previous_path.read_bytes()
    if sha256(previous_bytes) != previous.get("sha256"):
        raise ValueError("previous_failure_ref_invalid")

    partials = _partial_inventory(round3_root, failure)
    wires = wire_refs(wire_root)
    pins = wire_pins(wire_pin_path)
    actual = wire_paths(wires)
    if set(actual) != set(pins) or any(actual[path] != pins[path] for path in actual):
        raise ValueError("wire16_pin_mismatch")

    standards = []
    for family in ("watch", "h10", "muse"):
        path = round3_root / "standards" / f"{family}.json"
        value = json.loads(path.read_bytes())
        if value.get("family") != family or not isinstance(value.get("schema_sha"), str):
            raise ValueError("round3_standard_invalid")
        standards.append({"family": family, "uri": path.resolve().as_uri(),
                          "sha256": sha256(path.read_bytes()),
                          "schema_sha": value["schema_sha"]})
    references = round3_root / "personal_references.json"
    reference_values = json.loads(references.read_bytes())
    if not isinstance(reference_values, dict) or len(reference_values) != 30:
        raise ValueError("round3_personal_references_invalid")

    round3_wheel = next((round3_root / "runtime").glob("*.whl"))
    round3_kernel = wheel_kernel_sha(round3_wheel)
    wire16_kernel = wheel_kernel_sha(wire16_runtime_wheel)
    if wire16_runtime_wheel.suffix != ".whl" or wire16_kernel != runtime_kernel_sha256():
        raise ValueError("round3_runtime_kernel_mismatch")
    repo = failure_manifest.parents[2]
    if list(repo.rglob(".locked-test-use.json")):
        raise ValueError("locked_test_already_accessed")
    wire16_wheel_bytes = wire16_runtime_wheel.read_bytes()
    wire16_wheel_target = output / "runtime" / wire16_runtime_wheel.name

    body: dict[str, object] = {
        "schema_version": "synthetic_beta_wire16_compatibility_v1",
        "realm": "SYNTHETIC_BETA",
        "status": "NO_QUALIFIED_BETA_MODEL",
        "closure_status": "WIRE16_COMPATIBLE_PARTIAL_ARTIFACT_NOT_QUALIFIED",
        "failure_reason": "ROUND3_IMMUTABLE_MANIFEST_NOT_CREATED",
        "qualification_blockers": _BLOCKERS,
        "model_role": None,
        "allowed_use": "BETA_ONLY",
        "stage": None,
        "delivery_eligible": False,
        "promotion_eligible": False,
        "actual_real_performance": "NOT VERIFIED",
        "synthetic_performance_status": "NOT EVALUABLE",
        "corrected_fit_round_count": int(failure["corrected_fit_round_count"]),
        "corrected_m2_fit_attempt_count": int(failure["corrected_m2_fit_attempt_count"]),
        "diagnostic_m2_fit_attempt_count": (
            int(failure["diagnostic_pre_freeze_fit_attempt_count"])
            + int(failure["diagnostic_contract_invalid_fit_attempt_count"])
            + int(failure["diagnostic_export_debug_fit_attempt_count"])
        ),
        "total_recorded_m2_fit_attempt_count": int(
            failure["total_recorded_m2_fit_attempt_count"]
        ),
        "recovery_fit_count": 0,
        "locked_test_use_count": 0,
        "source_failure_sha256": sha256(failure_bytes),
        "source_failure_uri": failure_manifest.resolve().as_uri(),
        "previous_failure_sha256": sha256(previous_bytes),
        "previous_failure_uri": previous_path.resolve().as_uri(),
        "partial_model_file_count": len(partials),
        "partial_models_authorized_for_runtime": False,
        "partial_model_refs": partials,
        "wire_ref_count": len(wires),
        "wire_contract_refs": wires,
        "wire_pin_ref": {"uri": wire_pin_path.resolve().as_uri(),
                         "sha256": sha256(wire_pin_path.read_bytes())},
        "standard_refs": standards,
        "personal_reference_index": {
            "uri": references.resolve().as_uri(),
            "sha256": sha256(references.read_bytes()),
        },
        "round3_runtime_wheel_ref": {"uri": round3_wheel.resolve().as_uri(),
                                     "sha256": sha256(round3_wheel.read_bytes())},
        "round3_runtime_kernel_sha256": round3_kernel,
        "wire16_runtime_wheel_ref": {"uri": wire16_wheel_target.resolve().as_uri(),
                                     "sha256": sha256(wire16_wheel_bytes)},
        "wire16_runtime_kernel_sha256": wire16_kernel,
    }
    receipt = {**body, "content_sha256": sha256(canonical_bytes(body))}
    output.mkdir(parents=True)
    write_create_only(wire16_wheel_target, wire16_wheel_bytes)
    write_create_only(output / "compatibility_receipt.json", canonical_bytes(receipt))
    return receipt


def _partial_inventory(round3_root: Path, failure: dict[str, object]) -> list[dict[str, str]]:
    paths = {
        ("+".join(path.relative_to(round3_root / "models").parent.parts), path.stem): path
        for path in (round3_root / "models").rglob("*.onnx")
    }
    if set(paths) != _PAIRS:
        raise ValueError("round3_partial_inventory_invalid")
    recorded = failure.get("partial_model_refs")
    if not isinstance(recorded, list):
        raise ValueError("round3_partial_inventory_invalid")
    recorded_hashes = {str(item["sha256"]) for item in recorded if isinstance(item, dict)}
    refs = []
    for pair, path in sorted(paths.items()):
        payload = path.read_bytes()
        digest = sha256(payload)
        model = onnx.load_from_string(payload)
        if (
            digest not in recorded_hashes
            or {(item.domain, item.version) for item in model.opset_import}
            != {("", 17), ("ai.onnx.ml", 3)}
            or [node.op_type for node in model.graph.node[:3]]
            != ["Sub", "Div", "TreeEnsembleClassifier"]
        ):
            raise ValueError("round3_partial_model_invalid")
        refs.append({"source_set": pair[0], "head": pair[1],
                     "uri": path.resolve().as_uri(), "sha256": digest})
    return refs
