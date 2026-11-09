"""Create-only per-head evidence that survives later bundle finalization failure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.types import local_path

from .artifacts import sha256, write_create_only

SOURCES = (
    "watch", "h10", "muse", "watch+h10", "watch+muse", "h10+muse",
    "watch+h10+muse",
)
HEADS = ("current", "onset60")
LINEAGE_KEYS = {
    "dataset_manifest_sha256", "training_config_sha256",
    "generation_protocol_sha256", "kernel_sha256", "runtime_lock_sha256",
    "standard_refs_sha256", "personal_reference_sha256", "calibration_bounds_ms",
}
MODEL_KEYS = {
    "source_set", "head", "algorithm", "threshold", "validation_sensitivity",
    "validation_far_per_hour", "validation_median_lead_seconds",
    "feature_schema_sha256", "feature_names", "model_uri", "model_sha256",
    "onnx_receipt",
}
def write_model_evidence(
    output: Path, model: dict[str, object], lineage: dict[str, object],
) -> dict[str, str]:
    _validate_lineage(lineage)
    _validate_model(model)
    source, head = str(model["source_set"]), str(model["head"])
    body = {
        "schema_version": "synthetic_beta_model_evidence_v1",
        "realm": "SYNTHETIC_BETA",
        "state": "VALIDATION_SELECTED_EXPORT_VERIFIED",
        "selection_role": "VALIDATION",
        "model": model,
        "lineage": lineage,
    }
    receipt = {**body, "content_sha256": sha256(canonical_bytes(body))}
    path = output / "evidence" / "models" / source / f"{head}.json"
    try:
        write_create_only(path, canonical_bytes(receipt))
    except FileExistsError as error:
        raise FileExistsError("training_evidence_exists") from error
    return {"uri": path.resolve().as_uri(), "sha256": sha256(path.read_bytes())}
def write_policy_evidence(
    output: Path, policy: dict[str, object], lineage: dict[str, object],
) -> dict[str, str]:
    _validate_lineage(lineage)
    d_on, d_off = policy.get("d_on"), policy.get("d_off")
    if type(d_on) not in (int, float) or type(d_off) not in (int, float):
        raise ValueError("policy_evidence_invalid")
    d_on_value = float(cast(int | float, d_on))
    d_off_value = float(cast(int | float, d_off))
    if (
        policy.get("candidate_grid") != [1.0, 1.5, 2.0]
        or d_on_value not in (1.0, 1.5, 2.0)
        or d_off_value != d_on_value / 2
    ):
        raise ValueError("policy_evidence_invalid")
    body = {
        "schema_version": "synthetic_beta_policy_evidence_v1",
        "realm": "SYNTHETIC_BETA",
        "selection_role": "VALIDATION",
        "policy": policy,
        "lineage": lineage,
    }
    receipt = {**body, "content_sha256": sha256(canonical_bytes(body))}
    path = output / "evidence" / "policy.json"
    try:
        write_create_only(path, canonical_bytes(receipt))
    except FileExistsError as error:
        raise FileExistsError("training_evidence_exists") from error
    return {"uri": path.resolve().as_uri(), "sha256": sha256(path.read_bytes())}
def complete_training_evidence(
    output: Path, lineage: dict[str, object],
    model_refs: list[dict[str, str]], policy_ref: dict[str, str] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    _validate_lineage(lineage)
    expected = {(source, head) for source in SOURCES for head in HEADS}
    physical = list((output / "evidence/models").rglob("*.json"))
    if len(physical) != 14 or len(model_refs) != 14 or policy_ref is None:
        raise ValueError("training_evidence_set_incomplete")
    paths = {}
    for path in physical:
        relative = path.relative_to(output / "evidence/models")
        if len(relative.parts) != 2:
            raise ValueError("training_evidence_set_incomplete")
        pair = (relative.parts[0], path.stem)
        if pair in paths:
            raise ValueError("training_evidence_set_incomplete")
        paths[pair] = path
    if set(paths) != expected:
        raise ValueError("training_evidence_set_incomplete")
    bound = _bound_refs(model_refs, set(paths.values()))
    models = []
    for pair in [(source, head) for source in SOURCES for head in HEADS]:
        path = paths[pair]
        value = _read(path, "synthetic_beta_model_evidence_v1", bound[path.resolve()])
        if value.get("lineage") != lineage or value.get("selection_role") != "VALIDATION":
            raise ValueError("training_evidence_lineage_mismatch")
        model = value.get("model")
        if not isinstance(model, dict):
            raise ValueError("model_evidence_invalid")
        _validate_model(model)
        if (str(model["source_set"]), str(model["head"])) != pair:
            raise ValueError("model_evidence_invalid")
        models.append(model)
    policy_path = output / "evidence/policy.json"
    policy_bound = _bound_refs([policy_ref], {policy_path})
    policy_value = _read(
        policy_path, "synthetic_beta_policy_evidence_v1",
        policy_bound[policy_path.resolve()],
    )
    if (policy_value.get("lineage") != lineage
            or policy_value.get("selection_role") != "VALIDATION"):
        raise ValueError("training_evidence_lineage_mismatch")
    policy = policy_value.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("policy_evidence_invalid")
    _validate_policy(policy)
    return models, policy
def _validate_lineage(value: dict[str, object]) -> None:
    if set(value) != LINEAGE_KEYS:
        raise ValueError("training_evidence_lineage_invalid")
    for key in LINEAGE_KEYS - {"calibration_bounds_ms"}:
        item = value[key]
        if not isinstance(item, str) or len(item) != 64:
            raise ValueError("training_evidence_lineage_invalid")
    bounds = value["calibration_bounds_ms"]
    if (
        not isinstance(bounds, list) or len(bounds) != 2
        or any(type(item) is not int for item in bounds)
        or bounds[1] - bounds[0] != 1_800_000
    ):
        raise ValueError("training_evidence_lineage_invalid")
def _validate_model(value: dict[str, object]) -> None:
    if set(value) != MODEL_KEYS:
        raise ValueError("model_evidence_invalid")
    source, head = str(value["source_set"]), str(value["head"])
    receipt = value["onnx_receipt"]
    path = local_path(str(value["model_uri"]))
    if (
        source not in SOURCES or head not in HEADS
        or value["algorithm"] not in ("LR", "HGB")
        or not isinstance(receipt, dict)
        or receipt.get("sha256") != value["model_sha256"]
        or receipt.get("class_order") != [0, 1]
        or receipt.get("decision_mismatch_count") != 0
        or float(receipt.get("max_absolute_error", np.inf)) > 1e-5
        or sha256(path.read_bytes()) != value["model_sha256"]
    ):
        raise ValueError("model_evidence_artifact_mismatch")
def _validate_policy(policy: dict[str, object]) -> None:
    d_on, d_off = policy.get("d_on"), policy.get("d_off")
    if (
        policy.get("version") != "synthetic_beta_stage_policy_v1"
        or policy.get("candidate_grid") != [1.0, 1.5, 2.0]
        or type(d_on) not in (int, float) or d_on not in (1.0, 1.5, 2.0)
        or type(d_off) not in (int, float) or float(cast(int | float, d_off))
        != float(cast(int | float, d_on)) / 2
    ):
        raise ValueError("policy_evidence_invalid")


def _bound_refs(refs: list[dict[str, str]], paths: set[Path]) -> dict[Path, str]:
    if len(refs) != len(paths):
        raise ValueError("training_evidence_ref_mismatch")
    resolved = {local_path(ref["uri"]): ref["sha256"] for ref in refs}
    expected = {path.resolve() for path in paths}
    if set(resolved) != expected:
        raise ValueError("training_evidence_ref_mismatch")
    return resolved


def _read(path: Path, schema: str, expected_sha256: str) -> dict[str, object]:
    payload = path.read_bytes()
    if sha256(payload) != expected_sha256:
        raise ValueError("training_evidence_ref_mismatch")
    value = json.loads(payload)
    if (not isinstance(value, dict) or value.get("schema_version") != schema
            or value.get("realm") != "SYNTHETIC_BETA"):
        raise ValueError("training_evidence_invalid")
    content = value.get("content_sha256")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if content != sha256(canonical_bytes(body)):
        raise ValueError("training_evidence_content_mismatch")
    return value
