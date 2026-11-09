"""Closed manifest and immutable reference validators for beta runtime loading."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .codec import canonical_bytes, require_sha256, strict_float, validate_scope
from .kernel import runtime_kernel_sha256
from .model_types import PatternModelArtifact, SourceReference
from .pattern_runtime import SOURCE_ORDER, canonical_source_set
from .types import BetaScope, LocalArtifact, local_path
from .wire_contract import validate_wire16

FIELDS = {
    "schema_version", "realm", "status", "model_role", "allowed_use",
    "real_data_status", "stage", "beta_run_uuid", "run_epoch",
    "dataset_manifest_sha256", "training_config_sha256",
    "generation_protocol_sha256", "standard_fit_count",
    "personal_calibration_count", "calibration_support_kind",
    "locked_personal_calibration_count", "m2_fit_attempt_count", "standard_refs",
    "personal_reference_index", "pattern_models", "selection_failures",
    "calibration_bounds_ms", "policy", "policy_sha256", "wire_contract_refs",
    "kernel_sha256", "runtime_wheel_ref", "runtime_lock_sha256", "locked_test_used",
    "content_sha256",
}


def validate_manifest(value: object, scope: BetaScope) -> None:
    if (not isinstance(value, dict) or set(value) != FIELDS
            or value.get("schema_version") != "synthetic_beta_model_bundle_v1"
            or value.get("realm") != "SYNTHETIC_BETA"
            or value.get("status") != "BETA_SELECTED"
            or value.get("model_role") != "BETA_SELECTED"
            or value.get("allowed_use") != "BETA_ONLY"
            or value.get("real_data_status") != "SYNTHETIC_ONLY_NOT_REAL_VERIFIED"
            or value.get("stage") is not None
            or value.get("beta_run_uuid") != scope.run_uuid
            or value.get("run_epoch") != scope.epoch
            or value.get("standard_fit_count") != 3
            or value.get("calibration_support_kind") != "QUALITY_QUALIFIED_FEATURE_STRIDE"
            or value.get("m2_fit_attempt_count") != 28
            or value.get("locked_test_used") is not False
            or value.get("locked_personal_calibration_count") != 0
            or value.get("kernel_sha256") != runtime_kernel_sha256()
            or not isinstance(value.get("pattern_models"), list)
            or len(value["pattern_models"]) != 14
            or contains_forbidden_key(value)):
        raise ValueError("beta_bundle_manifest_invalid")
    validate_scope({"beta_run_uuid": value["beta_run_uuid"],
                    "run_epoch": value["run_epoch"],
                    "beta_subject_uuid": scope.subject_uuid}, scope)
    for key in ("content_sha256", "dataset_manifest_sha256", "training_config_sha256",
                "generation_protocol_sha256", "kernel_sha256",
                "runtime_lock_sha256", "policy_sha256"):
        require_sha256(value[key])
    bounds = value.get("calibration_bounds_ms")
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(type(item) is not int for item in bounds)
            or bounds[1] - bounds[0] != 1_800_000):
        raise ValueError("beta_bundle_manifest_invalid")
    policy = value.get("policy")
    if (not isinstance(policy, dict)
            or hashlib.sha256(canonical_bytes(policy)).hexdigest() != value["policy_sha256"]
            or policy.get("candidate_grid") != [1.0, 1.5, 2.0]):
        raise ValueError("beta_bundle_manifest_invalid")
    wire_refs = value.get("wire_contract_refs")
    validate_wire16(wire_refs)


def source_schemas(refs: list[dict[str, str]]) -> dict[str, str]:
    result = {}
    for ref in refs:
        value = read_ref(ref)
        if not isinstance(value, dict) or not isinstance(value.get("schema_sha"), str):
            raise ValueError("beta_standard_ref_invalid")
        result[ref["family"]] = value["schema_sha"]
    if set(result) != set(SOURCE_ORDER):
        raise ValueError("beta_standard_set_invalid")
    return result


def personal_references(
    ref: dict[str, str], subject: str,
) -> tuple[dict[str, SourceReference], dict[str, str]]:
    value = read_ref(ref)
    if not isinstance(value, dict) or subject not in value:
        raise ValueError("beta_subject_reference_missing")
    subject_value = value[subject]
    if not isinstance(subject_value, dict):
        raise ValueError("beta_subject_reference_missing")
    result, unready = {}, {}
    for family in SOURCE_ORDER:
        item = subject_value.get(family)
        seconds = item.get("scored_valid_seconds") if isinstance(item, dict) else None
        if (not isinstance(item, dict) or type(seconds) is not int or seconds < 1800
                or item.get("support_kind") != "QUALITY_QUALIFIED_FEATURE_STRIDE"):
            unready[family] = "CALIBRATION_SUPPORT_INSUFFICIENT_OR_MISSING"
            continue
        centers, scales = item.get("centers"), item.get("scales")
        if (not isinstance(centers, dict) or not isinstance(scales, dict)
                or set(centers) != set(scales)
                or not all(type(number) in (int, float) and np.isfinite(float(number))
                           for number in (*centers.values(), *scales.values()))
                or not all(float(number) > 0 for number in scales.values())):
            raise ValueError("beta_personal_reference_invalid")
        result[family] = SourceReference(
            family, MappingProxyType(cast(dict[str, float], centers)),
            MappingProxyType(cast(dict[str, float], scales)),
            tuple(str(name) for name in cast(list[object], item["feature_order"])),
            str(item["calibration_sha"]), seconds, str(item["support_kind"]),
        )
    return result, unready


def pattern_models(
    values: list[dict[str, object]],
) -> dict[tuple[str, ...], dict[str, PatternModelArtifact]]:
    result: dict[tuple[str, ...], dict[str, PatternModelArtifact]] = {}
    required = {"source_set", "head", "algorithm", "threshold",
                "validation_sensitivity", "validation_far_per_hour",
                "validation_median_lead_seconds",
                "feature_schema_sha256", "feature_names", "model_uri",
                "model_sha256", "onnx_receipt"}
    for value in values:
        if set(value) != required:
            raise ValueError("beta_pattern_model_invalid")
        source_set = canonical_source_set(str(value["source_set"]).split("+"))
        head = str(value["head"])
        if head not in {"current", "onset60"}:
            raise ValueError("beta_pattern_model_invalid")
        artifact = LocalArtifact(str(value["model_uri"]), str(value["model_sha256"]))
        read_ref({"uri": artifact.uri, "sha256": artifact.sha256}, parse=False)
        result.setdefault(source_set, {})[head] = PatternModelArtifact(
            source_set, head, artifact, strict_float(value["threshold"]),
            tuple(str(name) for name in cast(list[Any], value["feature_names"])),
            str(value["feature_schema_sha256"]),
        )
    if (len(result) != 7 or any(set(heads) != {"current", "onset60"}
                                for heads in result.values())):
        raise ValueError("beta_pattern_model_set_invalid")
    return result


def read_ref(ref: dict[str, str], *, parse: bool = True) -> object:
    payload = local_path(ref["uri"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref["sha256"]:
        raise ValueError("beta_artifact_sha_mismatch")
    return json.loads(payload) if parse else payload


def contains_forbidden_key(value: object) -> bool:
    forbidden = {"truth", "truth_uri", "label", "label_uri", "event_id",
                 "teacher_phase", "seed", "split"}
    if isinstance(value, dict):
        return any(key in forbidden or contains_forbidden_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(contains_forbidden_key(item) for item in value)
    return False
