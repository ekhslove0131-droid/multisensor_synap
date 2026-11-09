"""Local deterministic center/scale JSON format; not the Cloud ONNX registry format."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from multisensor_ml.physiology_compatibility import consume_descriptors
from multisensor_ml.physiology_handoff import canonical_bytes, digest, validate_receipt


def validate_export_request(request: Mapping[str, Any]) -> dict[str, Any]:
    if set(request) != {"model_release_uuid", "generated_at"}:
        raise ValueError("EXPORT_REQUEST_FIELDS")
    value = request["model_release_uuid"]
    parsed = UUID(value)
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("EXPORT_UUID")
    stamp = request["generated_at"]
    if not isinstance(stamp, str) or not stamp.endswith("Z"):
        raise ValueError("EXPORT_UTC")
    datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return dict(request)


def _write_baseline_artifact(
    path: Path, request: Mapping[str, Any], result: Mapping[str, Any], schema: Mapping[str, Any]
) -> str:
    body = dict(
        contract_version="physiology_baseline_artifact_v1",
        export_request=validate_export_request(request),
        result=result,
        schema=schema,
    )
    document = dict(body, content_sha256=digest(body))
    encoded = canonical_bytes(document)
    with path.open("xb") as handle:
        handle.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def read_baseline_artifact(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_bytes())
    return _validate_document(document)


def _validate_document(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict) or set(document) != {
        "contract_version",
        "export_request",
        "result",
        "schema",
        "content_sha256",
    }:
        raise ValueError("ARTIFACT_FIELDS")
    if document["contract_version"] != "physiology_baseline_artifact_v1":
        raise ValueError("ARTIFACT_VERSION")
    if document["content_sha256"] != digest(
        {k: v for k, v in document.items() if k != "content_sha256"}
    ):
        raise ValueError("ARTIFACT_DIGEST")
    validate_export_request(document["export_request"])
    result = document["result"]
    if set(result) != {
        "status",
        "fit_call_count",
        "fixture_fit_call_count",
        "evaluation_status",
        "promotion_eligible",
        "delivery_eligible",
        "models",
        "validation",
        "feature_names",
        "receipt",
        "task",
        "scale_method",
        "validation_semantics",
        "real_data_status",
        "stage",
    }:
        raise ValueError("ARTIFACT_RESULT_FIELDS")
    bound = validate_receipt(result["receipt"], document["schema"])
    if result["promotion_eligible"] is not False or result["delivery_eligible"] is not False:
        raise ValueError("ARTIFACT_NOT_FOR_DELIVERY")
    expected = (
        "CONTRACT_EVIDENCE_ONLY"
        if bound["origin"] == "CONTRACT_FIXTURE"
        else ("TRAINED_NOT_EVALUATED")
    )
    if (
        result["status"] != expected
        or result["feature_names"] != document["schema"]["core_features"]
    ):
        raise ValueError("ARTIFACT_MODEL_CONTRACT")
    if (result["fit_call_count"], result["fixture_fit_call_count"]) != (
        (0, 1) if bound["origin"] == "CONTRACT_FIXTURE" else (1, 0)
    ):
        raise ValueError("ARTIFACT_FIT_ORIGIN")
    width = len(result["feature_names"])
    subjects = set()
    for model in result["models"]:
        if set(model) != {"training_subject_uuid", "center", "scale", "train_row_count"}:
            raise ValueError("ARTIFACT_MODEL_FIELDS")
        subject = model["training_subject_uuid"]
        if subject in subjects or UUID(subject).version != 5:
            raise ValueError("ARTIFACT_SUBJECT")
        subjects.add(subject)
        for key in ("center", "scale"):
            if len(model[key]) != width or not all(
                type(v) in (int, float) and math.isfinite(v) for v in model[key]
            ):
                raise ValueError("ARTIFACT_PARAMETERS")
        if any(v <= 0 for v in model["scale"]):
            raise ValueError("ARTIFACT_SCALE")
    if not subjects:
        raise ValueError("ARTIFACT_EMPTY")
    return document


def residual(
    artifact: Mapping[str, Any], training_subject_uuid: str, values: Sequence[float]
) -> tuple[float, ...]:
    """Apply a verified local artifact; no fallback for unseen people."""
    artifact = _validate_document(dict(artifact))
    schema = artifact["schema"]
    if len(values) != len(schema["core_features"]):
        raise ValueError("INPUT_WIDTH")
    specs = {d["code"]: d for d in schema["descriptors"]}
    vector = consume_descriptors(
        schema,
        [
            dict(
                code=name, unit=specs[name]["unit"], value=value, readiness="AVAILABLE", reason=None
            )
            for name, value in zip(schema["core_features"], values, strict=True)
        ],
    )
    if vector is None:
        raise ValueError("INPUT_INVALID")
    for model in artifact["result"]["models"]:
        if model["training_subject_uuid"] == training_subject_uuid:
            output = tuple(
                (v - c) / s for v, c, s in zip(vector, model["center"], model["scale"], strict=True)
            )
            if not all(math.isfinite(v) for v in output):
                raise ValueError("RESIDUAL_NONFINITE")
            return output
    raise ValueError("SUBJECT_BASELINE_UNAVAILABLE")
