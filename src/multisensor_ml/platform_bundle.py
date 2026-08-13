"""Immutable CPU ONNX simple-reference bundles for Kidsignal sensor views."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from multisensor_ml.platform_contract import (
    REFERENCE_FEATURES,
    SENSOR_VARIANTS,
    canonical_json,
    feature_schema_for_variant,
    runtime_feature_schema_for_variant,
)

REFERENCE_BUNDLE_VERSION: Final[str] = "kidsignal-simple-reference-bundle/v1"
REFERENCE_METHODS: Final[tuple[str, ...]] = ("persistence", "rolling_median_300")


@dataclass(frozen=True, slots=True)
class ReferenceBundleResult:
    root: Path
    model_path: Path
    manifest_path: Path
    feature_schema_path: Path
    checksums_path: Path


def _require_uuid(value: str, field: str) -> str:
    from uuid import UUID

    try:
        parsed = UUID(value)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"{field} must be a UUID") from error
    if str(parsed) != value.lower():
        raise ValueError(f"{field} must use canonical UUID text")
    return str(parsed)


def _require_sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _selector_onnx(feature_count: int, feature_index: int) -> bytes:
    import onnx
    from onnx import TensorProto, helper

    input_info = helper.make_tensor_value_info(
        "features", TensorProto.FLOAT, ["N", feature_count]
    )
    output_info = helper.make_tensor_value_info("prediction", TensorProto.FLOAT, ["N", 1])
    index = helper.make_tensor("feature_index", TensorProto.INT64, [1], [feature_index])
    node = helper.make_node(
        "Gather", inputs=["features", "feature_index"], outputs=["prediction"], axis=1
    )
    graph = helper.make_graph(
        [node], "kidsignal_simple_reference_v1", [input_info], [output_info], [index]
    )
    model = helper.make_model(
        graph,
        producer_name="multisensor_ml",
        opset_imports=[helper.make_opsetid("", 15)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return cast(bytes, model.SerializeToString())


def export_simple_reference_bundle(
    *,
    output_dir: Path,
    sensor_variant: str,
    model_release_uuid: str,
    training_cohort_uuid: str,
    cohort_digest: str,
    training_mae: dict[str, float],
    validation_mae: dict[str, float],
    locked_holdout_read: bool,
    evidence_scope: str = "NON_PRODUCTION_UNSPECIFIED",
) -> ReferenceBundleResult:
    """Select a strongest simple reference from training evidence only."""

    if sensor_variant not in SENSOR_VARIANTS:
        raise ValueError(f"unsupported sensor_variant: {sensor_variant}")
    _require_uuid(model_release_uuid, "model_release_uuid")
    _require_uuid(training_cohort_uuid, "training_cohort_uuid")
    _require_sha256(cohort_digest, "cohort_digest")
    if locked_holdout_read:
        raise ValueError("locked_holdout cannot be read for reference selection")
    if set(training_mae) != set(REFERENCE_METHODS):
        raise ValueError("training_mae must contain both simple reference methods")
    if any(not isinstance(value, (int, float)) or value < 0 for value in training_mae.values()):
        raise ValueError("training_mae values must be finite and non-negative")
    selected = min(
        REFERENCE_METHODS,
        key=lambda method: (float(training_mae[method]), REFERENCE_METHODS.index(method)),
    )
    schema = feature_schema_for_variant(sensor_variant)
    runtime_schema = runtime_feature_schema_for_variant(sensor_variant)
    feature_names = list(cast(list[str], runtime_schema["ordered_feature_names"]))
    feature_name = REFERENCE_FEATURES[sensor_variant][selected]
    feature_index = feature_names.index(feature_name)

    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    model_path = root / "model.onnx"
    schema_path = root / "feature_schema.json"
    manifest_path = root / "manifest.json"
    checksums_path = root / "SHA256SUMS.json"
    model_path.write_bytes(_selector_onnx(len(feature_names), feature_index))
    schema_payload = {
        "runtime_identity": runtime_schema,
        "platform_description": {
            key: value for key, value in schema.items() if key != "canonical_json"
        },
    }
    _write_json(schema_path, schema_payload)
    manifest: dict[str, object] = {
        "bundle_version": REFERENCE_BUNDLE_VERSION,
        "model_release_uuid": model_release_uuid,
        "training_cohort_uuid": training_cohort_uuid,
        "cohort_digest": cohort_digest,
        "sensor_variant": sensor_variant,
        "evidence_scope": evidence_scope,
        "serving_contract_status": runtime_schema["training_status"],
        "training_status": runtime_schema["training_status"],
        "training_ready": runtime_schema["training_ready"],
        "reference_status": (
            "NOT VERIFIED"
            if sensor_variant == "watch_only"
            else runtime_schema["training_status"]
        ),
        "reference_method": selected,
        "reference_feature": feature_name,
        "reference_feature_index": feature_index,
        "feature_names": feature_names,
        "feature_schema_uuid": runtime_schema["schema_uuid"],
        "feature_schema_hash": runtime_schema["schema_hash"],
        "platform_description_sha256": runtime_schema[
            "platform_description_sha256"
        ],
        "selection_roles_read": ["train"],
        "validation_used_for_selection": False,
        "locked_holdout_read": False,
        "training_metrics": {key: float(training_mae[key]) for key in REFERENCE_METHODS},
        "validation_metrics_observational_only": {
            key: float(value) for key, value in sorted(validation_mae.items())
        },
        "model_path": "model.onnx",
        "model_sha256": _sha256(model_path),
        "onnx": {
            "input": {"name": "features", "dtype": "float32", "shape": [None, len(feature_names)]},
            "output": {"name": "prediction", "dtype": "float32", "shape": [None, 1]},
            "opset": 15,
            "execution_provider": "CPUExecutionProvider",
        },
        "real_data_status": "NOT VERIFIED",
        "stage": None,
        "pattern": None,
        "delivery_eligible": False,
        "promotion_eligible": False,
    }
    manifest["manifest_content_sha256"] = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    _write_json(manifest_path, manifest)
    checksums = {
        "feature_schema.json": _sha256(schema_path),
        "manifest.json": _sha256(manifest_path),
        "model.onnx": _sha256(model_path),
    }
    _write_json(checksums_path, checksums)
    return ReferenceBundleResult(root, model_path, manifest_path, schema_path, checksums_path)
