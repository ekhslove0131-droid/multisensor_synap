"""Receipt-bound standard candidate selection and immutable CPU ONNX export.

The public coordinator accepts one model-sync receipt and a read-only BigQuery
client.  It does not accept arrays, cohort UUIDs, or digests separately.  Only
the internally prepared standard dataset may reach candidate fitting.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast
from uuid import UUID

import numpy as np

from multisensor_ml.bigquery_sdk_reader import (
    BigQueryClientProtocol,
    QueryJobConfigFactory,
)
from multisensor_ml.model_sync_receipt import ModelSyncReceipt, parse_model_sync_receipt
from multisensor_ml.observational_contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    canonical_feature_schema,
)
from multisensor_ml.platform_contract import label_review_schema
from multisensor_ml.standard_baseline_bigquery import (
    ELIGIBILITY_POLICY,
    EXPECTED_MODEL_READER,
    RUNTIME_TO_TRAINER_INDICES,
    STANDARD_BASELINE_VIEW_FIELDS,
    STANDARD_LEAKAGE_PROBE_CASE_IDS,
    STANDARD_TRAINER_ENTRYPOINT,
    STANDARD_TRAINER_FEATURE_NAMES,
    SUPPORTED_STANDARD_VERSION,
    TARGET_LEAKAGE_STATUS,
    TARGET_SOURCE_FEATURE,
    TARGET_SOURCE_RUNTIME_INDEX,
    WATCH_SCHEMA_HASH,
    WATCH_SCHEMA_UUID,
    PreparedStandardDataset,
    audit_standard_target_leakage,
    build_standard_training_projection,
    run_read_only_standard_cohort_sdk_reader,
)

EXPORT_REQUEST_VERSION: Final[str] = "kidsignal-standard-candidate-export-request/v1"
BUNDLE_SCHEMA_VERSION: Final[str] = "kidsignal-immutable-model-bundle/v1"
SHA256SUMS_VERSION: Final[str] = "kidsignal-sha256sums/v1"
GOLDEN_FIXTURE_VERSION: Final[str] = "kidsignal-model-golden-fixture/v1"
GOLDEN_OUTPUT_VERSION: Final[str] = "kidsignal-model-golden-output/v1"
SEALED_REQUEST_VERSION: Final[str] = "kidsignal-sealed-evaluation-request/v1"
SEALED_EVALUATOR_VERSION: Final[str] = "kidsignal-sealed-evaluator/v1"
PAYLOAD_DIGEST_VERSION: Final[str] = "kidsignal-bundle-payload-digest/v1"
RUNTIME_VERSION: Final[str] = "kidsignal-model-runtime/v1"
PREREGISTERED_CANDIDATES: Final[tuple[str, ...]] = (
    "mean_constant_v1",
    "ridge_l2_alpha_1_v1",
)
ARCHIVE_MEMBERS: Final[frozenset[str]] = frozenset(
    {
        "SHA256SUMS.json",
        "feature_schema.json",
        "golden_fixture.json",
        "golden_output.json",
        "label_review_schema.json",
        "manifest.json",
        "model.onnx",
        "runtime.json",
        "runtime_requirements.txt",
        "sealed_evaluation_request.json",
        "split_manifest.json",
        "training_evidence.json",
        "training_projection.json",
    }
)
EXPORT_REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "evidence_status",
        "model_sync_handoff_digest",
        "standard_cohort_uuid",
        "public_cohort_digest",
        "public_split_digest",
        "feature_schema_uuid",
        "feature_schema_hash",
        "target_version",
        "model_release_uuid",
        "model_release",
        "fixture_uuid",
        "evaluation_uuid",
        "sealed_evaluation_input_uuid",
        "sealed_evaluation_input_digest",
        "reference_model_release_uuid",
        "reference_artifact_sha256",
        "created_at",
        "generated_at",
        "requested_metrics",
        "request_sha256",
    }
)
_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_SPLIT_POLICY_MAP: Final[dict[str, str]] = {
    "PERSON_GROUP": "person_group",
    "CHRONOLOGICAL_PER_SUBJECT": "chronological_per_subject",
}


@dataclass(frozen=True, slots=True)
class CandidateCoefficients:
    candidate_id: str
    coefficients: tuple[float, ...]
    intercept: float


class CandidateCoefficientProvider(Protocol):
    performs_real_fit: bool

    def provide(
        self,
        candidate_id: str,
        train_features: np.ndarray,
        train_targets: np.ndarray,
    ) -> CandidateCoefficients: ...


@dataclass(frozen=True, slots=True)
class StandardCandidateExportRequest:
    evidence_status: str
    model_sync_handoff_digest: str
    standard_cohort_uuid: str
    public_cohort_digest: str
    public_split_digest: str
    feature_schema_uuid: str
    feature_schema_hash: str
    target_version: str
    model_release_uuid: str
    model_release: str
    fixture_uuid: str
    evaluation_uuid: str
    sealed_evaluation_input_uuid: str
    sealed_evaluation_input_digest: str
    reference_model_release_uuid: str
    reference_artifact_sha256: str
    created_at: str
    generated_at: str
    requested_metrics: tuple[str, ...]
    request_sha256: str


@dataclass(frozen=True, slots=True)
class StandardCandidateExportResult:
    archive_path: Path
    archive_sha256: str
    selected_candidate: str
    metrics: dict[str, dict[str, float]]
    evidence_status: str
    fit_call_count: int
    candidate_provider_call_count: int
    training_row_count: int
    validation_row_count: int
    model_sync_handoff_digest: str


class _PreregisteredCandidateProvider:
    """Real fitting provider, reachable only after the receipt-bound gate."""

    performs_real_fit = True

    def provide(
        self,
        candidate_id: str,
        train_features: np.ndarray,
        train_targets: np.ndarray,
    ) -> CandidateCoefficients:
        if candidate_id == "mean_constant_v1":
            return CandidateCoefficients(
                candidate_id,
                (0.0,) * len(STANDARD_TRAINER_FEATURE_NAMES),
                float(np.mean(train_targets, dtype=np.float64)),
            )
        if candidate_id == "ridge_l2_alpha_1_v1":
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0, fit_intercept=True)
            model.fit(train_features, train_targets)
            coefficients = tuple(float(value) for value in np.asarray(model.coef_).ravel())
            return CandidateCoefficients(candidate_id, coefficients, float(model.intercept_))
        raise ValueError(f"unknown preregistered candidate: {candidate_id}")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_uuid(value: object, field: str, *, version: int | None = None) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical UUID")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a canonical UUID") from error
    if str(parsed) != value:
        raise ValueError(f"{field} must be a canonical UUID")
    if version is not None and parsed.version != version:
        raise ValueError(f"{field} must be UUIDv{version}")
    return value


def _require_opaque(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None
    ):
        raise ValueError(f"{field} must be bounded opaque text")
    return value


def _require_utc(value: object, field: str) -> str:
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise ValueError(f"{field} must be UTC seconds text")
    return value


def parse_standard_candidate_export_request(
    payload: object,
) -> StandardCandidateExportRequest:
    if not isinstance(payload, Mapping):
        raise ValueError("standard export request must be an object")
    values = {str(key): value for key, value in payload.items()}
    missing = sorted(EXPORT_REQUEST_FIELDS.difference(values))
    if missing:
        raise ValueError(f"standard export request missing field {missing[0]}")
    extra = sorted(set(values).difference(EXPORT_REQUEST_FIELDS))
    if extra:
        raise ValueError(f"standard export request has unknown field {extra[0]}")
    supplied_hash = _require_digest(values["request_sha256"], "request_sha256")
    unsigned = {key: value for key, value in values.items() if key != "request_sha256"}
    if _semantic_hash(unsigned) != supplied_hash:
        raise ValueError("request_sha256 does not match standard export request")
    if values["schema_version"] != EXPORT_REQUEST_VERSION:
        raise ValueError("unsupported standard export request schema_version")
    evidence_status = str(values["evidence_status"])
    if evidence_status not in {"CONTRACT_EVIDENCE_ONLY", "REAL_FROZEN_COHORT"}:
        raise ValueError("evidence_status is unsupported")
    metrics = values["requested_metrics"]
    if (
        not isinstance(metrics, list)
        or not metrics
        or len(metrics) != len(set(metrics))
        or any(_require_opaque(metric, "requested_metrics") != metric for metric in metrics)
    ):
        raise ValueError("requested_metrics must be unique opaque values")
    target_version = _require_opaque(values["target_version"], "target_version")
    if target_version != SUPPORTED_STANDARD_VERSION:
        raise ValueError("target_version must match the stable standard version")
    return StandardCandidateExportRequest(
        evidence_status=evidence_status,
        model_sync_handoff_digest=_require_digest(
            values["model_sync_handoff_digest"], "model_sync_handoff_digest"
        ),
        standard_cohort_uuid=_require_uuid(
            values["standard_cohort_uuid"], "standard_cohort_uuid", version=4
        ),
        public_cohort_digest=_require_digest(
            values["public_cohort_digest"], "public_cohort_digest"
        ),
        public_split_digest=_require_digest(
            values["public_split_digest"], "public_split_digest"
        ),
        feature_schema_uuid=_require_uuid(
            values["feature_schema_uuid"], "feature_schema_uuid"
        ),
        feature_schema_hash=_require_digest(
            values["feature_schema_hash"], "feature_schema_hash"
        ),
        target_version=target_version,
        model_release_uuid=_require_uuid(
            values["model_release_uuid"], "model_release_uuid"
        ),
        model_release=_require_opaque(values["model_release"], "model_release"),
        fixture_uuid=_require_uuid(values["fixture_uuid"], "fixture_uuid"),
        evaluation_uuid=_require_uuid(values["evaluation_uuid"], "evaluation_uuid"),
        sealed_evaluation_input_uuid=_require_uuid(
            values["sealed_evaluation_input_uuid"], "sealed_evaluation_input_uuid"
        ),
        sealed_evaluation_input_digest=_require_digest(
            values["sealed_evaluation_input_digest"],
            "sealed_evaluation_input_digest",
        ),
        reference_model_release_uuid=_require_uuid(
            values["reference_model_release_uuid"], "reference_model_release_uuid"
        ),
        reference_artifact_sha256=_require_digest(
            values["reference_artifact_sha256"], "reference_artifact_sha256"
        ),
        created_at=_require_utc(values["created_at"], "created_at"),
        generated_at=_require_utc(values["generated_at"], "generated_at"),
        requested_metrics=tuple(cast(list[str], metrics)),
        request_sha256=supplied_hash,
    )


def _validate_prepared_standard_dataset(prepared: PreparedStandardDataset) -> None:
    if not isinstance(prepared, PreparedStandardDataset):
        raise TypeError("internal trainer requires PreparedStandardDataset")
    if prepared.task != "stable_standard_regression":
        raise ValueError("prepared standard task is not canonical")
    if prepared.trainer_entrypoint != STANDARD_TRAINER_ENTRYPOINT:
        raise ValueError("prepared trainer entrypoint is not canonical")
    if prepared.fit_call_count != 0:
        raise ValueError("prepared dataset must precede fitting")
    _require_uuid(prepared.standard_cohort_uuid, "standard_cohort_uuid", version=4)
    for field, value in (
        ("cohort_digest", prepared.cohort_digest),
        ("split_digest", prepared.split_digest),
        ("public_cohort_digest", prepared.public_cohort_digest),
        ("public_split_digest", prepared.public_split_digest),
        ("canonical_handoff_digest", prepared.canonical_handoff_digest),
    ):
        _require_digest(value, field)
    if prepared.stable_standard_version != SUPPORTED_STANDARD_VERSION:
        raise ValueError("prepared stable standard version is unsupported")
    if (
        prepared.feature_schema_uuid != WATCH_SCHEMA_UUID
        or prepared.feature_schema_hash != WATCH_SCHEMA_HASH
    ):
        raise ValueError("prepared feature schema is not canonical Watch")
    if prepared.split_policy not in _SPLIT_POLICY_MAP:
        raise ValueError("prepared split policy is unsupported")
    if prepared.purge_seconds != 1800:
        raise ValueError("immutable standard bundle requires purge_seconds=1800")
    if prepared.eligibility_policy != ELIGIBILITY_POLICY:
        raise ValueError("prepared eligibility policy is not real no-pattern")
    if prepared.feature_names != tuple(FEATURE_NAMES):
        raise ValueError("prepared runtime feature order is not canonical")
    if prepared.trainer_feature_names != STANDARD_TRAINER_FEATURE_NAMES:
        raise ValueError("prepared trainer feature order is not canonical")
    if prepared.runtime_to_trainer_indices != RUNTIME_TO_TRAINER_INDICES:
        raise ValueError("prepared runtime projection is not canonical")
    if (
        prepared.target_source_feature != TARGET_SOURCE_FEATURE
        or prepared.target_source_runtime_index != TARGET_SOURCE_RUNTIME_INDEX
        or prepared.target_leakage_status != TARGET_LEAKAGE_STATUS
    ):
        raise ValueError("prepared target leakage contract is not canonical")
    arrays = (
        (prepared.runtime_train_features, len(FEATURE_NAMES), "runtime_train_features"),
        (prepared.train_features, len(STANDARD_TRAINER_FEATURE_NAMES), "train_features"),
        (
            prepared.runtime_validation_features,
            len(FEATURE_NAMES),
            "runtime_validation_features",
        ),
        (
            prepared.validation_features,
            len(STANDARD_TRAINER_FEATURE_NAMES),
            "validation_features",
        ),
    )
    for values, width, field in arrays:
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.float32
            or values.ndim != 2
            or values.shape[0] <= 0
            or values.shape[1] != width
            or not bool(np.isfinite(values).all())
            or values.flags.writeable
        ):
            raise ValueError(f"{field} violates prepared dataset contract")
    for values, rows, field in (
        (prepared.train_targets, prepared.train_features.shape[0], "train_targets"),
        (
            prepared.validation_targets,
            prepared.validation_features.shape[0],
            "validation_targets",
        ),
    ):
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.float32
            or values.shape != (rows,)
            or not bool(np.isfinite(values).all())
            or values.flags.writeable
        ):
            raise ValueError(f"{field} violates prepared dataset contract")
    train_rows = prepared.train_features.shape[0]
    validation_rows = prepared.validation_features.shape[0]
    metadata = (
        (prepared.train_subject_groups, train_rows, "train_subject_groups"),
        (prepared.train_sequence_groups, train_rows, "train_sequence_groups"),
        (prepared.train_row_ids, train_rows, "train_row_ids"),
        (
            prepared.validation_subject_groups,
            validation_rows,
            "validation_subject_groups",
        ),
        (
            prepared.validation_sequence_groups,
            validation_rows,
            "validation_sequence_groups",
        ),
        (prepared.validation_row_ids, validation_rows, "validation_row_ids"),
    )
    for metadata_values, rows, field in metadata:
        if len(metadata_values) != rows or any(
            not value for value in metadata_values
        ):
            raise ValueError(f"{field} does not match prepared rows")
    if set(prepared.train_row_ids).intersection(prepared.validation_row_ids):
        raise ValueError("prepared TRAIN and VALIDATION row IDs overlap")
    if prepared.split_policy == "PERSON_GROUP" and set(
        prepared.train_subject_groups
    ).intersection(prepared.validation_subject_groups):
        raise ValueError("prepared person-group split leaks subjects")
    if not np.array_equal(
        prepared.train_features,
        prepared.runtime_train_features[:, RUNTIME_TO_TRAINER_INDICES],
    ) or not np.array_equal(
        prepared.validation_features,
        prepared.runtime_validation_features[:, RUNTIME_TO_TRAINER_INDICES],
    ):
        raise ValueError("prepared 16-to-15 projection bytes do not match")
    audit_standard_target_leakage(prepared)


def _validate_receipt_binding(
    receipt: ModelSyncReceipt,
    readiness: Mapping[str, object],
    prepared: PreparedStandardDataset,
) -> None:
    expected = {
        "standard_cohort_uuid": receipt.cohort_uuid,
        "public_cohort_digest": receipt.public_cohort_digest,
        "public_split_digest": receipt.public_split_digest,
        "feature_schema_uuid": receipt.feature_schema_uuid,
        "feature_schema_hash": receipt.feature_schema_hash,
        "split_policy": receipt.split_policy,
        "purge_seconds": receipt.purge_seconds,
    }
    for field, value in expected.items():
        actual = (
            getattr(prepared, field)
            if hasattr(prepared, field)
            else readiness.get(field)
        )
        if actual != value:
            raise ValueError(f"receipt binding mismatch: {field}")
    if (
        readiness.get("status") != "READY_FOR_SYNC_NOT_TRAINED"
        or readiness.get("training_ready") is not True
        or readiness.get("locked_access") is not False
        or readiness.get("fit_call_count") != 0
        or readiness.get("model_reader_identity_verified") is not True
        or readiness.get("observed_principal") != EXPECTED_MODEL_READER
        or readiness.get("schema_fields") != list(STANDARD_BASELINE_VIEW_FIELDS)
    ):
        raise ValueError("readiness receipt is not safe for standard fitting")
    split_counts = readiness.get("split_counts")
    if not isinstance(split_counts, Mapping):
        raise ValueError("readiness receipt split_counts are missing")
    if (
        split_counts.get("TRAIN") != receipt.train_row_count
        or split_counts.get("VALIDATION") != receipt.validation_row_count
        or prepared.train_features.shape[0] != receipt.train_row_count
        or prepared.validation_features.shape[0] != receipt.validation_row_count
        or receipt.authorized_field_count != len(STANDARD_BASELINE_VIEW_FIELDS)
    ):
        raise ValueError("receipt binding mismatch: split row counts")


def _bind_export_request(
    request: StandardCandidateExportRequest,
    receipt: ModelSyncReceipt,
    prepared: PreparedStandardDataset | None,
) -> None:
    expected = {
        "model_sync_handoff_digest": receipt.handoff_digest,
        "standard_cohort_uuid": receipt.cohort_uuid,
        "public_cohort_digest": receipt.public_cohort_digest,
        "public_split_digest": receipt.public_split_digest,
        "feature_schema_uuid": receipt.feature_schema_uuid,
        "feature_schema_hash": receipt.feature_schema_hash,
    }
    for field, value in expected.items():
        if getattr(request, field) != value:
            raise ValueError(f"export request {field} does not match receipt")
    if prepared is not None and request.target_version != prepared.stable_standard_version:
        raise ValueError("export request target_version does not match prepared standard")


def _validate_coefficients(value: CandidateCoefficients, candidate_id: str) -> None:
    if not isinstance(value, CandidateCoefficients) or value.candidate_id != candidate_id:
        raise ValueError("candidate provider returned the wrong candidate identity")
    if len(value.coefficients) != len(STANDARD_TRAINER_FEATURE_NAMES):
        raise ValueError("candidate coefficients must contain 15 values")
    if not all(math.isfinite(float(item)) for item in value.coefficients) or not math.isfinite(
        float(value.intercept)
    ):
        raise ValueError("candidate coefficients must be finite")


def _predict(coefficients: CandidateCoefficients, features: np.ndarray) -> np.ndarray:
    weights = np.asarray(coefficients.coefficients, dtype=np.float32)
    prediction = features @ weights + np.float32(coefficients.intercept)
    return np.asarray(prediction, dtype=np.float32)


def _select_candidate(
    prepared: PreparedStandardDataset,
    provider: CandidateCoefficientProvider,
) -> tuple[CandidateCoefficients, dict[str, dict[str, float]]]:
    candidates: dict[str, CandidateCoefficients] = {}
    metrics: dict[str, dict[str, float]] = {}
    for candidate_id in PREREGISTERED_CANDIDATES:
        value = provider.provide(
            candidate_id,
            prepared.train_features,
            prepared.train_targets,
        )
        _validate_coefficients(value, candidate_id)
        candidates[candidate_id] = value
        prediction = _predict(value, prepared.validation_features)
        residual = prediction.astype(np.float64) - prepared.validation_targets.astype(
            np.float64
        )
        metrics[candidate_id] = {
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        }
    selected_id = min(
        PREREGISTERED_CANDIDATES,
        key=lambda candidate_id: (
            metrics[candidate_id]["mae"],
            metrics[candidate_id]["rmse"],
            PREREGISTERED_CANDIDATES.index(candidate_id),
        ),
    )
    return candidates[selected_id], metrics


def _linear_onnx(coefficients: CandidateCoefficients) -> bytes:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as error:
        raise RuntimeError("install the onnx-serving extra to export the bundle") from error
    input_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", 16])
    output_info = helper.make_tensor_value_info("prediction", TensorProto.FLOAT, ["N", 1])
    indices = helper.make_tensor(
        "trainer_indices",
        TensorProto.INT64,
        [15],
        list(RUNTIME_TO_TRAINER_INDICES),
    )
    weights = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        [15, 1],
        [float(value) for value in coefficients.coefficients],
    )
    intercept = helper.make_tensor(
        "intercept", TensorProto.FLOAT, [1], [float(coefficients.intercept)]
    )
    nodes = [
        helper.make_node(
            "Gather", ["features", "trainer_indices"], ["trainer_features"], axis=1
        ),
        helper.make_node("MatMul", ["trainer_features", "weights"], ["linear"]),
        helper.make_node("Add", ["linear", "intercept"], ["prediction"]),
    ]
    graph = helper.make_graph(
        nodes,
        "kidsignal_stable_standard_candidate_v1",
        [input_info],
        [output_info],
        [indices, weights, intercept],
    )
    model = helper.make_model(
        graph,
        producer_name="multisensor_ml",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return cast(bytes, model.SerializeToString())


def _run_onnx(model: bytes, features: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("install the onnx-serving extra to verify the bundle") from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    output = session.run(["prediction"], {"features": features.astype(np.float32)})[0]
    return np.asarray(output, dtype=np.float32)


def _payload_digest(files: Mapping[str, bytes]) -> str:
    material = {
        "digest_version": PAYLOAD_DIGEST_VERSION,
        "files": {
            name: _sha256_bytes(content)
            for name, content in sorted(files.items())
            if name not in {"SHA256SUMS.json", "sealed_evaluation_request.json"}
        },
    }
    return _semantic_hash(material)


def _build_files(
    prepared: PreparedStandardDataset,
    request: StandardCandidateExportRequest,
    selected: CandidateCoefficients,
) -> dict[str, bytes]:
    model = _linear_onnx(selected)
    model_hash = _sha256_bytes(model)
    feature_document = canonical_feature_schema()
    label_document = label_review_schema()
    feature_bytes = _canonical_bytes(feature_document)
    label_bytes = _canonical_bytes(label_document)
    feature_hash = _semantic_hash(feature_document)
    label_hash = _semantic_hash(label_document)
    if feature_hash != prepared.feature_schema_hash or feature_hash != FEATURE_SCHEMA_SHA256:
        raise ValueError("serialized feature schema hash does not match prepared schema")
    label_uuid = _require_uuid(label_document.get("schema_uuid"), "label_review_schema_uuid")
    split_policy = _SPLIT_POLICY_MAP.get(prepared.split_policy)
    if split_policy is None:
        raise ValueError("prepared split policy cannot be mapped to bundle contract")
    runtime_packages = {
        "numpy": importlib.metadata.version("numpy"),
        "onnxruntime": importlib.metadata.version("onnxruntime"),
    }
    runtime = {
        "schema_version": RUNTIME_VERSION,
        "python": platform.python_version(),
        "packages": runtime_packages,
        "onnx_providers": ["CPUExecutionProvider"],
        "onnx_intra_op_threads": 1,
        "onnx_inter_op_threads": 1,
    }
    requirements = "".join(
        f"{name}=={runtime_packages[name]}\n" for name in sorted(runtime_packages)
    ).encode("utf-8")
    projection = build_standard_training_projection(prepared)
    base_values = {name: float(index + 1) / 100.0 for index, name in enumerate(FEATURE_NAMES)}
    mutated_values = dict(base_values)
    mutated_values[TARGET_SOURCE_FEATURE] = 9.99
    fixture_cases = []
    for case_id, values in zip(
        STANDARD_LEAKAGE_PROBE_CASE_IDS,
        (base_values, mutated_values),
        strict=True,
    ):
        fixture_cases.append(
            {
                "case_id": case_id,
                "feature_values": values,
                "expected_input_vector": [values[name] for name in FEATURE_NAMES],
                "expected_runtime_action": "INFER",
                "expected_reason": None,
            }
        )
    runtime_input = np.asarray(
        [case["expected_input_vector"] for case in fixture_cases], dtype=np.float32
    )
    predictions = _run_onnx(model, runtime_input)
    if predictions.shape != (2, 1) or predictions[0].tobytes() != predictions[1].tobytes():
        raise ValueError("paired target-source probe must have byte-exact predictions")
    fixture = {
        "schema_version": GOLDEN_FIXTURE_VERSION,
        "fixture_uuid": request.fixture_uuid,
        "fixture_kind": "SYNTHETIC_NON_PERSONAL",
        "source_variant": "watch_only",
        "source_set": ["watch"],
        "feature_schema_uuid": prepared.feature_schema_uuid,
        "feature_schema_hash": feature_hash,
        "input_name": "features",
        "input_dtype": "float32",
        "feature_count": len(FEATURE_NAMES),
        "ordered_feature_names": list(FEATURE_NAMES),
        "cases": fixture_cases,
    }
    fixture_bytes = _canonical_bytes(fixture)
    output_cases = [
        {
            "case_id": case_id,
            "expected_runtime_action": "INFER",
            "expected_output": [[float(predictions[0, 0])]],
            "expected_status": "PREDICTED",
            "expected_reason": None,
        }
        for case_id in STANDARD_LEAKAGE_PROBE_CASE_IDS
    ]
    golden_output = {
        "schema_version": GOLDEN_OUTPUT_VERSION,
        "model_release_uuid": request.model_release_uuid,
        "model_artifact_sha256": model_hash,
        "fixture_uuid": request.fixture_uuid,
        "golden_fixture_sha256": _sha256_bytes(fixture_bytes),
        "input_name": "features",
        "output_name": "prediction",
        "output_dtype": "float32",
        "output_shape": ["N", 1],
        "tolerance": {
            "absolute": 1e-6,
            "relative": 1e-5,
            "non_finite_allowed": False,
        },
        "cases": output_cases,
    }
    manifest: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "contract_status": "FROZEN_CONTRACT",
        "bundle_role": "CANDIDATE",
        "model_release_uuid": request.model_release_uuid,
        "model_release": request.model_release,
        "model_status": "TRAINED_NOT_EVALUATED",
        "source_variant": "watch_only",
        "source_set": ["watch"],
        "task": prepared.task,
        "target_version": prepared.stable_standard_version,
        "feature_schema_uuid": prepared.feature_schema_uuid,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_hash,
        "label_review_schema_uuid": label_uuid,
        "label_review_schema_hash": label_hash,
        "label_schema_uuid": label_uuid,
        "label_schema_hash": label_hash,
        "review_schema_uuid": label_uuid,
        "review_schema_hash": label_hash,
        "training_cohort_uuid": prepared.standard_cohort_uuid,
        "cohort_digest": prepared.cohort_digest,
        "split_digest": prepared.split_digest,
        "split_policy": split_policy,
        "purge_seconds": prepared.purge_seconds,
        "locked_holdout_read": False,
        "threshold_lineage": {
            "status": "NOT_FROZEN",
            "threshold_set_uuid": None,
            "calibration_split_id": None,
            "thresholds_sha256": None,
            "evaluation_labels_used": False,
        },
        "adapter_lineage": {
            "status": "NOT_APPLICABLE",
            "adapter_uuid": None,
            "parent_model_release_uuid": None,
            "artifact_sha256": None,
            "selection_split_role": None,
        },
        "onnx": {
            "path": "model.onnx",
            "sha256": model_hash,
            "input_name": "features",
            "input_dtype": "float32",
            "input_shape": ["N", 16],
            "output_name": "prediction",
            "output_dtype": "float32",
            "output_shape": ["N", 1],
            "opset": 17,
            "execution_provider": "CPUExecutionProvider",
        },
        "real_data_status": "NOT VERIFIED",
        "stage": None,
        "delivery_eligible": False,
        "promotion_eligible": False,
        "sealed_evaluation_status": "NOT_REQUESTED",
        "created_at": request.created_at,
        "producer": "multisensor_ml",
    }
    manifest["manifest_content_sha256"] = _semantic_hash(manifest)
    files: dict[str, bytes] = {
        "feature_schema.json": feature_bytes,
        "golden_fixture.json": fixture_bytes,
        "golden_output.json": _canonical_bytes(golden_output),
        "label_review_schema.json": label_bytes,
        "manifest.json": _canonical_bytes(manifest),
        "model.onnx": model,
        "runtime.json": _canonical_bytes(runtime),
        "runtime_requirements.txt": requirements,
        "split_manifest.json": _canonical_bytes(
            {
                "training_cohort_uuid": prepared.standard_cohort_uuid,
                "cohort_digest": prepared.cohort_digest,
                "split_digest": prepared.split_digest,
            }
        ),
        "training_evidence.json": _canonical_bytes(
            {
                "training_status": "TRAINED_NOT_EVALUATED",
                "real_cohort_rows": int(
                    prepared.train_features.shape[0]
                    + prepared.validation_features.shape[0]
                ),
            }
        ),
        "training_projection.json": _canonical_bytes(projection),
    }
    sealed: dict[str, object] = {
        "schema_version": SEALED_REQUEST_VERSION,
        "evaluator_contract_version": SEALED_EVALUATOR_VERSION,
        "evaluation_uuid": request.evaluation_uuid,
        "sealed_evaluation_input_uuid": request.sealed_evaluation_input_uuid,
        "sealed_evaluation_input_digest": request.sealed_evaluation_input_digest,
        "model_release_uuid": request.model_release_uuid,
        "model_artifact_sha256": model_hash,
        "bundle_payload_digest": _payload_digest(files),
        "source_variant": "watch_only",
        "source_set": ["watch"],
        "task": prepared.task,
        "target_version": prepared.stable_standard_version,
        "feature_schema_uuid": prepared.feature_schema_uuid,
        "feature_schema_hash": feature_hash,
        "training_cohort_uuid": prepared.standard_cohort_uuid,
        "cohort_digest": prepared.cohort_digest,
        "split_digest": prepared.split_digest,
        "reference_model_release_uuid": request.reference_model_release_uuid,
        "reference_artifact_sha256": request.reference_artifact_sha256,
        "adapter_uuid": None,
        "adapter_artifact_sha256": None,
        "threshold_set_uuid": None,
        "thresholds_sha256": None,
        "requested_metrics": list(request.requested_metrics),
        "locked_rows_in_bundle": False,
        "row_level_labels_visible": False,
        "generated_at": request.generated_at,
    }
    sealed["request_sha256"] = _semantic_hash(sealed)
    files["sealed_evaluation_request.json"] = _canonical_bytes(sealed)
    sums: dict[str, object] = {
        "schema_version": SHA256SUMS_VERSION,
        "files": {
            name: _sha256_bytes(content) for name, content in sorted(files.items())
        },
    }
    sums["sha256sums_content_sha256"] = _semantic_hash(sums)
    files["SHA256SUMS.json"] = _canonical_bytes(sums)
    if set(files) != ARCHIVE_MEMBERS:
        raise RuntimeError("immutable archive member set drifted")
    return files


def _write_archive_create_only(path: Path, files: Mapping[str, bytes]) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.extra = b""
            entry.comment = b""
            archive.writestr(
                entry,
                content,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(output.getvalue())


def _export_internal(
    prepared: PreparedStandardDataset,
    request: StandardCandidateExportRequest,
    output_path: Path,
    provider: CandidateCoefficientProvider,
) -> StandardCandidateExportResult:
    _validate_prepared_standard_dataset(prepared)
    if type(provider.performs_real_fit) is not bool:
        raise ValueError("candidate provider fit authority must be boolean")
    if request.evidence_status == "CONTRACT_EVIDENCE_ONLY" and provider.performs_real_fit:
        raise ValueError("contract evidence must not invoke a real estimator fit")
    if request.evidence_status == "REAL_FROZEN_COHORT" and not provider.performs_real_fit:
        raise ValueError("real frozen cohort export requires the preregistered real trainer")
    selected, metrics = _select_candidate(prepared, provider)
    files = _build_files(prepared, request, selected)
    _write_archive_create_only(output_path, files)
    verified = verify_standard_candidate_bundle(output_path)
    if verified["status"] != "VERIFIED_IMMUTABLE_STANDARD_CANDIDATE":
        raise RuntimeError("model-side immutable bundle verification failed")
    return StandardCandidateExportResult(
        archive_path=output_path.resolve(),
        archive_sha256=_sha256_bytes(output_path.read_bytes()),
        selected_candidate=selected.candidate_id,
        metrics=metrics,
        evidence_status=request.evidence_status,
        fit_call_count=(len(PREREGISTERED_CANDIDATES) if provider.performs_real_fit else 0),
        candidate_provider_call_count=len(PREREGISTERED_CANDIDATES),
        training_row_count=int(prepared.train_features.shape[0]),
        validation_row_count=int(prepared.validation_features.shape[0]),
        model_sync_handoff_digest=request.model_sync_handoff_digest,
    )


def run_receipt_bound_standard_candidate_export(
    *,
    model_sync_receipt: object,
    client: BigQueryClientProtocol,
    observed_at_utc: str,
    observed_principal: str,
    export_request: object,
    output_path: Path,
    candidate_provider: CandidateCoefficientProvider | None = None,
    job_config_factory: QueryJobConfigFactory | None = None,
) -> StandardCandidateExportResult:
    """Read one receipt-selected standard cohort and export one candidate bundle."""

    receipt = parse_model_sync_receipt(model_sync_receipt)
    if receipt.plane != "standard":
        raise ValueError("standard candidate export requires a standard receipt")
    request = parse_standard_candidate_export_request(export_request)
    _bind_export_request(request, receipt, None)
    if request.evidence_status == "CONTRACT_EVIDENCE_ONLY":
        if (
            candidate_provider is None
            or getattr(candidate_provider, "performs_real_fit", None) is not False
        ):
            raise ValueError("contract evidence requires an explicit non-fit provider")
        provider = candidate_provider
    else:
        if candidate_provider is not None:
            raise ValueError("real frozen cohort forbids an external candidate provider")
        provider = _PreregisteredCandidateProvider()
    if output_path.exists():
        raise FileExistsError(f"immutable bundle already exists: {output_path}")
    result = run_read_only_standard_cohort_sdk_reader(
        client=client,
        standard_cohort_uuid=receipt.cohort_uuid,
        expected_public_cohort_digest=receipt.public_cohort_digest,
        expected_public_split_digest=receipt.public_split_digest,
        observed_at_utc=observed_at_utc,
        observed_principal=observed_principal,
        job_config_factory=job_config_factory,
    )
    if result.prepared_dataset is None:
        raise RuntimeError(str(result.receipt.get("status", "BLOCKED_STANDARD_INTAKE")))
    prepared = result.prepared_dataset
    _validate_receipt_binding(receipt, result.receipt, prepared)
    _validate_prepared_standard_dataset(prepared)
    _bind_export_request(request, receipt, prepared)
    return _export_internal(prepared, request, output_path, provider)


def _read_json(files: Mapping[str, bytes], name: str) -> dict[str, object]:
    try:
        value = json.loads(files[name])
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid immutable bundle member: {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid immutable bundle document: {name}")
    return cast(dict[str, object], value)


def verify_standard_candidate_bundle(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != ARCHIVE_MEMBERS:
            raise ValueError("immutable archive member set is invalid")
        files = {name: archive.read(name) for name in names}
    sums = _read_json(files, "SHA256SUMS.json")
    unsigned_sums = dict(sums)
    sums_hash = unsigned_sums.pop("sha256sums_content_sha256", None)
    if (
        set(unsigned_sums) != {"schema_version", "files"}
        or unsigned_sums["schema_version"] != SHA256SUMS_VERSION
        or sums_hash != _semantic_hash(unsigned_sums)
    ):
        raise ValueError("SHA256SUMS contract is invalid")
    expected_files = unsigned_sums["files"]
    if not isinstance(expected_files, dict) or set(expected_files) != ARCHIVE_MEMBERS - {
        "SHA256SUMS.json"
    }:
        raise ValueError("SHA256SUMS file allowlist is invalid")
    for name, digest in expected_files.items():
        if _sha256_bytes(files[str(name)]) != digest:
            raise ValueError(f"SHA256SUMS mismatch: {name}")
    manifest = _read_json(files, "manifest.json")
    unsigned_manifest = dict(manifest)
    manifest_hash = unsigned_manifest.pop("manifest_content_sha256", None)
    if manifest_hash != _semantic_hash(unsigned_manifest):
        raise ValueError("manifest semantic hash mismatch")
    if (
        manifest.get("contract_status") != "FROZEN_CONTRACT"
        or manifest.get("model_status") != "TRAINED_NOT_EVALUATED"
        or manifest.get("task") != "stable_standard_regression"
        or manifest.get("target_version") != SUPPORTED_STANDARD_VERSION
        or manifest.get("feature_schema_hash") != _semantic_hash(
            _read_json(files, "feature_schema.json")
        )
        or manifest.get("label_review_schema_hash") != _semantic_hash(
            _read_json(files, "label_review_schema.json")
        )
        or manifest.get("split_policy") not in set(_SPLIT_POLICY_MAP.values())
        or manifest.get("purge_seconds") != 1800
        or manifest.get("locked_holdout_read") is not False
        or manifest.get("delivery_eligible") is not False
        or manifest.get("promotion_eligible") is not False
        or manifest.get("real_data_status") != "NOT VERIFIED"
        or manifest.get("stage") is not None
    ):
        raise ValueError("manifest standard candidate contract is invalid")
    projection = _read_json(files, "training_projection.json")
    if (
        projection.get("runtime_to_trainer_indices")
        != list(RUNTIME_TO_TRAINER_INDICES)
        or projection.get("target_source_runtime_index")
        != TARGET_SOURCE_RUNTIME_INDEX
        or "target_version" in projection
    ):
        raise ValueError("training projection is invalid")
    sealed = _read_json(files, "sealed_evaluation_request.json")
    unsigned_request = dict(sealed)
    request_hash = unsigned_request.pop("request_sha256", None)
    if (
        request_hash != _semantic_hash(unsigned_request)
        or sealed.get("bundle_payload_digest") != _payload_digest(files)
        or sealed.get("target_version") != SUPPORTED_STANDARD_VERSION
    ):
        raise ValueError("sealed evaluation request is invalid")
    fixture = _read_json(files, "golden_fixture.json")
    output = _read_json(files, "golden_output.json")
    fixture_cases = cast(list[dict[str, object]], fixture.get("cases"))
    output_cases = cast(list[dict[str, object]], output.get("cases"))
    if (
        not isinstance(fixture_cases, list)
        or not isinstance(output_cases, list)
        or [case.get("case_id") for case in fixture_cases]
        != list(STANDARD_LEAKAGE_PROBE_CASE_IDS)
        or [case.get("case_id") for case in output_cases]
        != list(STANDARD_LEAKAGE_PROBE_CASE_IDS)
    ):
        raise ValueError("golden target leakage cases are invalid")
    runtime_input = np.asarray(
        [case["expected_input_vector"] for case in fixture_cases], dtype=np.float32
    )
    if (
        runtime_input.shape != (2, 16)
        or not np.array_equal(
            np.delete(runtime_input[0], TARGET_SOURCE_RUNTIME_INDEX),
            np.delete(runtime_input[1], TARGET_SOURCE_RUNTIME_INDEX),
        )
        or runtime_input[0, TARGET_SOURCE_RUNTIME_INDEX]
        == runtime_input[1, TARGET_SOURCE_RUNTIME_INDEX]
    ):
        raise ValueError("golden target leakage inputs are invalid")
    actual = _run_onnx(files["model.onnx"], runtime_input)
    if actual.shape != (2, 1) or actual[0].tobytes() != actual[1].tobytes():
        raise ValueError("runtime paired prediction is not byte-exact")
    expected = np.asarray(
        [cast(list[list[float]], case["expected_output"])[0][0] for case in output_cases],
        dtype=np.float32,
    )
    if expected[0].tobytes() != expected[1].tobytes() or not np.array_equal(
        actual[:, 0], expected
    ):
        raise ValueError("runtime golden output mismatch")
    return {
        "status": "VERIFIED_IMMUTABLE_STANDARD_CANDIDATE",
        "archive_sha256": _sha256_bytes(path.read_bytes()),
        "model_artifact_sha256": _sha256_bytes(files["model.onnx"]),
        "feature_schema_hash": manifest["feature_schema_hash"],
        "label_review_schema_hash": manifest["label_review_schema_hash"],
        "paired_prediction_byte_exact": True,
    }
