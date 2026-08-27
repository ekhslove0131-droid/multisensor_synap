"""Pure validator for create-only Kidsignal model-sync receipts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast
from uuid import UUID

from multisensor_ml.platform_contract import sha256_json
from multisensor_ml.standard_baseline_bigquery import (
    WATCH_SCHEMA_HASH,
    WATCH_SCHEMA_UUID,
)

MODEL_SYNC_RECEIPT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "contract_version",
        "status",
        "plane",
        "cohort_uuid",
        "public_cohort_digest",
        "public_split_digest",
        "feature_schema_uuid",
        "feature_schema_hash",
        "split_policy",
        "purge_seconds",
        "authorized_field_count",
        "train_row_count",
        "validation_row_count",
        "fit_call_count",
        "handoff_digest",
    }
)
MODEL_SYNC_STATUS: Final[str] = "READY_FOR_MODEL_SYNC_NOT_TRAINED"
AUTHORIZED_FIELD_COUNTS: Final[dict[str, int]] = {
    "standard": 24,
    "behavior": 26,
}
SUPPORTED_SPLIT_POLICIES: Final[frozenset[str]] = frozenset(
    {"PERSON_GROUP", "CHRONOLOGICAL_PER_SUBJECT"}
)
ModelSyncPlane = Literal["standard", "behavior"]


@dataclass(frozen=True, slots=True)
class ModelSyncReceipt:
    contract_version: int
    status: str
    plane: ModelSyncPlane
    cohort_uuid: str
    public_cohort_digest: str
    public_split_digest: str
    feature_schema_uuid: str
    feature_schema_hash: str
    split_policy: str
    purge_seconds: int
    authorized_field_count: int
    train_row_count: int
    validation_row_count: int
    fit_call_count: int
    handoff_digest: str


def _require_exact_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _require_uuid(
    value: object,
    field: str,
    *,
    version: int | None = None,
) -> str:
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


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def parse_model_sync_receipt(payload: object) -> ModelSyncReceipt:
    """Validate one bounded receipt without reading rows or starting fitting."""

    if not isinstance(payload, Mapping):
        raise ValueError("model sync receipt must be an object")
    fields = {str(key) for key in payload}
    missing = sorted(MODEL_SYNC_RECEIPT_FIELDS.difference(fields))
    if missing:
        raise ValueError(f"model sync receipt missing field {missing[0]}")
    extra = sorted(fields.difference(MODEL_SYNC_RECEIPT_FIELDS))
    if extra:
        raise ValueError(f"model sync receipt has unknown field {extra[0]}")

    receipt = {str(key): value for key, value in payload.items()}
    supplied_digest = _require_sha256(receipt["handoff_digest"], "handoff_digest")
    digest_body = {
        key: value for key, value in receipt.items() if key != "handoff_digest"
    }
    if sha256_json(digest_body) != supplied_digest:
        raise ValueError("handoff_digest does not match model sync receipt")

    contract_version = _require_exact_int(
        receipt["contract_version"], "contract_version"
    )
    if contract_version != 1:
        raise ValueError("contract_version must be 1")
    status = receipt["status"]
    if status != MODEL_SYNC_STATUS:
        raise ValueError(f"status must be {MODEL_SYNC_STATUS}")
    plane_value = receipt["plane"]
    if not isinstance(plane_value, str) or plane_value not in AUTHORIZED_FIELD_COUNTS:
        raise ValueError("plane must be standard or behavior")
    plane = cast(ModelSyncPlane, plane_value)

    cohort_uuid = _require_uuid(receipt["cohort_uuid"], "cohort_uuid", version=4)
    public_cohort_digest = _require_sha256(
        receipt["public_cohort_digest"], "public_cohort_digest"
    )
    public_split_digest = _require_sha256(
        receipt["public_split_digest"], "public_split_digest"
    )
    feature_schema_uuid = _require_uuid(
        receipt["feature_schema_uuid"], "feature_schema_uuid"
    )
    feature_schema_hash = _require_sha256(
        receipt["feature_schema_hash"], "feature_schema_hash"
    )
    if feature_schema_uuid != WATCH_SCHEMA_UUID:
        raise ValueError("feature_schema_uuid is not the canonical Watch schema")
    if feature_schema_hash != WATCH_SCHEMA_HASH:
        raise ValueError("feature_schema_hash is not the canonical Watch schema")

    split_policy = receipt["split_policy"]
    if not isinstance(split_policy, str) or split_policy not in SUPPORTED_SPLIT_POLICIES:
        raise ValueError("split_policy is unsupported")
    purge_seconds = _require_exact_int(receipt["purge_seconds"], "purge_seconds")
    if purge_seconds < 1800:
        raise ValueError("purge_seconds must be at least 1800")
    authorized_field_count = _require_exact_int(
        receipt["authorized_field_count"], "authorized_field_count"
    )
    if authorized_field_count != AUTHORIZED_FIELD_COUNTS[plane]:
        raise ValueError("authorized_field_count does not match plane")

    train_row_count = _require_exact_int(
        receipt["train_row_count"], "train_row_count"
    )
    if train_row_count <= 0:
        raise ValueError("train_row_count must be positive")
    validation_row_count = _require_exact_int(
        receipt["validation_row_count"], "validation_row_count"
    )
    if validation_row_count <= 0:
        raise ValueError("validation_row_count must be positive")
    fit_call_count = _require_exact_int(receipt["fit_call_count"], "fit_call_count")
    if fit_call_count != 0:
        raise ValueError("fit_call_count must be 0")

    return ModelSyncReceipt(
        contract_version=contract_version,
        status=MODEL_SYNC_STATUS,
        plane=plane,
        cohort_uuid=cohort_uuid,
        public_cohort_digest=public_cohort_digest,
        public_split_digest=public_split_digest,
        feature_schema_uuid=feature_schema_uuid,
        feature_schema_hash=feature_schema_hash,
        split_policy=split_policy,
        purge_seconds=purge_seconds,
        authorized_field_count=authorized_field_count,
        train_row_count=train_row_count,
        validation_row_count=validation_row_count,
        fit_call_count=fit_call_count,
        handoff_digest=supplied_digest,
    )
