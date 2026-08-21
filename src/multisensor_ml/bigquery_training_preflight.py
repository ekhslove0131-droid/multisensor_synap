"""Read-only preflight for the sole Kidsignal BigQuery model-training view."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Final, cast
from uuid import UUID

import numpy as np

from multisensor_ml.h10_runtime_contract import canonical_h10_feature_schema
from multisensor_ml.observational_contract import FEATURE_NAMES
from multisensor_ml.platform_contract import (
    AUTHORIZED_ENVELOPE_FIELDS,
    AUTHORIZED_ROW_FIELDS,
    synchronize_authorized_training_rows,
)

AUTHORIZED_TRAINING_VIEW: Final[str] = (
    "multi-app-kidsignal-260801.kidsignal_model_training."
    "training_examples_train_validation_v1"
)
EXPECTED_MODEL_READER: Final[str] = (
    "kidsignal-model-reader@multi-app-kidsignal-260801.iam.gserviceaccount.com"
)
GCP_PROJECT: Final[str] = "multi-app-kidsignal-260801"
GCP_LOCATION: Final[str] = "asia-southeast1"
H10_RUNTIME_SCHEMA_HASH: Final[str] = str(canonical_h10_feature_schema()["sha256"])
EXPECTED_VIEW_FIELDS: Final[tuple[str, ...]] = (
    *AUTHORIZED_ENVELOPE_FIELDS,
    *AUTHORIZED_ROW_FIELDS,
)
COHORT_READINESS_VERSION: Final[str] = "kidsignal-bigquery-cohort-readiness/v1"
PUBLIC_DIGEST_PIN_POLICY: Final[str] = (
    "EXPECTED_PUBLIC_COHORT_AND_SPLIT_DIGEST_PIN"
)
CANONICAL_HANDOFF_DIGEST_POLICY: Final[str] = (
    "RECOMPUTED_AFTER_EXPECTED_PUBLIC_DIGEST_MATCH"
)


@dataclass(frozen=True, slots=True)
class PreparedBehaviorDataset:
    """Validated arrays for the existing classifier trainer boundary."""

    task: str
    training_cohort_uuid: str
    public_cohort_digest: str
    public_split_digest: str
    canonical_handoff_digest: str
    feature_names: tuple[str, ...]
    train_features: np.ndarray
    train_targets: np.ndarray
    train_subject_groups: tuple[str, ...]
    train_sequence_groups: tuple[str, ...]
    train_row_ids: tuple[str, ...]
    validation_features: np.ndarray
    validation_targets: np.ndarray
    validation_subject_groups: tuple[str, ...]
    validation_sequence_groups: tuple[str, ...]
    validation_row_ids: tuple[str, ...]
    fit_call_count: int = 0


def _canonical_uuid(
    value: str,
    field: str,
    *,
    versions: frozenset[int] | None = None,
) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a canonical UUID") from error
    if str(parsed) != value.lower():
        raise ValueError(f"{field} must be a canonical UUID")
    if versions is not None and parsed.version not in versions:
        raise ValueError(f"{field} has unsupported UUID version")
    return str(parsed)


def _require_sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def prepare_behavior_dataset_handoff(
    *,
    rows: Sequence[Mapping[str, object]],
    expected_public_cohort_digest: str,
    expected_public_split_digest: str,
) -> PreparedBehaviorDataset:
    """Validate pins and expose typed arrays without invoking model fitting."""

    expected_cohort = _require_sha256(
        expected_public_cohort_digest, "expected_public_cohort_digest"
    )
    expected_split = _require_sha256(
        expected_public_split_digest, "expected_public_split_digest"
    )
    handoff = synchronize_authorized_training_rows(rows)
    if (
        handoff["public_cohort_digest"] != expected_cohort
        or handoff["public_split_digest"] != expected_split
    ):
        raise ValueError("returned public digests do not match expected handoff")
    if handoff["truth_state"] != "REVIEWED_REAL":
        raise ValueError("prepared behavior dataset requires REVIEWED_REAL truth")
    handoff_rows = cast(Sequence[Mapping[str, object]], handoff["rows"])

    def prepare_split(
        split: str,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[str, ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        selected = [row for row in handoff_rows if row["split_role"] == split]
        labels = [str(row["evaluation_class"]) for row in selected]
        if {"POSITIVE", "NEGATIVE"}.difference(labels):
            raise ValueError(f"{split} requires positive and negative support")
        features = np.asarray(
            [
                [
                    cast(Mapping[str, object], row["feature_values"])[name]
                    for name in FEATURE_NAMES
                ]
                for row in selected
            ],
            dtype=np.float32,
        )
        targets = np.asarray(
            [1 if label == "POSITIVE" else 0 for label in labels],
            dtype=np.int8,
        )
        features.setflags(write=False)
        targets.setflags(write=False)
        return (
            features,
            targets,
            tuple(str(row["training_subject_uuid"]) for row in selected),
            tuple(str(row["training_capture_set_uuid"]) for row in selected),
            tuple(str(row["exact_window_id"]) for row in selected),
        )

    train = prepare_split("TRAIN")
    validation = prepare_split("VALIDATION")
    return PreparedBehaviorDataset(
        task="reviewed_behavior_binary",
        training_cohort_uuid=str(handoff["training_cohort_uuid"]),
        public_cohort_digest=expected_cohort,
        public_split_digest=expected_split,
        canonical_handoff_digest=str(handoff["handoff_digest"]),
        feature_names=tuple(FEATURE_NAMES),
        train_features=train[0],
        train_targets=train[1],
        train_subject_groups=train[2],
        train_sequence_groups=train[3],
        train_row_ids=train[4],
        validation_features=validation[0],
        validation_targets=validation[1],
        validation_subject_groups=validation[2],
        validation_sequence_groups=validation[3],
        validation_row_ids=validation[4],
    )


def _validate_exact_schema(schema_fields: Sequence[str]) -> tuple[str, ...]:
    fields = tuple(schema_fields)
    actual = set(fields)
    expected = set(EXPECTED_VIEW_FIELDS)
    unexpected = sorted(actual.difference(expected))
    if unexpected:
        raise ValueError(f"forbidden or unknown field: {unexpected[0]}")
    missing = sorted(expected.difference(actual))
    if missing:
        raise ValueError(f"missing authorized field: {missing[0]}")
    if len(fields) != len(actual):
        raise ValueError("duplicate authorized field")
    return fields


def _schema_digest(fields: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            tuple(fields),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_preflight_receipt(
    *,
    schema_fields: Sequence[str],
    row_count: int,
    observed_at_utc: str,
    observed_principal: str,
) -> dict[str, object]:
    fields = _validate_exact_schema(schema_fields)
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise ValueError("row_count must be a non-negative integer")
    if not observed_at_utc.endswith("Z"):
        raise ValueError("observed_at_utc must be UTC Z text")

    identity_verified = observed_principal == EXPECTED_MODEL_READER
    if row_count == 0:
        status = "BLOCKED_NO_REAL_COHORT"
    elif not identity_verified:
        status = "BLOCKED_MODEL_READER_IDENTITY_NOT_VERIFIED"
    else:
        status = "AWAITING_EXPLICIT_FROZEN_COHORT_KEY"
    schema_digest = _schema_digest(fields)
    return {
        "schema_version": "kidsignal-bigquery-training-preflight/v2",
        "observed_at_utc": observed_at_utc,
        "mode": "READ_ONLY",
        "gcp_project": GCP_PROJECT,
        "gcp_location": GCP_LOCATION,
        "authorized_view": AUTHORIZED_TRAINING_VIEW,
        "authorized_view_is_only_training_input": True,
        "expected_model_reader": EXPECTED_MODEL_READER,
        "observed_principal": observed_principal,
        "model_reader_identity_verified": identity_verified,
        "schema_fields": list(fields),
        "schema_field_sha256": schema_digest,
        "private_fields_detected": [],
        "unknown_fields_detected": [],
        "observed_row_count": row_count,
        "status": status,
        "sync_status": "NOT STARTED",
        "training_status": "NOT STARTED",
        "evaluation_status": "NOT EVALUABLE",
        "real_performance_status": "NOT VERIFIED",
        "kaggle_started": False,
        "locked_access": False,
        "allowed_split_roles": ["TRAIN", "VALIDATION"],
        "sequence_group_key": "training_capture_set_uuid",
        "handoff_digest_policy": CANONICAL_HANDOFF_DIGEST_POLICY,
        "h10_feature_schema_hash": H10_RUNTIME_SCHEMA_HASH,
        "h10_only_status": "PROPOSED_NOT_IMPLEMENTED",
        "watch_h10_status": "PROPOSED_NOT_APPROVED",
        "required_next_input": [
            "training_cohort_uuid",
            "expected_public_cohort_digest",
            "expected_public_split_digest",
            "TRAIN and VALIDATION rows",
            "purge_seconds>=1800",
            "feature_schema_uuid and feature_schema_hash",
            "reviewed truth support",
        ],
    }


def _empty_split_counts() -> dict[str, dict[str, int]]:
    return {
        "TRAIN": {"POSITIVE": 0, "NEGATIVE": 0},
        "VALIDATION": {"POSITIVE": 0, "NEGATIVE": 0},
    }


def _split_class_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    counts = _empty_split_counts()
    for row in rows:
        split = str(row.get("split_role", ""))
        evaluation = str(row.get("evaluation_class", ""))
        if split in counts and evaluation in counts[split]:
            counts[split][evaluation] += 1
    return counts


def _blocked_cohort_receipt(
    *,
    status: str,
    requested_training_cohort_uuid: str,
    observed_at_utc: str,
    observed_principal: str,
    schema_fields: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    blocked_reason: str,
    expected_public_cohort_digest: str | None = None,
    expected_public_split_digest: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": COHORT_READINESS_VERSION,
        "observed_at_utc": observed_at_utc,
        "mode": "READ_ONLY",
        "authorized_view": AUTHORIZED_TRAINING_VIEW,
        "requested_training_cohort_uuid": requested_training_cohort_uuid,
        "training_cohort_uuid": requested_training_cohort_uuid,
        "cohort_digest": None,
        "split_digest": None,
        "public_cohort_digest": None,
        "public_split_digest": None,
        "expected_public_cohort_digest": expected_public_cohort_digest,
        "expected_public_split_digest": expected_public_split_digest,
        "expected_public_digests_match": False,
        "row_count": len(rows),
        "split_class_counts": _split_class_counts(rows),
        "feature_schema_uuid": None,
        "feature_schema_hash": None,
        "schema_fields": list(schema_fields),
        "schema_field_sha256": _schema_digest(schema_fields),
        "observed_principal": observed_principal,
        "model_reader_identity_verified": observed_principal == EXPECTED_MODEL_READER,
        "locked_access": False,
        "selection_contract_valid": False,
        "training_ready": False,
        "canonical_handoff_digest": None,
        "canonical_handoff_digest_policy": CANONICAL_HANDOFF_DIGEST_POLICY,
        "public_digest_policy": PUBLIC_DIGEST_PIN_POLICY,
        "status": status,
        "blocked_reason": blocked_reason,
        "fit_call_count": 0,
        "training_status": "NOT STARTED",
        "evaluation_status": "NOT EVALUABLE",
    }


def build_cohort_readiness_receipt(
    *,
    rows: Sequence[Mapping[str, object]],
    schema_fields: Sequence[str],
    requested_training_cohort_uuid: str,
    expected_public_cohort_digest: str,
    expected_public_split_digest: str,
    observed_at_utc: str,
    observed_principal: str,
) -> dict[str, object]:
    """Validate one public cohort without fitting or exposing row content."""

    requested_uuid = _canonical_uuid(
        requested_training_cohort_uuid,
        "training_cohort_uuid",
        versions=frozenset({4}),
    )
    fields = _validate_exact_schema(schema_fields)
    expected_cohort_digest = _require_sha256(
        expected_public_cohort_digest, "expected_public_cohort_digest"
    )
    expected_split_digest = _require_sha256(
        expected_public_split_digest, "expected_public_split_digest"
    )
    if not observed_at_utc.endswith("Z"):
        raise ValueError("observed_at_utc must be UTC Z text")
    values = tuple(dict(row) for row in rows)
    if not values:
        return _blocked_cohort_receipt(
            status="BLOCKED_NO_REAL_COHORT",
            requested_training_cohort_uuid=requested_uuid,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            schema_fields=fields,
            rows=values,
            blocked_reason="AUTHORIZED_VIEW_RETURNED_ZERO_ROWS",
            expected_public_cohort_digest=expected_cohort_digest,
            expected_public_split_digest=expected_split_digest,
        )
    if any(row.get("training_cohort_uuid") != requested_uuid for row in values):
        return _blocked_cohort_receipt(
            status="BLOCKED_MIXED_OR_INVALID_COHORT",
            requested_training_cohort_uuid=requested_uuid,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            schema_fields=fields,
            rows=values,
            blocked_reason="REQUESTED_COHORT_UUID_MISMATCH",
            expected_public_cohort_digest=expected_cohort_digest,
            expected_public_split_digest=expected_split_digest,
        )
    try:
        handoff = synchronize_authorized_training_rows(values)
    except (TypeError, ValueError) as error:
        return _blocked_cohort_receipt(
            status="BLOCKED_MIXED_OR_INVALID_COHORT",
            requested_training_cohort_uuid=requested_uuid,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            schema_fields=fields,
            rows=values,
            blocked_reason=str(error),
            expected_public_cohort_digest=expected_cohort_digest,
            expected_public_split_digest=expected_split_digest,
        )

    if (
        handoff["public_cohort_digest"] != expected_cohort_digest
        or handoff["public_split_digest"] != expected_split_digest
    ):
        return _blocked_cohort_receipt(
            status="BLOCKED_EXPECTED_PUBLIC_DIGEST_MISMATCH",
            requested_training_cohort_uuid=requested_uuid,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            schema_fields=fields,
            rows=values,
            blocked_reason="RETURNED_PUBLIC_DIGESTS_DO_NOT_MATCH_EXPECTED_HANDOFF",
            expected_public_cohort_digest=expected_cohort_digest,
            expected_public_split_digest=expected_split_digest,
        )

    counts = _split_class_counts(values)
    class_ready = all(
        counts[split][label] >= 1
        for split in ("TRAIN", "VALIDATION")
        for label in ("POSITIVE", "NEGATIVE")
    )
    identity_verified = observed_principal == EXPECTED_MODEL_READER
    if handoff["truth_state"] != "REVIEWED_REAL":
        status = "BLOCKED_TRUTH_NOT_REVIEWED_REAL"
        blocked_reason = "TRUTH_STATE_MUST_BE_REVIEWED_REAL"
    elif not class_ready:
        status = "BLOCKED_INSUFFICIENT_CLASS_SUPPORT"
        blocked_reason = "EACH_SPLIT_REQUIRES_POSITIVE_AND_NEGATIVE"
    elif not identity_verified:
        status = "BLOCKED_MODEL_READER_IDENTITY_NOT_VERIFIED"
        blocked_reason = "DEDICATED_MODEL_READER_REQUIRED"
    else:
        status = "READY_FOR_SYNC_NOT_TRAINED"
        blocked_reason = None
    return {
        "schema_version": COHORT_READINESS_VERSION,
        "observed_at_utc": observed_at_utc,
        "mode": "READ_ONLY",
        "authorized_view": AUTHORIZED_TRAINING_VIEW,
        "requested_training_cohort_uuid": requested_uuid,
        "training_cohort_uuid": str(handoff["training_cohort_uuid"]),
        "cohort_digest": str(handoff["cohort_digest"]),
        "split_digest": str(handoff["split_digest"]),
        "public_cohort_digest": str(handoff["public_cohort_digest"]),
        "public_split_digest": str(handoff["public_split_digest"]),
        "expected_public_cohort_digest": expected_cohort_digest,
        "expected_public_split_digest": expected_split_digest,
        "expected_public_digests_match": True,
        "row_count": len(values),
        "split_class_counts": counts,
        "feature_schema_uuid": str(handoff["feature_schema_uuid"]),
        "feature_schema_hash": str(handoff["feature_schema_hash"]),
        "schema_fields": list(fields),
        "schema_field_sha256": _schema_digest(fields),
        "observed_principal": observed_principal,
        "model_reader_identity_verified": identity_verified,
        "locked_access": False,
        "selection_contract_valid": (
            handoff["truth_state"] == "REVIEWED_REAL"
            and class_ready
            and identity_verified
        ),
        "training_ready": status == "READY_FOR_SYNC_NOT_TRAINED",
        "canonical_handoff_digest": str(handoff["handoff_digest"]),
        "canonical_handoff_digest_policy": CANONICAL_HANDOFF_DIGEST_POLICY,
        "public_digest_policy": PUBLIC_DIGEST_PIN_POLICY,
        "status": status,
        "blocked_reason": blocked_reason,
        "fit_call_count": 0,
        "training_status": "NOT STARTED",
        "evaluation_status": "NOT EVALUABLE",
    }


def _run(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def active_gcloud_principal(
    runner: Callable[[list[str]], CompletedProcess[str]] = _run,
) -> str:
    """Read the current local gcloud principal without changing authentication."""

    result = runner(["gcloud", "config", "get-value", "account"])
    if result.returncode != 0:
        return "NOT VERIFIED"
    principal = result.stdout.strip()
    return principal if principal and principal != "(unset)" else "NOT VERIFIED"


def _parse_schema(stdout: str) -> tuple[str, ...]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError("BigQuery schema output is not JSON") from error
    if not isinstance(value, list):
        raise ValueError("BigQuery schema output must be a list")
    fields: list[str] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("BigQuery schema field is invalid")
        fields.append(item["name"])
    return tuple(fields)


def _parse_row_count(stdout: str) -> int:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError("BigQuery count output is not JSON") from error
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], dict)
        or "row_count" not in value[0]
    ):
        raise ValueError("BigQuery count output is invalid")
    try:
        count = int(value[0]["row_count"])
    except (TypeError, ValueError) as error:
        raise ValueError("BigQuery row_count is invalid") from error
    if count < 0:
        raise ValueError("BigQuery row_count cannot be negative")
    return count


def _parse_authorized_rows(stdout: str) -> tuple[dict[str, object], ...]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError("BigQuery cohort output is not JSON") from error
    if not isinstance(value, list):
        raise ValueError("BigQuery cohort output must be a list")
    output: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"BigQuery cohort row {index} is invalid")
        row = {str(key): field for key, field in item.items()}
        for field in ("purge_seconds", "window_start_ms", "window_end_ms", "temporal_stage"):
            try:
                row[field] = int(cast(str | int, row[field]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"BigQuery cohort row {index} has invalid {field}") from error
        features = row.get("feature_values")
        if isinstance(features, str):
            try:
                features = json.loads(features)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"BigQuery cohort row {index} has invalid feature_values"
                ) from error
            row["feature_values"] = features
        output.append(row)
    return tuple(output)


def behavior_cohort_query_contract(
    training_cohort_uuid: str,
) -> dict[str, object]:
    """Return the parameterized authorized-view query shared by CLI and Kaggle."""

    cohort_uuid = _canonical_uuid(
        training_cohort_uuid,
        "training_cohort_uuid",
        versions=frozenset({4}),
    )
    selected_fields = ", ".join(EXPECTED_VIEW_FIELDS)
    return {
        "authorized_view": AUTHORIZED_TRAINING_VIEW,
        "query": (
            f"SELECT {selected_fields} FROM `{AUTHORIZED_TRAINING_VIEW}` "
            "WHERE training_cohort_uuid=@training_cohort_uuid "
            "ORDER BY split_role, training_subject_uuid, window_start_ms, "
            "exact_window_id"
        ),
        "parameter": {
            "name": "training_cohort_uuid",
            "type": "STRING",
            "value": cohort_uuid,
        },
        "field_count": len(EXPECTED_VIEW_FIELDS),
        "locked_access": False,
        "allowed_transports": ["bq_cli", "google_cloud_bigquery_sdk"],
    }


def run_read_only_cohort_reader(
    output_receipt: Path,
    *,
    training_cohort_uuid: str,
    expected_public_cohort_digest: str,
    expected_public_split_digest: str,
    observed_at_utc: str,
    observed_principal: str,
    runner: Callable[[list[str]], CompletedProcess[str]] = _run,
) -> dict[str, object]:
    """Read exactly one public cohort with a bound UUID parameter."""

    cohort_uuid = _canonical_uuid(
        training_cohort_uuid,
        "training_cohort_uuid",
        versions=frozenset({4}),
    )
    expected_cohort_digest = _require_sha256(
        expected_public_cohort_digest, "expected_public_cohort_digest"
    )
    expected_split_digest = _require_sha256(
        expected_public_split_digest, "expected_public_split_digest"
    )
    if output_receipt.exists():
        raise FileExistsError(output_receipt)
    bq_target = AUTHORIZED_TRAINING_VIEW.replace(".", ":", 1)
    common = [
        "bq",
        f"--project_id={GCP_PROJECT}",
        f"--location={GCP_LOCATION}",
    ]
    schema_result = runner(
        [*common, "show", "--schema", "--format=prettyjson", bq_target]
    )
    if schema_result.returncode != 0:
        raise RuntimeError(f"BigQuery schema read failed: {schema_result.stderr.strip()}")
    parsed_schema = _parse_schema(schema_result.stdout)
    try:
        _validate_exact_schema(parsed_schema)
    except ValueError as error:
        receipt = _blocked_cohort_receipt(
            status="BLOCKED_SCHEMA_CONTRACT_MISMATCH",
            requested_training_cohort_uuid=cohort_uuid,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            schema_fields=EXPECTED_VIEW_FIELDS,
            rows=(),
            blocked_reason=str(error),
            expected_public_cohort_digest=expected_cohort_digest,
            expected_public_split_digest=expected_split_digest,
        )
        output_receipt.parent.mkdir(parents=True, exist_ok=True)
        with output_receipt.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return receipt
    query_contract = behavior_cohort_query_contract(cohort_uuid)
    parameter = cast(Mapping[str, object], query_contract["parameter"])
    rows_result = runner(
        [
            *common,
            "query",
            "--use_legacy_sql=false",
            "--format=json",
            f"--parameter={parameter['name']}:{parameter['type']}:{parameter['value']}",
            str(query_contract["query"]),
        ]
    )
    if rows_result.returncode != 0:
        raise RuntimeError(f"BigQuery cohort read failed: {rows_result.stderr.strip()}")
    receipt = build_cohort_readiness_receipt(
        rows=_parse_authorized_rows(rows_result.stdout),
        schema_fields=parsed_schema,
        requested_training_cohort_uuid=cohort_uuid,
        expected_public_cohort_digest=expected_cohort_digest,
        expected_public_split_digest=expected_split_digest,
        observed_at_utc=observed_at_utc,
        observed_principal=observed_principal,
    )
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    with output_receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def run_read_only_preflight(
    output_receipt: Path,
    *,
    observed_at_utc: str,
    observed_principal: str,
    runner: Callable[[list[str]], CompletedProcess[str]] = _run,
) -> dict[str, object]:
    if output_receipt.exists():
        raise FileExistsError(output_receipt)
    bq_target = AUTHORIZED_TRAINING_VIEW.replace(".", ":", 1)
    common = [
        "bq",
        f"--project_id={GCP_PROJECT}",
        f"--location={GCP_LOCATION}",
    ]
    schema_result = runner(
        [*common, "show", "--schema", "--format=prettyjson", bq_target]
    )
    if schema_result.returncode != 0:
        raise RuntimeError(f"BigQuery schema read failed: {schema_result.stderr.strip()}")
    count_query = f"SELECT COUNT(*) AS row_count FROM `{AUTHORIZED_TRAINING_VIEW}`"
    count_result = runner(
        [
            *common,
            "query",
            "--use_legacy_sql=false",
            "--format=json",
            "--max_rows=1",
            count_query,
        ]
    )
    if count_result.returncode != 0:
        raise RuntimeError(f"BigQuery count read failed: {count_result.stderr.strip()}")
    receipt = build_preflight_receipt(
        schema_fields=_parse_schema(schema_result.stdout),
        row_count=_parse_row_count(count_result.stdout),
        observed_at_utc=observed_at_utc,
        observed_principal=observed_principal,
    )
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    with output_receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt
