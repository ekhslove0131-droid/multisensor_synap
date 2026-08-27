"""Bounded Kaggle intake for validated Kidsignal BigQuery cohorts.

This module deliberately stops after read-only synchronization and prepared-array
shape validation.  It never fits, evaluates, exports, bundles, or promotes a model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

from multisensor_ml.bigquery_sdk_reader import (
    BigQueryClientProtocol,
    QueryJobConfigFactory,
)
from multisensor_ml.bigquery_training_preflight import (
    EXPECTED_VIEW_FIELDS,
    GCP_LOCATION,
    GCP_PROJECT,
    run_read_only_behavior_cohort_sdk_reader,
)
from multisensor_ml.model_sync_receipt import (
    ModelSyncReceipt,
    parse_model_sync_receipt,
)
from multisensor_ml.standard_baseline_bigquery import (
    STANDARD_BASELINE_VIEW_FIELDS,
    run_read_only_standard_cohort_sdk_reader,
)

INTAKE_SCHEMA_VERSION: Final[str] = "kidsignal-kaggle-bigquery-intake/v1"
IntakeMode = Literal["standard", "behavior"]

_RECEIPT_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "upstream_contract_commit",
        "authorized_view",
        "requested_standard_cohort_uuid",
        "requested_training_cohort_uuid",
        "standard_cohort_uuid",
        "training_cohort_uuid",
        "cohort_digest",
        "split_digest",
        "public_cohort_digest",
        "public_split_digest",
        "expected_public_cohort_digest",
        "expected_public_split_digest",
        "expected_public_digests_match",
        "stable_standard_version",
        "row_count",
        "split_counts",
        "split_class_counts",
        "target_name",
        "target_unit",
        "schema_field_sha256",
        "model_reader_identity_verified",
        "locked_access",
        "selection_contract_valid",
        "training_ready",
        "canonical_handoff_digest",
        "canonical_handoff_digest_policy",
        "public_digest_policy",
        "status",
        "blocked_reason",
        "fit_call_count",
        "training_status",
        "evaluation_status",
        "real_performance_status",
    }
)


def _bounded_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    bounded = {
        key: value for key, value in receipt.items() if key in _RECEIPT_ALLOWLIST
    }
    if bounded.get("fit_call_count") != 0:
        raise RuntimeError("Kaggle intake must stop before model fitting")
    return bounded


def _row_count(receipt: Mapping[str, object]) -> int:
    value = receipt.get("row_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("Kaggle intake receipt has invalid row_count")
    return value


def _empty_summary(mode: IntakeMode, row_count: int) -> dict[str, object]:
    summary: dict[str, object] = {
        "row_count": row_count,
        "train_shape": None,
        "validation_shape": None,
        "authorized_field_count": (
            len(STANDARD_BASELINE_VIEW_FIELDS)
            if mode == "standard"
            else len(EXPECTED_VIEW_FIELDS)
        ),
    }
    if mode == "standard":
        summary.update(
            {
                "runtime_feature_count": 16,
                "trainer_feature_count": 15,
                "runtime_train_shape": None,
                "runtime_validation_shape": None,
            }
        )
    else:
        summary["feature_count"] = 16
    return summary


def _run_authorized_bigquery_intake(
    *,
    mode: IntakeMode,
    client: BigQueryClientProtocol,
    cohort_uuid: str,
    expected_public_cohort_digest: str,
    expected_public_split_digest: str,
    observed_at_utc: str,
    observed_principal: str,
    job_config_factory: QueryJobConfigFactory | None = None,
) -> dict[str, object]:
    """Read and validate one cohort, returning no row-level or identity payload."""

    if mode == "standard":
        standard_result = run_read_only_standard_cohort_sdk_reader(
            client=client,
            standard_cohort_uuid=cohort_uuid,
            expected_public_cohort_digest=expected_public_cohort_digest,
            expected_public_split_digest=expected_public_split_digest,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            job_config_factory=job_config_factory,
        )
        receipt = _bounded_receipt(standard_result.receipt)
        summary = _empty_summary(mode, _row_count(receipt))
        standard_prepared = standard_result.prepared_dataset
        if standard_prepared is not None:
            summary.update(
                {
                    "train_shape": list(standard_prepared.train_features.shape),
                    "validation_shape": list(
                        standard_prepared.validation_features.shape
                    ),
                    "runtime_train_shape": list(
                        standard_prepared.runtime_train_features.shape
                    ),
                    "runtime_validation_shape": list(
                        standard_prepared.runtime_validation_features.shape
                    ),
                    "runtime_feature_count": len(standard_prepared.feature_names),
                    "trainer_feature_count": len(
                        standard_prepared.trainer_feature_names
                    ),
                    "feature_schema_uuid": standard_prepared.feature_schema_uuid,
                    "feature_schema_hash": standard_prepared.feature_schema_hash,
                    "split_policy": standard_prepared.split_policy,
                    "purge_seconds": standard_prepared.purge_seconds,
                    "target_leakage_status": (
                        standard_prepared.target_leakage_status
                    ),
                }
            )
    elif mode == "behavior":
        behavior_result = run_read_only_behavior_cohort_sdk_reader(
            client=client,
            training_cohort_uuid=cohort_uuid,
            expected_public_cohort_digest=expected_public_cohort_digest,
            expected_public_split_digest=expected_public_split_digest,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            job_config_factory=job_config_factory,
        )
        receipt = _bounded_receipt(behavior_result.receipt)
        summary = _empty_summary(mode, _row_count(receipt))
        summary["split_class_counts"] = receipt.get("split_class_counts")
        behavior_prepared = behavior_result.prepared_dataset
        if behavior_prepared is not None:
            summary.update(
                {
                    "train_shape": list(behavior_prepared.train_features.shape),
                    "validation_shape": list(
                        behavior_prepared.validation_features.shape
                    ),
                    "feature_count": len(behavior_prepared.feature_names),
                    "feature_schema_uuid": behavior_prepared.feature_schema_uuid,
                    "feature_schema_hash": behavior_prepared.feature_schema_hash,
                    "split_policy": behavior_prepared.split_policy,
                    "purge_seconds": behavior_prepared.purge_seconds,
                }
            )
    else:
        raise ValueError("mode must be 'standard' or 'behavior'")

    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "mode": mode,
        "gcp_project": GCP_PROJECT,
        "gcp_location": GCP_LOCATION,
        "readiness_receipt": receipt,
        "summary": summary,
        "execution_boundary": {
            "fit_call_count": 0,
            "training_started": False,
            "evaluation_started": False,
            "onnx_exported": False,
            "bundle_created": False,
            "promotion_started": False,
        },
    }


def _split_row_counts(
    mode: IntakeMode,
    readiness_receipt: Mapping[str, object],
) -> tuple[object, object]:
    if mode == "standard":
        counts = readiness_receipt.get("split_counts")
        if not isinstance(counts, Mapping):
            return None, None
        return counts.get("TRAIN"), counts.get("VALIDATION")
    counts = readiness_receipt.get("split_class_counts")
    if not isinstance(counts, Mapping):
        return None, None

    def total(split: str) -> int | None:
        values = counts.get(split)
        if not isinstance(values, Mapping):
            return None
        positive = values.get("POSITIVE")
        negative = values.get("NEGATIVE")
        if (
            isinstance(positive, bool)
            or not isinstance(positive, int)
            or isinstance(negative, bool)
            or not isinstance(negative, int)
        ):
            return None
        return positive + negative

    return total("TRAIN"), total("VALIDATION")


def _model_sync_binding_mismatches(
    receipt: ModelSyncReceipt,
    artifact: Mapping[str, object],
) -> list[str]:
    readiness = artifact.get("readiness_receipt")
    summary = artifact.get("summary")
    if not isinstance(readiness, Mapping) or not isinstance(summary, Mapping):
        return ["artifact_contract"]
    mismatches: list[str] = []
    if artifact.get("mode") != receipt.plane:
        mismatches.append("mode")
    if readiness.get("training_ready") is not True:
        mismatches.append("training_ready")
    cohort_field = (
        "standard_cohort_uuid"
        if receipt.plane == "standard"
        else "training_cohort_uuid"
    )
    comparisons = {
        "cohort_uuid": (receipt.cohort_uuid, readiness.get(cohort_field)),
        "public_cohort_digest": (
            receipt.public_cohort_digest,
            readiness.get("public_cohort_digest"),
        ),
        "public_split_digest": (
            receipt.public_split_digest,
            readiness.get("public_split_digest"),
        ),
        "feature_schema_uuid": (
            receipt.feature_schema_uuid,
            summary.get("feature_schema_uuid"),
        ),
        "feature_schema_hash": (
            receipt.feature_schema_hash,
            summary.get("feature_schema_hash"),
        ),
        "authorized_field_count": (
            receipt.authorized_field_count,
            summary.get("authorized_field_count"),
        ),
        "split_policy": (receipt.split_policy, summary.get("split_policy")),
        "purge_seconds": (receipt.purge_seconds, summary.get("purge_seconds")),
    }
    for field, (expected, actual) in comparisons.items():
        if expected != actual:
            mismatches.append(field)
    train_rows, validation_rows = _split_row_counts(receipt.plane, readiness)
    if receipt.train_row_count != train_rows:
        mismatches.append("train_row_count")
    if receipt.validation_row_count != validation_rows:
        mismatches.append("validation_row_count")
    return sorted(set(mismatches))


def _blocked_binding_artifact(
    *,
    receipt: ModelSyncReceipt,
    artifact: Mapping[str, object],
    mismatches: list[str],
) -> dict[str, object]:
    readiness_value = artifact.get("readiness_receipt")
    readiness = (
        dict(readiness_value) if isinstance(readiness_value, Mapping) else {}
    )
    readiness.update(
        {
            "status": "BLOCKED_MODEL_SYNC_RECEIPT_BINDING_MISMATCH",
            "blocked_reason": "MODEL_SYNC_RECEIPT_FIELDS_DO_NOT_MATCH_BIGQUERY",
            "selection_contract_valid": False,
            "training_ready": False,
            "fit_call_count": 0,
            "training_status": "NOT STARTED",
            "evaluation_status": "NOT EVALUABLE",
        }
    )
    row_count = readiness.get("row_count")
    bounded_row_count = (
        row_count
        if isinstance(row_count, int) and not isinstance(row_count, bool) and row_count >= 0
        else 0
    )
    return {
        "schema_version": INTAKE_SCHEMA_VERSION,
        "mode": receipt.plane,
        "gcp_project": GCP_PROJECT,
        "gcp_location": GCP_LOCATION,
        "readiness_receipt": readiness,
        "summary": _empty_summary(receipt.plane, bounded_row_count),
        "model_sync_binding": {
            "status": "BLOCKED_MODEL_SYNC_RECEIPT_BINDING_MISMATCH",
            "matched": False,
            "mismatch_fields": mismatches,
            "handoff_digest": receipt.handoff_digest,
        },
        "execution_boundary": {
            "fit_call_count": 0,
            "training_started": False,
            "evaluation_started": False,
            "onnx_exported": False,
            "bundle_created": False,
            "promotion_started": False,
        },
    }


def run_receipt_bound_kaggle_bigquery_intake(
    *,
    model_sync_receipt: object,
    client: BigQueryClientProtocol,
    observed_at_utc: str,
    observed_principal: str,
    job_config_factory: QueryJobConfigFactory | None = None,
) -> dict[str, object]:
    """Use one validated receipt as the sole authorized cohort selector."""

    receipt = parse_model_sync_receipt(model_sync_receipt)
    artifact = _run_authorized_bigquery_intake(
        mode=receipt.plane,
        client=client,
        cohort_uuid=receipt.cohort_uuid,
        expected_public_cohort_digest=receipt.public_cohort_digest,
        expected_public_split_digest=receipt.public_split_digest,
        observed_at_utc=observed_at_utc,
        observed_principal=observed_principal,
        job_config_factory=job_config_factory,
    )
    mismatches = _model_sync_binding_mismatches(receipt, artifact)
    if mismatches:
        return _blocked_binding_artifact(
            receipt=receipt,
            artifact=artifact,
            mismatches=mismatches,
        )
    artifact["model_sync_binding"] = {
        "status": "BOUND_READY_NOT_TRAINED",
        "matched": True,
        "plane": receipt.plane,
        "cohort_uuid": receipt.cohort_uuid,
        "public_cohort_digest": receipt.public_cohort_digest,
        "public_split_digest": receipt.public_split_digest,
        "feature_schema_uuid": receipt.feature_schema_uuid,
        "feature_schema_hash": receipt.feature_schema_hash,
        "authorized_field_count": receipt.authorized_field_count,
        "split_policy": receipt.split_policy,
        "purge_seconds": receipt.purge_seconds,
        "train_row_count": receipt.train_row_count,
        "validation_row_count": receipt.validation_row_count,
        "handoff_digest": receipt.handoff_digest,
    }
    return artifact


def write_kaggle_intake_artifact(
    path: Path, artifact: Mapping[str, object]
) -> None:
    """Create one bounded receipt without overwriting prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            artifact,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
