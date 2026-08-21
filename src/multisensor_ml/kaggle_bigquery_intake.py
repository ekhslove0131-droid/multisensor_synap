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
    GCP_LOCATION,
    GCP_PROJECT,
    run_read_only_behavior_cohort_sdk_reader,
)
from multisensor_ml.standard_baseline_bigquery import (
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


def run_kaggle_bigquery_intake(
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
