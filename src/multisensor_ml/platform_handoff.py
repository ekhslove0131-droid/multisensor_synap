"""Model-side Kidsignal contract handoff and evaluation evidence builders."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Final
from uuid import UUID

import numpy as np

from multisensor_ml.platform_bundle import ReferenceBundleResult, export_simple_reference_bundle
from multisensor_ml.platform_contract import (
    PLATFORM_CONTRACT_VERSION,
    REQUIRED_SOURCES,
    SENSOR_VARIANTS,
    baseline_schema,
    bigquery_training_contract,
    canonical_platform_contract,
    feature_schema_for_variant,
    label_review_schema,
    source_schema,
)

HANDOFF_VERSION: Final[str] = "kidsignal-model-platform-handoff/v1"
EVALUATION_VERSION: Final[str] = "kidsignal-reviewed-evaluation/v1"
COMPARISON_VERSION: Final[str] = "kidsignal-active-candidate-comparison/v1"


@dataclass(frozen=True, slots=True)
class PlatformHandoffResult:
    root: Path
    manifest_path: Path
    checksums_path: Path
    reference_bundles: dict[str, ReferenceBundleResult]


def _uuid(value: object, field: str) -> str:
    try:
        parsed = UUID(str(value))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"{field} must be a UUID") from error
    if str(parsed) != str(value).lower():
        raise ValueError(f"{field} must use canonical UUID text")
    return str(parsed)


def _sha(value: object, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_evaluation_manifest(
    *,
    evaluation_uuid: str,
    model_release_uuid: str,
    training_cohort_uuid: str,
    cohort_digest: str,
    sensor_variant: str,
    evaluated_cases: list[dict[str, object]],
    locked_holdout_read: bool,
) -> dict[str, object]:
    """Build reviewed FP/FN evidence without treating unknowns as negatives."""

    for field, value in (
        ("evaluation_uuid", evaluation_uuid),
        ("model_release_uuid", model_release_uuid),
        ("training_cohort_uuid", training_cohort_uuid),
    ):
        _uuid(value, field)
    _sha(cohort_digest, "cohort_digest")
    if sensor_variant not in SENSOR_VARIANTS:
        raise ValueError("sensor_variant is not canonical")
    if locked_holdout_read and evaluated_cases:
        raise ValueError(
            "locked holdout row-level cases are forbidden until holdout retirement"
        )
    required = {
        "training_subject_uuid",
        "source_set",
        "exact_window_id",
        "event_time_utc",
        "label_revision_uuid",
        "review_uuid",
        "prediction",
        "threshold",
        "error_type",
        "quality_status",
        "artifact_uri",
        "artifact_sha256",
    }
    normalized: list[dict[str, object]] = []
    for index, case in enumerate(evaluated_cases):
        missing = sorted(required.difference(case))
        if missing:
            raise ValueError(f"evaluated_cases[{index}] missing {missing[0]}")
        _uuid(case["training_subject_uuid"], f"evaluated_cases[{index}].training_subject_uuid")
        _uuid(case["label_revision_uuid"], f"evaluated_cases[{index}].label_revision_uuid")
        _uuid(case["review_uuid"], f"evaluated_cases[{index}].review_uuid")
        _sha(case["exact_window_id"], f"evaluated_cases[{index}].exact_window_id")
        _sha(case["artifact_sha256"], f"evaluated_cases[{index}].artifact_sha256")
        if case["error_type"] not in {"FP", "FN"}:
            raise ValueError("evaluated error_type must be FP or FN")
        raw_prediction = case["prediction"]
        raw_threshold = case["threshold"]
        if not isinstance(raw_prediction, Real) or not isinstance(raw_threshold, Real):
            raise ValueError("prediction and threshold must be numeric")
        prediction = float(raw_prediction)
        threshold = float(raw_threshold)
        if not math.isfinite(prediction) or not 0 <= prediction <= 1:
            raise ValueError("prediction must be finite and in [0,1]")
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("threshold must be finite and in (0,1)")
        source_set = case["source_set"]
        if source_set != list(REQUIRED_SOURCES[sensor_variant]):
            raise ValueError("source_set does not match sensor_variant")
        normalized.append({key: case[key] for key in sorted(case)})

    fp_cases = [case for case in normalized if case["error_type"] == "FP"]
    fn_cases = [case for case in normalized if case["error_type"] == "FN"]
    status = "EVALUABLE_REVIEWED_ERRORS_ONLY" if normalized else "NOT EVALUABLE"
    return {
        "contract_version": EVALUATION_VERSION,
        "evaluation_uuid": evaluation_uuid,
        "model_release_uuid": model_release_uuid,
        "training_cohort_uuid": training_cohort_uuid,
        "cohort_digest": cohort_digest,
        "sensor_variant": sensor_variant,
        "locked_holdout_read": locked_holdout_read,
        "error_slice_scope": "OPERATIONAL_ADJUDICATED",
        "locked_holdout_row_level_return": "FORBIDDEN_UNTIL_RETIREMENT",
        "post_retirement_destination": "NEW_COHORT_CANDIDATE_ONLY",
        "status": status,
        "real_accuracy_status": "NOT VERIFIED",
        "fp_cases": fp_cases,
        "fn_cases": fn_cases,
        "unknowns_counted_as_negative": False,
        "required_next_evidence": [
            "independent_target_event_truth",
            "reviewed_valid_non_event_or_hard_negative",
            "frozen_locked_holdout",
        ],
    }


def build_comparison_manifest(
    *,
    comparison_uuid: str,
    training_cohort_uuid: str,
    cohort_digest: str,
    evaluation_input_digest: str,
    active_model_release_uuid: str,
    candidate_model_release_uuid: str,
    active_evaluation_input_digest: str,
    candidate_evaluation_input_digest: str,
    reviewed_metric_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Require ACTIVE/CANDIDATE to use exactly the same frozen inputs."""

    for field, value in (
        ("comparison_uuid", comparison_uuid),
        ("training_cohort_uuid", training_cohort_uuid),
        ("active_model_release_uuid", active_model_release_uuid),
        ("candidate_model_release_uuid", candidate_model_release_uuid),
    ):
        _uuid(value, field)
    for field, value in (
        ("cohort_digest", cohort_digest),
        ("evaluation_input_digest", evaluation_input_digest),
        ("active_evaluation_input_digest", active_evaluation_input_digest),
        ("candidate_evaluation_input_digest", candidate_evaluation_input_digest),
    ):
        _sha(value, field)
    if not (
        active_evaluation_input_digest
        == candidate_evaluation_input_digest
        == evaluation_input_digest
    ):
        raise ValueError("ACTIVE and CANDIDATE must use the same frozen evaluation input")
    status = "REVIEWED_COMPARISON" if reviewed_metric_rows else "NOT EVALUABLE"
    return {
        "contract_version": COMPARISON_VERSION,
        "comparison_uuid": comparison_uuid,
        "training_cohort_uuid": training_cohort_uuid,
        "cohort_digest": cohort_digest,
        "evaluation_input_digest": evaluation_input_digest,
        "active_model_release_uuid": active_model_release_uuid,
        "candidate_model_release_uuid": candidate_model_release_uuid,
        "same_window_comparison": True,
        "reviewed_metric_rows": reviewed_metric_rows,
        "status": status,
        "real_accuracy_status": "NOT VERIFIED",
        "promotion_eligible": False,
        "promotion_requires_explicit_user_approval": True,
    }


def _fixture_metrics() -> tuple[dict[str, float], dict[str, float]]:
    current = np.asarray([0.2, 0.8, 0.4, 1.1], dtype=np.float64)
    rolling = np.asarray([0.3, 0.7, 0.5, 0.9], dtype=np.float64)
    target = np.asarray([0.35, 0.65, 0.55, 0.85], dtype=np.float64)
    training = {
        "persistence": float(np.mean(np.abs(current - target))),
        "rolling_median_300": float(np.mean(np.abs(rolling - target))),
    }
    validation = {
        "persistence": float(np.mean(np.abs(current[::-1] - target))),
        "rolling_median_300": float(np.mean(np.abs(rolling[::-1] - target))),
    }
    return training, validation


def export_contract_fixture_handoff(output_dir: Path) -> PlatformHandoffResult:
    """Export deterministic local fixtures; this is not a production model release."""

    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    contracts = root / "contracts"
    contracts.mkdir()
    _write_json(contracts / "platform_contract.json", canonical_platform_contract())
    _write_json(contracts / "baseline_schema.json", baseline_schema())
    _write_json(contracts / "bigquery_training_contract.json", bigquery_training_contract())
    _write_json(contracts / "label_review_schema.json", label_review_schema())
    for source_name in ("galaxy_watch8", "polar_h10"):
        _write_json(contracts / f"source_schema_{source_name}.json", source_schema(source_name))
    for variant in SENSOR_VARIANTS:
        _write_json(
            contracts / f"feature_schema_{variant}.json",
            feature_schema_for_variant(variant),
        )

    training_mae, validation_mae = _fixture_metrics()
    cohort_digest = hashlib.sha256(b"kidsignal-local-contract-fixture-v1").hexdigest()
    references: dict[str, ReferenceBundleResult] = {}
    for index, variant in enumerate(SENSOR_VARIANTS, start=1):
        references[variant] = export_simple_reference_bundle(
            output_dir=root / "references" / variant,
            sensor_variant=variant,
            model_release_uuid=f"40000000-0000-4000-8000-{index:012d}",
            training_cohort_uuid="10000000-0000-4000-8000-000000000001",
            cohort_digest=cohort_digest,
            training_mae=training_mae,
            validation_mae=validation_mae,
            locked_holdout_read=False,
            evidence_scope="LOCAL_CONTRACT_FIXTURE",
        )

    evaluation = build_evaluation_manifest(
        evaluation_uuid="30000000-0000-4000-8000-000000000001",
        model_release_uuid="40000000-0000-4000-8000-000000000001",
        training_cohort_uuid="10000000-0000-4000-8000-000000000001",
        cohort_digest=cohort_digest,
        sensor_variant="watch_only",
        evaluated_cases=[],
        locked_holdout_read=False,
    )
    _write_json(root / "evaluation_manifest.json", evaluation)
    evaluation_input_digest = hashlib.sha256(
        b"kidsignal-local-empty-evaluation-input-v1"
    ).hexdigest()
    comparison = build_comparison_manifest(
        comparison_uuid="60000000-0000-4000-8000-000000000001",
        training_cohort_uuid="10000000-0000-4000-8000-000000000001",
        cohort_digest=cohort_digest,
        evaluation_input_digest=evaluation_input_digest,
        active_model_release_uuid="40000000-0000-4000-8000-000000000001",
        candidate_model_release_uuid="40000000-0000-4000-8000-000000000002",
        active_evaluation_input_digest=evaluation_input_digest,
        candidate_evaluation_input_digest=evaluation_input_digest,
        reviewed_metric_rows=[],
    )
    _write_json(root / "comparison_manifest.json", comparison)
    _write_json(
        root / "candidate_adapter_status.json",
        {
            "status": "BLOCKED_NO_FROZEN_REAL_COHORT",
            "threshold_status": "NOT FIT",
            "adapter_status": "NOT FIT",
            "locked_holdout_read": False,
            "required_inputs": [
                "authorized_or_materialized_training_view",
                "training_cohort_uuid",
                "cohort_digest",
                "independent_label_revisions_and_reviews",
            ],
        },
    )
    manifest = {
        "handoff_version": HANDOFF_VERSION,
        "platform_contract_version": PLATFORM_CONTRACT_VERSION,
        "evidence_scope": "LOCAL_CONTRACT_FIXTURE",
        "sensor_variants": list(SENSOR_VARIANTS),
        "training_ready_variants": ["watch_only"],
        "non_training_ready_variants": {
            "h10_only": "PROPOSED_NOT_IMPLEMENTED",
            "watch_h10": "PROPOSED_NOT_APPROVED",
        },
        "live_bigquery_status": "NOT VERIFIED",
        "real_accuracy_status": "NOT VERIFIED",
        "deployment_eligible": False,
        "candidate_adapter_status": "BLOCKED_NO_FROZEN_REAL_COHORT",
        "reference_bundle_paths": {
            variant: str(result.root.relative_to(root)) for variant, result in references.items()
        },
        "cloud_handoff": {
            "artifacts": [
                "contracts/",
                "references/<sensor_variant>/",
                "evaluation_manifest.json",
                "comparison_manifest.json",
                "candidate_adapter_status.json",
                "SHA256SUMS.json",
            ],
            "runtime": "onnxruntime CPUExecutionProvider",
            "serving_action": "DO_NOT_DEPLOY_FIXTURE",
        },
    }
    manifest_path = root / "handoff_manifest.json"
    _write_json(manifest_path, manifest)
    checksums = {
        str(path.relative_to(root)): _file_sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    checksums_path = root / "SHA256SUMS.json"
    _write_json(checksums_path, checksums)
    return PlatformHandoffResult(root, manifest_path, checksums_path, references)
