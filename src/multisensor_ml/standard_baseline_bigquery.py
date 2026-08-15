"""Read-only Kidsignal standard-baseline BigQuery contract.

This plane is intentionally separate from the behavior-truth 26-field reader.
It validates hourly no-pattern standard labels and never fits a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Final, cast
from uuid import UUID

import numpy as np

from multisensor_ml.observational_contract import FEATURE_NAMES, FEATURE_SCHEMA_SHA256

GCP_PROJECT: Final[str] = "multi-app-kidsignal-260801"
GCP_LOCATION: Final[str] = "asia-southeast1"
STANDARD_BASELINE_AUTHORIZED_VIEW: Final[str] = (
    "multi-app-kidsignal-260801.kidsignal_model_training."
    "standard_baseline_train_validation_v1"
)
EXPECTED_MODEL_READER: Final[str] = (
    "kidsignal-model-reader@multi-app-kidsignal-260801.iam.gserviceaccount.com"
)
STANDARD_BASELINE_UPSTREAM_COMMIT: Final[str] = (
    "bce5b0f8c54266e28dde1b7780c2a228bf50622c"
)
STANDARD_BASELINE_CONTRACT_VERSION: Final[str] = (
    "kidsignal-standard-baseline-handoff/v1"
)
STANDARD_BASELINE_RECEIPT_VERSION: Final[str] = (
    "kidsignal-standard-baseline-readiness/v1"
)
PUBLIC_DIGEST_PIN_POLICY: Final[str] = (
    "EXPECTED_PUBLIC_COHORT_AND_SPLIT_DIGEST_PIN"
)
CANONICAL_HANDOFF_DIGEST_POLICY: Final[str] = (
    "RECOMPUTED_AFTER_EXPECTED_PUBLIC_DIGEST_MATCH"
)
WATCH_SCHEMA_UUID: Final[str] = "9b842d8c-8889-5259-acca-77baa0c7729d"
WATCH_SCHEMA_HASH: Final[str] = FEATURE_SCHEMA_SHA256
SUPPORTED_STANDARD_VERSION: Final[str] = "stable-stress-standard-v1"
ELIGIBILITY_POLICY: Final[str] = "ELIGIBLE_NO_PATTERN_REAL"
TARGET_NAME: Final[str] = "no_pattern_median"
TARGET_UNIT: Final[str] = "positive_robust_z"


@dataclass(frozen=True, slots=True)
class PreparedStandardDataset:
    """Validated hourly arrays for the standard regression trainer boundary."""

    task: str
    standard_cohort_uuid: str
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

STANDARD_BASELINE_ENVELOPE_FIELDS: Final[tuple[str, ...]] = (
    "standard_cohort_uuid",
    "cohort_digest",
    "split_digest",
    "public_cohort_digest",
    "public_split_digest",
    "stable_standard_version",
    "feature_schema_uuid",
    "feature_schema_hash",
    "split_policy",
    "purge_seconds",
    "eligibility_policy",
)
STANDARD_BASELINE_ROW_FIELDS: Final[tuple[str, ...]] = (
    "training_subject_uuid",
    "training_capture_set_uuid",
    "standard_hour_uuid",
    "source_set",
    "hour_start_ms",
    "hour_end_ms",
    "feature_values",
    "target_name",
    "target_value",
    "target_unit",
    "valid_coverage_seconds",
    "split_role",
    "source_row_digest",
)
STANDARD_BASELINE_VIEW_FIELDS: Final[tuple[str, ...]] = (
    *STANDARD_BASELINE_ENVELOPE_FIELDS,
    *STANDARD_BASELINE_ROW_FIELDS,
)
_STANDARD_VIEW_FIELD_SET: Final[frozenset[str]] = frozenset(
    STANDARD_BASELINE_VIEW_FIELDS
)
_STANDARD_ROW_FIELD_SET: Final[frozenset[str]] = frozenset(
    STANDARD_BASELINE_ROW_FIELDS
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_uuid(
    value: object,
    field: str,
    *,
    versions: frozenset[int] | None = None,
) -> str:
    try:
        parsed = UUID(str(value))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a canonical UUID") from error
    if str(parsed) != str(value).lower():
        raise ValueError(f"{field} must be a canonical UUID")
    if versions is not None and parsed.version not in versions:
        raise ValueError(f"{field} has unsupported UUID version")
    return str(parsed)


def _require_sha256(value: object, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


def prepare_standard_dataset_handoff(
    *,
    rows: Sequence[Mapping[str, object]],
    expected_public_cohort_digest: str,
    expected_public_split_digest: str,
) -> PreparedStandardDataset:
    """Validate pins and expose typed arrays without invoking model fitting."""

    expected_cohort = _require_sha256(
        expected_public_cohort_digest, "expected_public_cohort_digest"
    )
    expected_split = _require_sha256(
        expected_public_split_digest, "expected_public_split_digest"
    )
    handoff = synchronize_standard_baseline_rows(rows)
    if (
        handoff["public_cohort_digest"] != expected_cohort
        or handoff["public_split_digest"] != expected_split
    ):
        raise ValueError("returned public digests do not match expected handoff")
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
        if not selected:
            raise ValueError(f"{split} support is required")
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
            [row["target_value"] for row in selected], dtype=np.float32
        )
        features.setflags(write=False)
        targets.setflags(write=False)
        return (
            features,
            targets,
            tuple(str(row["training_subject_uuid"]) for row in selected),
            tuple(str(row["training_capture_set_uuid"]) for row in selected),
            tuple(str(row["standard_hour_uuid"]) for row in selected),
        )

    train = prepare_split("TRAIN")
    validation = prepare_split("VALIDATION")
    return PreparedStandardDataset(
        task="stable_standard_regression",
        standard_cohort_uuid=str(handoff["standard_cohort_uuid"]),
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


def _row_mapping(row: object) -> dict[str, object]:
    if isinstance(row, Mapping):
        return {str(key): value for key, value in row.items()}
    items = getattr(row, "items", None)
    if not callable(items):
        raise ValueError("standard authorized row must be a mapping or Row-like object")
    try:
        return {str(key): value for key, value in items()}
    except (TypeError, ValueError) as error:
        raise ValueError(
            "standard authorized row must be a mapping or Row-like object"
        ) from error


def _validate_exact_schema(schema_fields: Sequence[str]) -> tuple[str, ...]:
    fields = tuple(schema_fields)
    actual = set(fields)
    expected = set(STANDARD_BASELINE_VIEW_FIELDS)
    unexpected = sorted(actual.difference(expected))
    if unexpected:
        raise ValueError(f"forbidden or unknown standard field: {unexpected[0]}")
    missing = sorted(expected.difference(actual))
    if missing:
        raise ValueError(f"missing standard authorized field: {missing[0]}")
    if len(fields) != len(actual):
        raise ValueError("duplicate standard authorized field")
    if fields != STANDARD_BASELINE_VIEW_FIELDS:
        raise ValueError("standard authorized field order does not match contract")
    return fields


def _public_digests(
    rows: Sequence[Mapping[str, object]],
    envelope: Mapping[str, object],
) -> tuple[str, str]:
    ordered_members = sorted(
        [
            row["training_subject_uuid"],
            row["training_capture_set_uuid"],
            row["standard_hour_uuid"],
            row["split_role"],
            row["hour_start_ms"],
            row["source_row_digest"],
        ]
        for row in rows
    )
    split_material = [
        [member[0], member[2], member[3]] for member in ordered_members
    ]
    cohort_material = [
        envelope["stable_standard_version"],
        envelope["feature_schema_uuid"],
        envelope["feature_schema_hash"],
        envelope["split_policy"],
        envelope["purge_seconds"],
        envelope["eligibility_policy"],
        ordered_members,
    ]
    return _sha256_json(cohort_material), _sha256_json(split_material)


def _validate_public_digests(
    rows: Sequence[Mapping[str, object]],
    envelope: Mapping[str, object],
) -> None:
    supplied_cohort = _require_sha256(
        envelope["public_cohort_digest"], "public_cohort_digest"
    )
    supplied_split = _require_sha256(
        envelope["public_split_digest"], "public_split_digest"
    )
    computed_cohort, computed_split = _public_digests(rows, envelope)
    if supplied_cohort != computed_cohort:
        raise ValueError("public_cohort_digest does not match standard rows")
    if supplied_split != computed_split:
        raise ValueError("public_split_digest does not match standard rows")


def synchronize_standard_baseline_rows(rows: Iterable[object]) -> dict[str, object]:
    """Build one deterministic handoff from the separate 24-field view."""

    values = tuple(_row_mapping(row) for row in rows)
    if not values:
        raise ValueError("standard authorized rows must not be empty")
    if any(set(row) != _STANDARD_VIEW_FIELD_SET for row in values):
        raise ValueError("unknown or private standard authorized field")
    envelope = {
        field: values[0][field] for field in STANDARD_BASELINE_ENVELOPE_FIELDS
    }
    if any(
        any(row[field] != envelope[field] for field in STANDARD_BASELINE_ENVELOPE_FIELDS)
        for row in values[1:]
    ):
        raise ValueError("mixed standard handoff envelope")

    projected_rows = [
        {field: row[field] for field in STANDARD_BASELINE_ROW_FIELDS}
        for row in values
    ]
    roles = {str(row["split_role"]) for row in projected_rows}
    if not roles.issubset({"TRAIN", "VALIDATION"}):
        raise ValueError("LOCKED or unknown split_role is forbidden")
    if roles != {"TRAIN", "VALIDATION"}:
        raise ValueError("train and validation support are required")
    hours = [str(row["standard_hour_uuid"]) for row in projected_rows]
    if len(hours) != len(set(hours)):
        raise ValueError("duplicate standard_hour_uuid")
    source_digests = [str(row["source_row_digest"]) for row in projected_rows]
    if len(source_digests) != len(set(source_digests)):
        raise ValueError("duplicate source_row_digest")
    for row in projected_rows:
        features = row["feature_values"]
        if isinstance(features, Mapping) and any(
            type(value) not in {int, float} or not math.isfinite(float(value))
            for value in features.values()
        ):
            raise ValueError("feature_values must be flat finite numeric values")
    _validate_public_digests(values, envelope)

    projected_rows.sort(
        key=lambda row: (
            str(row["split_role"]),
            str(row["training_subject_uuid"]),
            int(cast(int, row["hour_start_ms"])),
            str(row["standard_hour_uuid"]),
        )
    )
    body: dict[str, object] = {
        "contract_version": STANDARD_BASELINE_CONTRACT_VERSION,
        "upstream_contract_commit": STANDARD_BASELINE_UPSTREAM_COMMIT,
        **envelope,
        "sequence_group_key": "training_capture_set_uuid",
        "locked_access": False,
        "rows": projected_rows,
    }
    handoff = {**body, "handoff_digest": _sha256_json(body)}
    validate_standard_baseline_handoff(handoff)
    return handoff


def validate_standard_baseline_handoff(
    handoff: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "contract_version",
        "upstream_contract_commit",
        *STANDARD_BASELINE_ENVELOPE_FIELDS,
        "sequence_group_key",
        "locked_access",
        "rows",
        "handoff_digest",
    }
    missing = sorted(required.difference(handoff))
    if missing:
        raise ValueError(f"standard handoff missing {missing[0]}")
    extras = sorted(set(handoff).difference(required))
    if extras:
        raise ValueError(f"standard handoff has unknown field {extras[0]}")
    if handoff["contract_version"] != STANDARD_BASELINE_CONTRACT_VERSION:
        raise ValueError("standard contract_version is invalid")
    if handoff["upstream_contract_commit"] != STANDARD_BASELINE_UPSTREAM_COMMIT:
        raise ValueError("upstream standard contract commit is invalid")
    supplied_handoff_digest = _require_sha256(
        handoff["handoff_digest"], "handoff_digest"
    )
    digest_body = {
        str(key): value for key, value in handoff.items() if key != "handoff_digest"
    }
    if _sha256_json(digest_body) != supplied_handoff_digest:
        raise ValueError("handoff_digest does not match standard handoff body")

    _canonical_uuid(
        handoff["standard_cohort_uuid"],
        "standard_cohort_uuid",
        versions=frozenset({4}),
    )
    for field in (
        "cohort_digest",
        "split_digest",
        "public_cohort_digest",
        "public_split_digest",
        "feature_schema_hash",
    ):
        _require_sha256(handoff[field], field)
    if (handoff["feature_schema_uuid"], handoff["feature_schema_hash"]) != (
        WATCH_SCHEMA_UUID,
        WATCH_SCHEMA_HASH,
    ):
        raise ValueError("standard feature schema is not canonical Watch 16")
    if handoff["stable_standard_version"] != SUPPORTED_STANDARD_VERSION:
        raise ValueError("stable_standard_version is unsupported")
    if handoff["eligibility_policy"] != ELIGIBILITY_POLICY:
        raise ValueError("eligibility_policy must be ELIGIBLE_NO_PATTERN_REAL")
    split_policy = str(handoff["split_policy"])
    if split_policy not in {"PERSON_GROUP", "CHRONOLOGICAL_PER_SUBJECT"}:
        raise ValueError("unsupported split_policy")
    purge_seconds = handoff["purge_seconds"]
    if type(purge_seconds) is not int or purge_seconds < 1800:
        raise ValueError("purge_seconds must be an integer of at least 1800")
    if handoff["sequence_group_key"] != "training_capture_set_uuid":
        raise ValueError("sequence_group_key must be training_capture_set_uuid")
    if handoff["locked_access"] is not False:
        raise ValueError("LOCKED access is forbidden in standard handoff")

    rows = handoff["rows"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("standard handoff rows must be a non-empty sequence")
    split_counts = {"TRAIN": 0, "VALIDATION": 0}
    exact_hours: set[str] = set()
    source_digests: set[str] = set()
    subjects_by_split: dict[str, set[str]] = {"TRAIN": set(), "VALIDATION": set()}
    subjects_by_capture: dict[str, set[str]] = {}
    bounds_by_subject_split: dict[tuple[str, str], list[tuple[int, int]]] = {}
    row_order_keys: list[tuple[str, str, int, str]] = []
    public_rows: list[dict[str, object]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"rows[{index}] must be an object")
        if set(raw) != _STANDARD_ROW_FIELD_SET:
            raise ValueError("unknown or private standard handoff row field")
        subject = _canonical_uuid(
            raw["training_subject_uuid"],
            "training_subject_uuid",
            versions=frozenset({5}),
        )
        capture = _canonical_uuid(
            raw["training_capture_set_uuid"],
            "training_capture_set_uuid",
            versions=frozenset({5}),
        )
        hour = _canonical_uuid(
            raw["standard_hour_uuid"],
            "standard_hour_uuid",
            versions=frozenset({5}),
        )
        source_digest = _require_sha256(
            raw["source_row_digest"], "source_row_digest"
        )
        if hour in exact_hours:
            raise ValueError("duplicate standard_hour_uuid")
        if source_digest in source_digests:
            raise ValueError("duplicate source_row_digest")
        exact_hours.add(hour)
        source_digests.add(source_digest)
        if raw["source_set"] != ["watch"]:
            raise ValueError("standard source_set must be exactly ['watch']")
        start = raw["hour_start_ms"]
        end = raw["hour_end_ms"]
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or start % 3_600_000 != 0
            or end != start + 3_600_000
        ):
            raise ValueError("standard UTC hour bounds are invalid")
        features = raw["feature_values"]
        if not isinstance(features, Mapping) or set(features) != set(FEATURE_NAMES):
            raise ValueError("feature_values must use exact Watch 16 keys")
        if any(
            type(features[name]) not in {int, float}
            or not math.isfinite(float(cast(float, features[name])))
            for name in FEATURE_NAMES
        ):
            raise ValueError("feature_values must be flat finite numeric values")
        if raw["target_name"] != TARGET_NAME:
            raise ValueError("target_name must be no_pattern_median")
        if raw["target_unit"] != TARGET_UNIT:
            raise ValueError("target_unit must be positive_robust_z")
        target = raw["target_value"]
        if (
            type(target) not in {int, float}
            or not math.isfinite(float(cast(float, target)))
            or float(cast(float, target)) < 0
        ):
            raise ValueError("target_value must be finite and nonnegative")
        if float(cast(float, target)) != float(
            cast(float, features["watch_load_median_300"])
        ):
            raise ValueError("target_value must equal watch_load_median_300")
        coverage = raw["valid_coverage_seconds"]
        if type(coverage) is not int or not 1 <= coverage <= 3600:
            raise ValueError("valid_coverage_seconds must be from 1 to 3600")
        split = str(raw["split_role"])
        if split not in split_counts:
            raise ValueError("LOCKED or unknown split_role is forbidden")
        split_counts[split] += 1
        subjects_by_split[split].add(subject)
        subjects_by_capture.setdefault(capture, set()).add(subject)
        bounds_by_subject_split.setdefault((subject, split), []).append((start, end))
        row_order_keys.append((split, subject, start, hour))
        public_rows.append(
            {
                **{field: handoff[field] for field in STANDARD_BASELINE_ENVELOPE_FIELDS},
                **{field: raw[field] for field in STANDARD_BASELINE_ROW_FIELDS},
            }
        )

    if split_counts["TRAIN"] < 1 or split_counts["VALIDATION"] < 1:
        raise ValueError("train and validation support are required")
    if any(len(subjects) != 1 for subjects in subjects_by_capture.values()):
        raise ValueError("training capture set spans subjects")
    if row_order_keys != sorted(row_order_keys):
        raise ValueError("standard handoff rows are not in canonical order")
    if split_policy == "PERSON_GROUP":
        if subjects_by_split["TRAIN"].intersection(subjects_by_split["VALIDATION"]):
            raise ValueError("person-group split leakage")
    else:
        purge_ms = purge_seconds * 1000
        for subject in subjects_by_split["TRAIN"].intersection(
            subjects_by_split["VALIDATION"]
        ):
            train_end = max(
                end_value
                for _, end_value in bounds_by_subject_split[(subject, "TRAIN")]
            )
            validation_start = min(
                start_value
                for start_value, _ in bounds_by_subject_split[
                    (subject, "VALIDATION")
                ]
            )
            if train_end + purge_ms > validation_start:
                raise ValueError("chronological target purge violated")
    _validate_public_digests(public_rows, handoff)
    return {
        "status": "VALID_STANDARD_BASELINE_HANDOFF",
        "training_eligible": True,
        "split_counts": split_counts,
        "sequence_group_count": len(subjects_by_capture),
        "sequence_group_key": "training_capture_set_uuid",
        "target_name": TARGET_NAME,
        "target_unit": TARGET_UNIT,
        "locked_access": False,
    }


def _schema_digest(fields: Sequence[str]) -> str:
    return _sha256_json(list(fields))


def _blocked_receipt(
    *,
    status: str,
    requested_uuid: str,
    observed_at_utc: str,
    observed_principal: str,
    schema_fields: Sequence[str],
    row_count: int,
    blocked_reason: str,
    expected_public_cohort_digest: str | None = None,
    expected_public_split_digest: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": STANDARD_BASELINE_RECEIPT_VERSION,
        "upstream_contract_commit": STANDARD_BASELINE_UPSTREAM_COMMIT,
        "observed_at_utc": observed_at_utc,
        "mode": "READ_ONLY",
        "authorized_view": STANDARD_BASELINE_AUTHORIZED_VIEW,
        "requested_standard_cohort_uuid": requested_uuid,
        "standard_cohort_uuid": requested_uuid,
        "row_count": row_count,
        "split_counts": {"TRAIN": 0, "VALIDATION": 0},
        "schema_fields": list(schema_fields),
        "schema_field_sha256": _schema_digest(schema_fields),
        "observed_principal": observed_principal,
        "model_reader_identity_verified": observed_principal == EXPECTED_MODEL_READER,
        "locked_access": False,
        "training_ready": False,
        "fit_call_count": 0,
        "training_status": "NOT STARTED",
        "evaluation_status": "NOT EVALUABLE",
        "real_performance_status": "NOT VERIFIED",
        "status": status,
        "blocked_reason": blocked_reason,
        "canonical_handoff_digest": None,
        "public_cohort_digest": None,
        "public_split_digest": None,
        "expected_public_cohort_digest": expected_public_cohort_digest,
        "expected_public_split_digest": expected_public_split_digest,
        "expected_public_digests_match": False,
        "canonical_handoff_digest_policy": CANONICAL_HANDOFF_DIGEST_POLICY,
        "public_digest_policy": PUBLIC_DIGEST_PIN_POLICY,
    }


def build_standard_cohort_readiness_receipt(
    *,
    rows: Sequence[Mapping[str, object]],
    schema_fields: Sequence[str],
    requested_standard_cohort_uuid: str,
    expected_public_cohort_digest: str,
    expected_public_split_digest: str,
    observed_at_utc: str,
    observed_principal: str,
) -> dict[str, object]:
    """Validate a standard cohort without fitting or exposing its rows."""

    requested_uuid = _canonical_uuid(
        requested_standard_cohort_uuid,
        "standard_cohort_uuid",
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
        return _blocked_receipt(
            status="BLOCKED_NO_REAL_COHORT",
            requested_uuid=requested_uuid,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            schema_fields=fields,
            row_count=0,
            blocked_reason="AUTHORIZED_STANDARD_VIEW_RETURNED_ZERO_ROWS",
            expected_public_cohort_digest=expected_cohort_digest,
            expected_public_split_digest=expected_split_digest,
        )
    if any(row.get("standard_cohort_uuid") != requested_uuid for row in values):
        return _blocked_receipt(
            status="BLOCKED_MIXED_OR_INVALID_COHORT",
            requested_uuid=requested_uuid,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            schema_fields=fields,
            row_count=len(values),
            blocked_reason="REQUESTED_STANDARD_COHORT_UUID_MISMATCH",
            expected_public_cohort_digest=expected_cohort_digest,
            expected_public_split_digest=expected_split_digest,
        )
    try:
        handoff = synchronize_standard_baseline_rows(values)
        validation = validate_standard_baseline_handoff(handoff)
    except (TypeError, ValueError) as error:
        return _blocked_receipt(
            status="BLOCKED_MIXED_OR_INVALID_COHORT",
            requested_uuid=requested_uuid,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            schema_fields=fields,
            row_count=len(values),
            blocked_reason=str(error),
            expected_public_cohort_digest=expected_cohort_digest,
            expected_public_split_digest=expected_split_digest,
        )
    if (
        handoff["public_cohort_digest"] != expected_cohort_digest
        or handoff["public_split_digest"] != expected_split_digest
    ):
        return _blocked_receipt(
            status="BLOCKED_EXPECTED_PUBLIC_DIGEST_MISMATCH",
            requested_uuid=requested_uuid,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            schema_fields=fields,
            row_count=len(values),
            blocked_reason="RETURNED_PUBLIC_DIGESTS_DO_NOT_MATCH_EXPECTED_HANDOFF",
            expected_public_cohort_digest=expected_cohort_digest,
            expected_public_split_digest=expected_split_digest,
        )
    identity_verified = observed_principal == EXPECTED_MODEL_READER
    status = (
        "READY_FOR_SYNC_NOT_TRAINED"
        if identity_verified
        else "BLOCKED_MODEL_READER_IDENTITY_NOT_VERIFIED"
    )
    return {
        "schema_version": STANDARD_BASELINE_RECEIPT_VERSION,
        "upstream_contract_commit": STANDARD_BASELINE_UPSTREAM_COMMIT,
        "observed_at_utc": observed_at_utc,
        "mode": "READ_ONLY",
        "authorized_view": STANDARD_BASELINE_AUTHORIZED_VIEW,
        "requested_standard_cohort_uuid": requested_uuid,
        "standard_cohort_uuid": str(handoff["standard_cohort_uuid"]),
        "cohort_digest": str(handoff["cohort_digest"]),
        "split_digest": str(handoff["split_digest"]),
        "public_cohort_digest": str(handoff["public_cohort_digest"]),
        "public_split_digest": str(handoff["public_split_digest"]),
        "expected_public_cohort_digest": expected_cohort_digest,
        "expected_public_split_digest": expected_split_digest,
        "expected_public_digests_match": True,
        "stable_standard_version": str(handoff["stable_standard_version"]),
        "row_count": len(values),
        "split_counts": validation["split_counts"],
        "target_name": TARGET_NAME,
        "target_unit": TARGET_UNIT,
        "schema_fields": list(fields),
        "schema_field_sha256": _schema_digest(fields),
        "observed_principal": observed_principal,
        "model_reader_identity_verified": identity_verified,
        "locked_access": False,
        "training_ready": status == "READY_FOR_SYNC_NOT_TRAINED",
        "fit_call_count": 0,
        "training_status": "NOT STARTED",
        "evaluation_status": "NOT EVALUABLE",
        "real_performance_status": "NOT VERIFIED",
        "status": status,
        "blocked_reason": (
            None if identity_verified else "DEDICATED_MODEL_READER_REQUIRED"
        ),
        "canonical_handoff_digest": str(handoff["handoff_digest"]),
        "canonical_handoff_digest_policy": CANONICAL_HANDOFF_DIGEST_POLICY,
        "public_digest_policy": PUBLIC_DIGEST_PIN_POLICY,
    }


def _run(command: list[str]) -> CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _parse_schema(stdout: str) -> tuple[str, ...]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError("BigQuery standard schema output is not JSON") from error
    if not isinstance(value, list):
        raise ValueError("BigQuery standard schema output must be a list")
    fields: list[str] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("BigQuery standard schema field is invalid")
        fields.append(item["name"])
    return tuple(fields)


def _parse_rows(stdout: str) -> tuple[dict[str, object], ...]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ValueError("BigQuery standard cohort output is not JSON") from error
    if not isinstance(value, list):
        raise ValueError("BigQuery standard cohort output must be a list")
    output: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"BigQuery standard cohort row {index} is invalid")
        row = {str(key): field for key, field in item.items()}
        for field in (
            "purge_seconds",
            "hour_start_ms",
            "hour_end_ms",
            "valid_coverage_seconds",
        ):
            try:
                row[field] = int(cast(str | int, row[field]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"BigQuery standard cohort row {index} has invalid {field}"
                ) from error
        try:
            row["target_value"] = float(cast(str | float, row["target_value"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"BigQuery standard cohort row {index} has invalid target_value"
            ) from error
        features = row.get("feature_values")
        if isinstance(features, str):
            try:
                row["feature_values"] = json.loads(features)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"BigQuery standard cohort row {index} has invalid feature_values"
                ) from error
        output.append(row)
    return tuple(output)


def _write_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_read_only_standard_cohort_reader(
    output_receipt: Path,
    *,
    standard_cohort_uuid: str,
    expected_public_cohort_digest: str,
    expected_public_split_digest: str,
    observed_at_utc: str,
    observed_principal: str,
    runner: Callable[[list[str]], CompletedProcess[str]] = _run,
) -> dict[str, object]:
    """Read exactly one standard cohort from the sole authorized view."""

    cohort_uuid = _canonical_uuid(
        standard_cohort_uuid,
        "standard_cohort_uuid",
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
    target = STANDARD_BASELINE_AUTHORIZED_VIEW.replace(".", ":", 1)
    common = [
        "bq",
        f"--project_id={GCP_PROJECT}",
        f"--location={GCP_LOCATION}",
    ]
    schema_result = runner(
        [*common, "show", "--schema", "--format=prettyjson", target]
    )
    if schema_result.returncode != 0:
        raise RuntimeError(
            f"BigQuery standard schema read failed: {schema_result.stderr.strip()}"
        )
    parsed_schema = _parse_schema(schema_result.stdout)
    try:
        _validate_exact_schema(parsed_schema)
    except ValueError as error:
        receipt = _blocked_receipt(
            status="BLOCKED_SCHEMA_CONTRACT_MISMATCH",
            requested_uuid=cohort_uuid,
            observed_at_utc=observed_at_utc,
            observed_principal=observed_principal,
            schema_fields=parsed_schema,
            row_count=0,
            blocked_reason=str(error),
            expected_public_cohort_digest=expected_cohort_digest,
            expected_public_split_digest=expected_split_digest,
        )
        _write_receipt(output_receipt, receipt)
        return receipt
    selected_fields = ", ".join(STANDARD_BASELINE_VIEW_FIELDS)
    query = (
        f"SELECT {selected_fields} FROM `{STANDARD_BASELINE_AUTHORIZED_VIEW}` "
        "WHERE standard_cohort_uuid=@standard_cohort_uuid "
        "ORDER BY split_role, training_subject_uuid, hour_start_ms, standard_hour_uuid"
    )
    rows_result = runner(
        [
            *common,
            "query",
            "--use_legacy_sql=false",
            "--format=json",
            f"--parameter=standard_cohort_uuid:STRING:{cohort_uuid}",
            query,
        ]
    )
    if rows_result.returncode != 0:
        raise RuntimeError(
            f"BigQuery standard cohort read failed: {rows_result.stderr.strip()}"
        )
    receipt = build_standard_cohort_readiness_receipt(
        rows=_parse_rows(rows_result.stdout),
        schema_fields=parsed_schema,
        requested_standard_cohort_uuid=cohort_uuid,
        expected_public_cohort_digest=expected_cohort_digest,
        expected_public_split_digest=expected_split_digest,
        observed_at_utc=observed_at_utc,
        observed_principal=observed_principal,
    )
    _write_receipt(output_receipt, receipt)
    return receipt
