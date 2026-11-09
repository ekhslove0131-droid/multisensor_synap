"""Proposed physiology-only authorized handoff v1; no private-table fallback."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any
from uuid import UUID

from multisensor_ml.physiology_compatibility import consume_descriptors, load_schema

PROJECT = "multi-app-kidsignal-260801"
LOCATION = "asia-southeast1"
AUTHORIZED_VIEW = f"{PROJECT}.kidsignal_model_training.physiology_baseline_train_validation_v1"
ENVELOPE_FIELDS = (
    "cohort_uuid",
    "source_kind",
    "feature_schema_uuid",
    "feature_schema_hash",
    "native_hr_policy",
    "split_policy",
    "purge_seconds",
    "eligibility_policy",
    "origin",
)
VIEW_FIELDS = (
    *ENVELOPE_FIELDS,
    "public_cohort_digest",
    "public_split_digest",
    "training_subject_uuid",
    "training_capture_set_uuid",
    "exact_window_id",
    "source_row_digest",
    "window_start_ms",
    "window_end_ms",
    "support_start_ms",
    "support_end_ms",
    "feature_values",
    "quality_valid",
    "training_eligible",
    "admission",
    "no_pattern_review_uuid",
    "split_role",
)
RECEIPT_FIELDS = (
    *ENVELOPE_FIELDS,
    "contract_version",
    "public_cohort_digest",
    "public_split_digest",
    "train_row_count",
    "validation_row_count",
    "locked_access",
    "fit_call_count",
    "handoff_digest",
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _uuid(value: object, versions: tuple[int, ...]) -> None:
    if not isinstance(value, str):
        raise ValueError("UUID_INVALID")
    parsed = UUID(value)
    if str(parsed) != value or parsed.version not in versions:
        raise ValueError("UUID_INVALID")


def _sha(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("DIGEST_INVALID")


def validate_receipt(receipt: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    if set(receipt) != set(RECEIPT_FIELDS):
        raise ValueError("RECEIPT_FIELDS")
    if receipt["contract_version"] != "physiology_baseline_handoff_v1":
        raise ValueError("CONTRACT_VERSION")
    _uuid(receipt["cohort_uuid"], (4,))
    doc = load_schema(schema, receipt["source_kind"])
    if receipt["feature_schema_uuid"] != doc["schema_uuid"] or receipt[
        "feature_schema_hash"
    ] != digest(doc):
        raise ValueError("SCHEMA_IDENTITY")
    for name in ("public_cohort_digest", "public_split_digest", "handoff_digest"):
        _sha(receipt[name])
    if receipt["handoff_digest"] != digest(
        {k: v for k, v in receipt.items() if k != "handoff_digest"}
    ):
        raise ValueError("RECEIPT_DIGEST")
    if (
        receipt["locked_access"] is not False
        or type(receipt["fit_call_count"]) is not int
        or receipt["fit_call_count"] != 0
    ):
        raise ValueError("FIT_OR_LOCKED_ACCESS")
    if receipt["split_policy"] != "CHRONOLOGICAL_PER_SUBJECT":
        raise ValueError("PERSONAL_BASELINE_REQUIRES_CHRONOLOGICAL_SPLIT")
    if type(receipt["purge_seconds"]) is not int or receipt["purge_seconds"] < 0:
        raise ValueError("PURGE_INVALID")
    if receipt["eligibility_policy"] != "REVIEWED_NO_PATTERN_V1" or receipt["origin"] not in (
        "REAL_OBSERVED",
        "CONTRACT_FIXTURE",
    ):
        raise ValueError("ADMISSION_POLICY")
    expected_hr = "polar_native_hr_bpm_v1" if receipt["source_kind"] == "h10" else "NOT_APPLICABLE"
    if receipt["native_hr_policy"] != expected_hr:
        raise ValueError("NATIVE_HR_POLICY")
    for key in ("train_row_count", "validation_row_count"):
        if type(receipt[key]) is not int or receipt[key] <= 0:
            raise ValueError("SPLIT_COUNT")
    return dict(json.loads(canonical_bytes(receipt)))


@dataclass(frozen=True)
class PreparedPhysiologyDataset:
    feature_names: tuple[str, ...]
    # Immutable (subject, role, ordered vector), never raw or private identity.
    samples: tuple[tuple[str, str, tuple[float, ...]], ...]
    receipt_json: bytes


def prepare_dataset(
    rows: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any], schema: Mapping[str, Any]
) -> PreparedPhysiologyDataset:
    bound = validate_receipt(receipt, schema)
    if not rows:
        raise ValueError("NO_REAL_COHORT")
    if any(set(row) != set(VIEW_FIELDS) for row in rows):
        raise ValueError("ROW_FIELDS")
    ordered = sorted(
        rows, key=lambda r: (r["training_subject_uuid"], r["window_start_ms"], r["exact_window_id"])
    )
    seen_windows: set[str] = set()
    seen_digests: set[str] = set()
    owners: dict[str, str] = {}
    spans: dict[str, dict[str, list[tuple[int, int]]]] = {}
    samples = []
    specs = {d["code"]: d for d in schema["descriptors"]}
    for row in ordered:
        if any(
            row[k] != bound[k] or type(row[k]) is not type(bound[k])
            for k in (*ENVELOPE_FIELDS, "public_cohort_digest", "public_split_digest")
        ):
            raise ValueError("MIXED_ENVELOPE")
        subject, capture = row["training_subject_uuid"], row["training_capture_set_uuid"]
        _uuid(subject, (5,))
        _uuid(capture, (5,))
        _uuid(row["no_pattern_review_uuid"], (4,))
        if owners.setdefault(capture, subject) != subject:
            raise ValueError("CAPTURE_SUBJECT_MIXED")
        for key, seen in (("exact_window_id", seen_windows), ("source_row_digest", seen_digests)):
            _sha(row[key])
            if row[key] in seen:
                raise ValueError("DUPLICATE_ROW")
            seen.add(row[key])
        if (
            row["quality_valid"] is not True
            or row["training_eligible"] is not True
            or row["admission"] != "ELIGIBLE_NO_PATTERN"
        ):
            raise ValueError("NO_PATTERN_OR_QUALITY_UNAPPROVED")
        role = row["split_role"]
        if role not in ("TRAIN", "VALIDATION"):
            raise ValueError("LOCKED_OR_UNKNOWN_SPLIT")
        for key in ("window_start_ms", "window_end_ms", "support_start_ms", "support_end_ms"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("SUPPORT_INVALID")
        if (
            row["window_end_ms"] - row["window_start_ms"] != 10000
            or row["support_start_ms"] != row["window_start_ms"]
            or row["support_end_ms"] != row["window_end_ms"]
        ):
            raise ValueError("SUPPORT_INVALID")
        spans.setdefault(subject, {}).setdefault(role, []).append(
            (row["support_start_ms"], row["support_end_ms"])
        )
        features = row["feature_values"]
        if not isinstance(features, Mapping) or set(features) != set(schema["core_features"]):
            raise ValueError("FEATURE_FIELDS")
        vector = consume_descriptors(
            schema,
            [
                dict(
                    code=name,
                    unit=specs[name]["unit"],
                    value=value,
                    readiness="AVAILABLE",
                    reason=None,
                )
                for name, value in features.items()
            ],
        )
        if vector is None:
            raise ValueError("CORE_MISSING")
        samples.append((subject, role, vector))
    for roles in spans.values():
        if set(roles) != {"TRAIN", "VALIDATION"}:
            raise ValueError("SUBJECT_SPLIT_SUPPORT")
        for intervals in roles.values():
            intervals.sort()
            if any(b[0] < a[1] for a, b in pairwise(intervals)):
                raise ValueError("OVERLAPPING_SUPPORT")
        if min(s for s, _ in roles["VALIDATION"]) < (
            max(e for _, e in roles["TRAIN"]) + bound["purge_seconds"] * 1000
        ):
            raise ValueError("PURGE_OR_SUPPORT_LEAKAGE")
    for role, key in (("TRAIN", "train_row_count"), ("VALIDATION", "validation_row_count")):
        if sum(r["split_role"] == role for r in rows) != bound[key]:
            raise ValueError("SPLIT_COUNT_MISMATCH")
    members = [
        {k: v for k, v in r.items() if k not in {"public_cohort_digest", "public_split_digest"}}
        for r in ordered
    ]
    envelope = {k: bound[k] for k in ENVELOPE_FIELDS}
    split = [
        [
            r["training_subject_uuid"],
            r["training_capture_set_uuid"],
            r["exact_window_id"],
            r["split_role"],
            r["support_start_ms"],
            r["support_end_ms"],
        ]
        for r in ordered
    ]
    if (
        digest({"envelope": envelope, "rows": members}) != bound["public_cohort_digest"]
        or digest(split) != bound["public_split_digest"]
    ):
        raise ValueError("PUBLIC_DIGEST_MISMATCH")
    return PreparedPhysiologyDataset(
        tuple(schema["core_features"]), tuple(samples), canonical_bytes(bound)
    )


def query_contract(receipt: Mapping[str, Any]) -> dict[str, object]:
    return {
        "query": f"SELECT {', '.join(VIEW_FIELDS)} FROM `{AUTHORIZED_VIEW}` "
        "WHERE cohort_uuid=@cohort_uuid",
        "parameter": {"name": "cohort_uuid", "type": "STRING", "value": receipt["cohort_uuid"]},
        "locked_access": False,
        "allowed_transports": ["google_cloud_bigquery_sdk"],
    }
