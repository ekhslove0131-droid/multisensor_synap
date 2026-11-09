"""Verified local application of an exploratory personal reference.

This module opens no route.  The caller remains responsible for actor
authorization and timestamp freshness before supplying a current feature row.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, NamedTuple, cast

from multisensor_ml.exploratory_personal_reference import _validate_model


class _FeatureSpec(NamedTuple):
    code: str
    unit: str
    domain: str


class _PinnedSchema(NamedTuple):
    uuid: str
    sha256: str
    features: tuple[_FeatureSpec, ...]


class _BaselineFeature(NamedTuple):
    code: str
    unit: str
    center: float
    scale: float


_WATCH_FEATURES = (
    _FeatureSpec("hr_median_bpm", "bpm", "POSITIVE"),
    _FeatureSpec("eda_median_us", "uS", "NONNEGATIVE"),
    _FeatureSpec("eda_slope_us_per_s", "uS/s", "SIGNED_FINITE"),
)
_H10_FEATURES = (
    _FeatureSpec("hr_median_bpm", "bpm", "POSITIVE"),
    _FeatureSpec("rr_median_ms", "ms", "POSITIVE"),
    _FeatureSpec("rr_rmssd_ms", "ms", "NONNEGATIVE"),
)
_MUSE_FEATURES = tuple(
    _FeatureSpec(f"eeg_{channel}_{name}", "ratio", "UNIT_INTERVAL")
    for channel in range(1, 5)
    for name in (
        "delta_relative_power",
        "theta_relative_power",
        "alpha_relative_power",
        "beta_relative_power",
        "gamma_relative_power",
        "spectral_entropy",
    )
)
PINNED_SCHEMAS: Final[dict[str, _PinnedSchema]] = {
    "watch": _PinnedSchema(
        "ce01a4e8-8c3b-4b5c-8a25-e4e217b62d16",
        "4e949a5033e36b9c2f0ec3d348c52088717a0904dc774a923941f468061bc6fb",
        _WATCH_FEATURES,
    ),
    "h10": _PinnedSchema(
        "c191c0cb-c5d0-45f2-bc53-69ab3fe97c1b",
        "055bfcee108dd6e059ddd930eb682ef43760c92bc62ef68cb004953a2ae92ae2",
        _H10_FEATURES,
    ),
    "muse": _PinnedSchema(
        "2d6ea429-1d13-4707-86c4-ddbc2539b8d5",
        "97139686fab24c88ed844ba2d69ea81384b76711a7217a954c994e4eaa334ce9",
        _MUSE_FEATURES,
    ),
}

_RECEIPT_FIELDS: Final = {
    "schema_version",
    "status",
    "source_kind",
    "model_release_uuid",
    "source_dataset_sha256",
    "source_snapshot_sha256",
    "source_lineage_digest",
    "scope_digest",
    "feature_schema_hash",
    "feature_order",
    "parameters_digest",
    "model_sha256",
    "model_content_sha256",
    "train_row_count",
    "validation_row_count",
    "fit_call_count",
    "fixture_fit_call_count",
    "readback_verified",
    "delivery_eligible",
    "promotion_eligible",
    "receipt_content_sha256",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field.upper()}_INVALID")
    return value


def _read_json_once(path: Path, expected_sha256: str, kind: str) -> tuple[dict[str, Any], str]:
    expected = _require_sha256(expected_sha256, f"expected_{kind}_sha256")
    payload = path.read_bytes()
    actual = _sha256(payload)
    if actual != expected:
        raise ValueError(f"{kind.upper()}_FILE_SHA256_MISMATCH")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{kind.upper()}_DOCUMENT_INVALID")
    return cast(dict[str, Any], value), actual


def _validate_receipt(document: dict[str, Any]) -> dict[str, Any]:
    if set(document) != _RECEIPT_FIELDS:
        raise ValueError("RECEIPT_FIELDS_INVALID")
    body = {key: value for key, value in document.items() if key != "receipt_content_sha256"}
    if document["receipt_content_sha256"] != _sha256(_canonical(body)):
        raise ValueError("RECEIPT_CONTENT_DIGEST_MISMATCH")
    for field in (
        "source_dataset_sha256",
        "source_snapshot_sha256",
        "source_lineage_digest",
        "scope_digest",
        "feature_schema_hash",
        "parameters_digest",
        "model_sha256",
        "model_content_sha256",
        "receipt_content_sha256",
    ):
        _require_sha256(document[field], field)
    return document


def _validate_binding(
    model: Mapping[str, Any],
    receipt: Mapping[str, Any],
    model_sha256: str,
) -> None:
    bindings = {
        "schema_version": "exploratory_personal_reference_run_receipt_v1",
        "status": model["model_status"],
        "source_kind": model["source_kind"],
        "model_release_uuid": model["model_release_uuid"],
        "source_dataset_sha256": model["source_dataset_sha256"],
        "source_snapshot_sha256": model["source_snapshot_sha256"],
        "source_lineage_digest": model["source_lineage_digest"],
        "scope_digest": model["scope_digest"],
        "feature_schema_hash": model["feature_schema_hash"],
        "feature_order": model["feature_order"],
        "parameters_digest": model["parameters_digest"],
        "model_sha256": model_sha256,
        "model_content_sha256": model["content_sha256"],
        "fit_call_count": 1,
        "fixture_fit_call_count": 0,
        "readback_verified": True,
        "delivery_eligible": False,
        "promotion_eligible": False,
    }
    if any(receipt.get(key) != value for key, value in bindings.items()):
        raise ValueError("RECEIPT_MODEL_BINDING_MISMATCH")
    for field in ("train_row_count", "validation_row_count"):
        if type(receipt[field]) is not int or receipt[field] <= 0:
            raise ValueError("RECEIPT_MODEL_BINDING_MISMATCH")
    if (
        model["origin"] != "REAL_OBSERVED"
        or model["fit_call_count"] != 1
        or model["fixture_fit_call_count"] != 0
    ):
        raise ValueError("MODEL_NOT_REAL_OBSERVED_REFERENCE")


def _pinned_schema(model: Mapping[str, Any], expected_source_kind: str) -> _PinnedSchema:
    if expected_source_kind not in PINNED_SCHEMAS:
        raise ValueError("EXPECTED_SOURCE_KIND_INVALID")
    schema = PINNED_SCHEMAS[expected_source_kind]
    if (
        model["source_kind"] != expected_source_kind
        or model["feature_schema_uuid"] != schema.uuid
        or model["feature_schema_hash"] != schema.sha256
        or model["feature_order"] != [spec.code for spec in schema.features]
    ):
        raise ValueError("PINNED_FEATURE_SCHEMA_MISMATCH")
    return schema


def _current_value(value: object, domain: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("CURRENT_FEATURE_DOMAIN_INVALID")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("CURRENT_FEATURE_DOMAIN_INVALID")
    accepted = {
        "POSITIVE": number > 0,
        "NONNEGATIVE": number >= 0,
        "UNIT_INTERVAL": 0 <= number <= 1,
        "SIGNED_FINITE": True,
    }.get(domain, False)
    if not accepted:
        raise ValueError("CURRENT_FEATURE_DOMAIN_INVALID")
    return number


def _baseline_features(
    model: Mapping[str, Any], schema: _PinnedSchema
) -> tuple[_BaselineFeature, ...]:
    features: list[_BaselineFeature] = []
    for spec, statistic in zip(schema.features, model["feature_statistics"], strict=True):
        try:
            center = _current_value(statistic["center"], spec.domain)
        except ValueError as error:
            raise ValueError("BASELINE_CENTER_DOMAIN_INVALID") from error
        scale = float(statistic["scale"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("BASELINE_SCALE_INVALID")
        features.append(_BaselineFeature(spec.code, spec.unit, center, scale))
    return tuple(features)


def _read_verified_reference(
    *,
    model_path: Path,
    expected_model_sha256: str,
    receipt_path: Path,
    expected_receipt_sha256: str,
    expected_training_subject_uuid: str,
    expected_source_kind: str,
    expected_feature_schema_hash: str,
) -> tuple[
    dict[str, Any], str, str, _PinnedSchema, tuple[_BaselineFeature, ...]
]:
    model_document, model_sha256 = _read_json_once(
        model_path, expected_model_sha256, "model"
    )
    receipt_document, receipt_sha256 = _read_json_once(
        receipt_path, expected_receipt_sha256, "receipt"
    )
    model = _validate_model(model_document)
    receipt = _validate_receipt(receipt_document)
    _validate_binding(model, receipt, model_sha256)
    schema = _pinned_schema(model, expected_source_kind)
    if model["training_subject_uuid"] != expected_training_subject_uuid:
        raise ValueError("TRAINING_SUBJECT_UUID_MISMATCH")
    expected_hash = _require_sha256(expected_feature_schema_hash, "feature_schema_hash")
    if expected_hash != schema.sha256:
        raise ValueError("FEATURE_SCHEMA_HASH_MISMATCH")
    return model, model_sha256, receipt_sha256, schema, _baseline_features(model, schema)


def read_verified_exploratory_baseline(
    *,
    model_path: Path,
    expected_model_sha256: str,
    receipt_path: Path,
    expected_receipt_sha256: str,
    expected_training_subject_uuid: str,
    expected_source_kind: str,
    expected_feature_schema_hash: str,
) -> dict[str, Any]:
    """Read verified personal baseline parameters without a current feature row."""
    model, model_sha256, receipt_sha256, _, features = _read_verified_reference(
        model_path=model_path,
        expected_model_sha256=expected_model_sha256,
        receipt_path=receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_training_subject_uuid=expected_training_subject_uuid,
        expected_source_kind=expected_source_kind,
        expected_feature_schema_hash=expected_feature_schema_hash,
    )
    return {
        "schema_version": "verified_exploratory_personal_baseline_v1",
        "role": model["role"],
        "source_kind": model["source_kind"],
        "model_release_uuid": model["model_release_uuid"],
        "model_sha256": model_sha256,
        "receipt_sha256": receipt_sha256,
        "training_subject_uuid": model["training_subject_uuid"],
        "feature_schema_uuid": model["feature_schema_uuid"],
        "feature_schema_hash": model["feature_schema_hash"],
        "features": [
            {
                "code": feature.code,
                "unit": feature.unit,
                "baseline_center": feature.center,
                "baseline_scale": feature.scale,
            }
            for feature in features
        ],
        "semantics": "PERSONAL_BASELINE_PARAMETERS",
        "label_status": "UNREVIEWED",
        "delivery_eligible": False,
        "promotion_eligible": False,
    }


def apply_verified_exploratory_reference(
    *,
    model_path: Path,
    expected_model_sha256: str,
    receipt_path: Path,
    expected_receipt_sha256: str,
    expected_training_subject_uuid: str,
    expected_source_kind: str,
    expected_feature_schema_hash: str,
    feature_values: Mapping[str, object],
) -> dict[str, Any]:
    """Apply one verified model/receipt pair to a current feature row.

    Model and receipt bytes are each read exactly once.  Actor authorization and
    timestamp freshness are preconditions owned by the caller.
    """
    model, model_sha256, receipt_sha256, schema, baselines = _read_verified_reference(
        model_path=model_path,
        expected_model_sha256=expected_model_sha256,
        receipt_path=receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_training_subject_uuid=expected_training_subject_uuid,
        expected_source_kind=expected_source_kind,
        expected_feature_schema_hash=expected_feature_schema_hash,
    )
    if set(feature_values) != {spec.code for spec in schema.features}:
        raise ValueError("FEATURE_VALUES_INVALID")

    features: list[dict[str, object]] = []
    for spec, baseline in zip(schema.features, baselines, strict=True):
        current = _current_value(feature_values[spec.code], spec.domain)
        deviation = (current - baseline.center) / baseline.scale
        if not math.isfinite(deviation):
            raise ValueError("CURRENT_DEVIATION_NONFINITE")
        features.append(
            {
                "code": spec.code,
                "unit": spec.unit,
                "current_value": current,
                "baseline_center": baseline.center,
                "baseline_scale": baseline.scale,
                "deviation_z": deviation,
            }
        )
    return {
        "schema_version": "verified_exploratory_personal_deviation_v1",
        "role": model["role"],
        "source_kind": model["source_kind"],
        "model_release_uuid": model["model_release_uuid"],
        "model_sha256": model_sha256,
        "receipt_sha256": receipt_sha256,
        "training_subject_uuid": model["training_subject_uuid"],
        "feature_schema_uuid": model["feature_schema_uuid"],
        "feature_schema_hash": model["feature_schema_hash"],
        "features": features,
        "semantics": "CURRENT_BASELINE_DEVIATION",
        "label_status": "UNREVIEWED",
        "delivery_eligible": False,
        "promotion_eligible": False,
    }
