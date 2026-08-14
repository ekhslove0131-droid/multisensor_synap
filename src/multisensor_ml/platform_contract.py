"""Canonical Kidsignal model-platform contracts.

This module owns model-side identities, frozen-cohort validation, sensor-view
feature schemas, and reviewed error accounting.  It deliberately excludes
account and membership identities and has no GCP, Android, or serving writes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Final, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from multisensor_ml.h10_runtime_contract import (
    H10_FEATURE_NAMES,
    H10_FEATURE_SCHEMA_VERSION,
    canonical_h10_feature_schema,
)
from multisensor_ml.h10_runtime_contract import (
    summarize_h10_v2_window as _summarize_h10_v2_window,
)
from multisensor_ml.observational_contract import (
    FEATURE_NAMES as WATCH_RUNTIME_FEATURES,
)
from multisensor_ml.observational_contract import (
    FEATURE_SCHEMA_SHA256 as WATCH_RUNTIME_SCHEMA_HASH,
)
from multisensor_ml.observational_contract import (
    FEATURE_SCHEMA_VERSION as WATCH_RUNTIME_SCHEMA_VERSION,
)
from multisensor_ml.observational_contract import canonical_feature_schema

PLATFORM_CONTRACT_VERSION: Final[str] = "kidsignal-model-platform/v1"
FEATURE_CONTRACT_VERSION: Final[str] = "kidsignal-feature-contract/v1"
COHORT_CONTRACT_VERSION: Final[str] = "kidsignal-frozen-training-cohort/v1"
LABEL_REVIEW_CONTRACT_VERSION: Final[str] = "kidsignal-label-review/v1"
SOURCE_SCHEMA_VERSION: Final[str] = "kidsignal-source-schema/v1"
BIGQUERY_CONTRACT_VERSION: Final[str] = "kidsignal-bigquery-training-input/v1"
BASELINE_CONTRACT_VERSION: Final[str] = "personal_robust_baseline_900_eligible_v2"

SENSOR_VARIANTS: Final[tuple[str, ...]] = (
    "watch_only",
    "h10_only",
    "watch_h10",
)
REVIEW_DISPOSITIONS: Final[tuple[str, ...]] = (
    "target_event",
    "valid_non_event",
    "hard_negative",
    "artifact_or_quality_failure",
    "correction_required",
    "novel_pattern",
    "unreviewed",
)
SPLIT_ROLES: Final[tuple[str, ...]] = ("train", "validation", "locked_holdout")
FORBIDDEN_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {"account_uuid", "membership_uuid", "person_key"}
)
FORBIDDEN_MODEL_VIEW_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "account_uuid",
        "membership_uuid",
        "person_uuid",
        "source_uuid",
        "source_uuids",
        "capture_session_uuid",
        "capture_session_uuids",
        "app_instance_uuid",
        "person_key",
        "artifact_uri",
        "generation",
        "checksum",
    }
)
UUID_FIELDS: Final[tuple[str, ...]] = (
    "training_cohort_uuid",
    "feature_set_uuid",
    "feature_schema_uuid",
)
WINDOW_UUID_FIELDS: Final[tuple[str, ...]] = (
    "training_subject_uuid",
    "source_uuid",
    "capture_session_uuid",
)

WATCH_FEATURES: Final[tuple[str, ...]] = (
    "watch_eda_z",
    "watch_hr_z",
    "watch_motion_z",
    "watch_load_raw",
    "watch_eda_mean_30",
    "watch_hr_mean_30",
    "watch_motion_mean_30",
    "watch_eda_slope_60",
    "watch_hr_slope_60",
    "watch_motion_slope_60",
    "watch_load_std_60",
    "watch_load_median_300",
    "watch_load_ema_1800",
    "watch_load_ema_21600",
    "quality_confidence",
    "watch_ineligible_fraction_60",
)
H10_FEATURES: Final[tuple[str, ...]] = H10_FEATURE_NAMES
WATCH_H10_FEATURES: Final[tuple[str, ...]] = (
    *WATCH_FEATURES,
    *H10_FEATURES,
    "fused_load_raw",
    "fused_load_median_300",
    "source_agreement_60",
)

FEATURES_BY_VARIANT: Final[dict[str, tuple[str, ...]]] = {
    "watch_only": WATCH_FEATURES,
    "h10_only": H10_FEATURES,
    "watch_h10": WATCH_H10_FEATURES,
}
REQUIRED_SOURCES: Final[dict[str, tuple[str, ...]]] = {
    "watch_only": ("galaxy_watch8",),
    "h10_only": ("polar_h10",),
    "watch_h10": ("galaxy_watch8", "polar_h10"),
}
REFERENCE_FEATURES: Final[dict[str, dict[str, str]]] = {
    "watch_only": {
        "persistence": "watch_load_raw",
        "rolling_median_300": "watch_load_median_300",
    },
    "h10_only": {
        "persistence": "h10_load_raw",
        "rolling_median_300": "h10_load_median_300",
    },
    "watch_h10": {
        "persistence": "fused_load_raw",
        "rolling_median_300": "fused_load_median_300",
    },
}
SOURCE_SET_TO_VARIANT: Final[dict[tuple[str, ...], str]] = {
    ("watch",): "watch_only",
    ("h10",): "h10_only",
    ("h10", "watch"): "watch_h10",
}
RUNTIME_SCHEMA_STATUS: Final[dict[str, str]] = {
    "watch_only": "CANONICAL_RUNTIME",
    "h10_only": "PROPOSED_NOT_IMPLEMENTED",
    "watch_h10": "PROPOSED_NOT_APPROVED",
}
RUNTIME_SCHEMA_HASHES: Final[dict[str, str]] = {
    "watch_only": WATCH_RUNTIME_SCHEMA_HASH,
    "h10_only": str(canonical_h10_feature_schema()["sha256"]),
    "watch_h10": "10f0be5737cea4965f8ca9552f4f8ae27154b3c195da7ff266e5703ae7d55b66",
}
AUTHORIZED_ENVELOPE_FIELDS: Final[tuple[str, ...]] = (
    "training_cohort_uuid",
    "cohort_digest",
    "split_digest",
    "public_cohort_digest",
    "public_split_digest",
    "feature_schema_uuid",
    "feature_schema_hash",
    "split_policy",
    "purge_seconds",
    "truth_state",
)
AUTHORIZED_ROW_FIELDS: Final[tuple[str, ...]] = (
    "training_subject_uuid",
    "training_capture_set_uuid",
    "exact_window_id",
    "source_set",
    "window_start_ms",
    "window_end_ms",
    "feature_values",
    "label_uuid",
    "label_revision_uuid",
    "review_uuid",
    "review_disposition",
    "evaluation_class",
    "temporal_stage",
    "observation_code",
    "split_role",
    "source_row_digest",
)
AUTHORIZED_VIEW_FIELDS: Final[frozenset[str]] = frozenset(
    (*AUTHORIZED_ENVELOPE_FIELDS, *AUTHORIZED_ROW_FIELDS)
)


def canonical_json(value: object) -> str:
    """Serialize a contract deterministically for SHA-256 identity."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _schema_identity(name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"https://kidsignal.local/schema/{name}"))


def _hash_closed_schema(body: Mapping[str, object]) -> dict[str, object]:
    serialized = canonical_json(body)
    return {
        **body,
        "canonical_json": serialized,
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _require_uuid(value: object, field: str) -> str:
    try:
        parsed = UUID(str(value))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"{field} must be a UUID") from error
    if str(parsed) != str(value).lower():
        raise ValueError(f"{field} must use canonical UUID text")
    return str(parsed)


def _require_sha256(value: object, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


def _find_forbidden_identity(value: object, path: str = "manifest") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if name in FORBIDDEN_IDENTITY_FIELDS:
                return f"{path}.{name}"
            nested = _find_forbidden_identity(child, f"{path}.{name}")
            if nested is not None:
                return nested
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            nested = _find_forbidden_identity(child, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _find_forbidden_model_view_field(value: object, path: str = "handoff") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            if name in FORBIDDEN_MODEL_VIEW_FIELDS:
                return f"{path}.{name}"
            nested = _find_forbidden_model_view_field(child, f"{path}.{name}")
            if nested is not None:
                return nested
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            nested = _find_forbidden_model_view_field(child, f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _feature_definitions(variant: str) -> list[dict[str, object]]:
    if variant == "h10_only":
        runtime_features = canonical_h10_feature_schema()["features"]
        if not isinstance(runtime_features, list):  # pragma: no cover - fixed schema
            raise AssertionError("H10 runtime features must be a list")
        return [
            {**cast(dict[str, object], item), "status": "PROPOSED_NOT_IMPLEMENTED"}
            for item in runtime_features
        ]
    definitions: dict[str, tuple[str, str, str]] = {
        "watch_eda_z": ("robust_z(eda_us,baseline_900)", "EDA", "robust_z"),
        "watch_hr_z": ("robust_z(heart_rate_bpm,baseline_900)", "PPG_HR", "robust_z"),
        "watch_motion_z": ("robust_z(acc_magnitude,baseline_900)", "ACC", "robust_z"),
        "watch_load_raw": (
            "mean(max(watch_eda_z,0),max(watch_hr_z,0),max(watch_motion_z,0))",
            "DERIVED_WATCH",
            "positive_robust_z",
        ),
        "h10_hr_z": ("robust_z(heart_rate_bpm,baseline_900)", "ECG_RR_HR", "robust_z"),
        "h10_rmssd_inverse_z": (
            "-robust_z(rr_rmssd_ms,baseline_900)",
            "RR_INTERVAL",
            "inverse_robust_z",
        ),
        "h10_load_raw": (
            "mean(max(h10_hr_z,0),max(h10_rmssd_inverse_z,0))",
            "DERIVED_H10",
            "positive_robust_z",
        ),
        "fused_load_raw": (
            "mean(watch_load_raw,h10_load_raw)",
            "DERIVED_FUSION",
            "positive_robust_z",
        ),
        "fused_load_median_300": (
            "median(fused_load_raw[t-300:t])",
            "DERIVED_FUSION",
            "positive_robust_z",
        ),
        "source_agreement_60": (
            "1-mean(abs(watch_load_raw-h10_load_raw)[t-60:t])/(1+mean(fused_load_raw[t-60:t]))",
            "DERIVED_FUSION",
            "ratio",
        ),
        "quality_confidence": (
            "min(watch_core_source_quality)",
            "CAPTURE_METADATA",
            "ratio_0_1",
        ),
        "h10_quality_confidence": (
            "min(hr_rr_presence,ecg_contact,clock_trusted,source_quality_sufficient)",
            "CAPTURE_METADATA",
            "ratio_0_1",
        ),
    }
    output: list[dict[str, object]] = []
    for name in FEATURES_BY_VARIANT[variant]:
        if name in definitions:
            formula, source_metric, unit = definitions[name]
        elif name.endswith("_mean_30"):
            base = name.removesuffix("_mean_30")
            formula, source_metric, unit = (
                f"mean({base}[t-30:t])",
                "DERIVED_CAUSAL",
                "robust_z",
            )
        elif name.endswith("_slope_60"):
            base = name.removesuffix("_slope_60")
            formula, source_metric, unit = (
                f"corrected_utc_ols_slope({base}[t-60:t])",
                "DERIVED_CAUSAL",
                "robust_z_per_second",
            )
        elif name.endswith("_std_60"):
            base = name.removesuffix("_std_60")
            formula, source_metric, unit = (
                f"population_std({base}[t-60:t],ddof=0)",
                "DERIVED_CAUSAL",
                "positive_robust_z",
            )
        elif name.endswith("_median_300"):
            base = name.removesuffix("_median_300")
            formula, source_metric, unit = (
                f"median({base}[t-300:t])",
                "DERIVED_CAUSAL",
                "positive_robust_z",
            )
        elif name.endswith("_ema_1800"):
            base = name.removesuffix("_ema_1800")
            formula, source_metric, unit = (
                f"bounded_ema({base},half_life_eligible_seconds=1800)",
                "DERIVED_CAUSAL",
                "positive_robust_z",
            )
        elif name.endswith("_ema_21600"):
            base = name.removesuffix("_ema_21600")
            formula, source_metric, unit = (
                f"bounded_ema({base},half_life_eligible_seconds=21600)",
                "DERIVED_CAUSAL",
                "positive_robust_z",
            )
        elif name.endswith("_ineligible_fraction_60"):
            formula, source_metric, unit = (
                "1-eligible_seconds[t-60:t]/60",
                "CAPTURE_METADATA",
                "ratio_0_1",
            )
        else:  # pragma: no cover - every canonical feature is classified above
            raise AssertionError(f"feature definition missing: {name}")
        output.append(
            {
                "name": name,
                "formula": formula,
                "source_metric": source_metric,
                "unit": unit,
                "causal": True,
                "status": (
                    "PROPOSED_NOT_VERIFIED"
                    if variant == "watch_h10"
                    and name in {
                        "fused_load_raw",
                        "fused_load_median_300",
                        "source_agreement_60",
                    }
                    else "CANONICAL_CONTRACT"
                ),
            }
        )
    return output


def feature_schema_for_variant(variant: str) -> dict[str, object]:
    """Return one canonical, hash-closed feature schema."""

    if variant not in SENSOR_VARIANTS:
        raise ValueError(f"unsupported sensor_variant: {variant}")
    features = FEATURES_BY_VARIANT[variant]
    body: dict[str, object] = {
        "contract_version": FEATURE_CONTRACT_VERSION,
        "schema_uuid": _schema_identity(f"feature/{FEATURE_CONTRACT_VERSION}/{variant}"),
        "sensor_variant": variant,
        "required_sources": list(REQUIRED_SOURCES[variant]),
        "input_tensor": {"name": "features", "dtype": "float32", "shape": [None, len(features)]},
        "ordered_feature_names": list(features),
        "features": _feature_definitions(variant),
        "corrected_time": "UTC_1HZ",
        "baseline": "personal_robust_baseline_900_eligible_v2",
        "decision_warmup_eligible_seconds": 1800,
        "max_gap_seconds": 5,
        "quality_policy": "CONSERVATIVE_AND_FAIL_CLOSED",
        "missing_source_policy": "NOT_DECISIONABLE",
        "biological_null_policy": "FORBID_ZERO_OR_NULL_IMPUTATION",
        "reference_features": REFERENCE_FEATURES[variant],
        "real_data_status": "NOT VERIFIED",
        "serving_contract_status": (
            "PROPOSED_NOT_APPROVED"
            if variant == "watch_h10"
            else "PROPOSED_NOT_IMPLEMENTED"
            if variant == "h10_only"
            else "CANONICAL_CONTRACT"
        ),
    }
    if variant == "h10_only":
        h10_runtime = canonical_h10_feature_schema()
        body["runtime_contract"] = {
            "schema_uuid": h10_runtime["schema_uuid"],
            "schema_version": H10_FEATURE_SCHEMA_VERSION,
            "schema_hash": h10_runtime["sha256"],
        }
    return _hash_closed_schema(body)


def h10_runtime_feature_schema() -> dict[str, object]:
    """Expose the executable H10 v2 proposal through the platform API."""

    return canonical_h10_feature_schema()


def summarize_h10_v2_window(payload: Mapping[str, object]) -> dict[str, object]:
    """Expose the executable H10 source-window quality summary."""

    return _summarize_h10_v2_window(payload)


def runtime_feature_schema_for_variant(variant: str) -> dict[str, object]:
    """Return the runtime/training identity, not the platform description hash."""

    if variant not in SENSOR_VARIANTS:
        raise ValueError(f"unsupported sensor_variant: {variant}")
    description = feature_schema_for_variant(variant)
    feature_names = (
        WATCH_RUNTIME_FEATURES if variant == "watch_only" else FEATURES_BY_VARIANT[variant]
    )
    if variant == "watch_only" and tuple(feature_names) != WATCH_FEATURES:
        raise ValueError("Watch runtime features differ from platform feature order")
    status = RUNTIME_SCHEMA_STATUS[variant]
    h10_schema = canonical_h10_feature_schema() if variant == "h10_only" else None
    schema_uuid = (
        h10_schema["schema_uuid"] if h10_schema is not None else description["schema_uuid"]
    )
    schema_version = (
        WATCH_RUNTIME_SCHEMA_VERSION
        if variant == "watch_only"
        else H10_FEATURE_SCHEMA_VERSION
        if variant == "h10_only"
        else "PROPOSED_NO_RUNTIME_SCHEMA_VERSION"
    )
    return {
        "schema_uuid": schema_uuid,
        "schema_version": schema_version,
        "schema_hash": RUNTIME_SCHEMA_HASHES[variant],
        "ordered_feature_names": list(feature_names),
        "training_status": status,
        "training_ready": status == "CANONICAL_RUNTIME",
        "runtime_schema": (
            canonical_feature_schema()
            if variant == "watch_only"
            else h10_schema
            if variant == "h10_only"
            else None
        ),
        "platform_description_sha256": description["sha256"],
    }


def source_schema(source_name: str) -> dict[str, object]:
    """Return one vendor-source contract without backend-derived features."""

    shared_fields: list[dict[str, object]] = [
        {"name": "corrected_utc", "type": "timestamp_utc", "required": True},
        {"name": "source_uuid", "type": "uuid", "required": True},
        {"name": "capture_session_uuid", "type": "uuid", "required": True},
        {"name": "clock_trusted", "type": "boolean", "required": True},
        {"name": "presence_active", "type": "boolean", "required": True},
        {"name": "source_quality_sufficient", "type": "boolean", "required": True},
        {"name": "capture_sequence", "type": "int64", "required": True},
    ]
    source_fields: dict[str, list[dict[str, object]]] = {
        "galaxy_watch8": [
            {"name": "heart_rate_bpm", "type": "float32", "unit": "bpm", "required": True},
            {"name": "eda_us", "type": "float32", "unit": "microsiemens", "required": True},
            {"name": "acc_x_mps2", "type": "float32", "unit": "m/s2", "required": True},
            {"name": "acc_y_mps2", "type": "float32", "unit": "m/s2", "required": True},
            {"name": "acc_z_mps2", "type": "float32", "unit": "m/s2", "required": True},
            {
                "name": "vendor_native_observation",
                "type": "json",
                "required": False,
            },
        ],
        "polar_h10": [
            {
                "name": "payload_schema_version",
                "type": "literal<kidsignal_polar_h10_ecg_v2>",
                "required": True,
            },
            {"name": "device_profile", "type": "literal<polar_h10_v1>", "required": True},
            {"name": "window_start_corrected_utc_ms", "type": "int64", "required": True},
            {"name": "sample_rate_hz", "type": "int32", "required": True},
            {"name": "expected_sample_count", "type": "int32", "required": True},
            {"name": "observed_sample_count", "type": "int32", "required": True},
            {"name": "gap_count", "type": "int32", "required": True},
            {"name": "timestamp_regression_count", "type": "int32", "required": True},
            {"name": "is_sufficient", "type": "boolean", "required": True},
            {
                "name": "hr_stream_state",
                "type": "enum<not_observed|observed|failed>",
                "required": True,
            },
            {
                "name": "ecg_samples",
                "type": "list<struct<corrected_utc_ms:int64,voltage_uv:float32>>",
                "sample_rate_hz": 130,
                "required": True,
            },
            {
                "name": "hr_observations",
                "type": (
                    "list<struct<phone_utc_ms:int64,hr_bpm:int32,corrected_hr_bpm:int32,"
                    "rr_ms:list<float32>,rr_available:boolean,contact_status:boolean,"
                    "contact_status_supported:boolean>>"
                ),
                "required": True,
            },
            {"name": "battery_observations", "type": "list<json>", "required": False},
        ],
    }
    if source_name not in source_fields:
        raise ValueError(f"unsupported source_name: {source_name}")
    body: dict[str, object] = {
        "contract_version": SOURCE_SCHEMA_VERSION,
        "schema_uuid": _schema_identity(f"source/{SOURCE_SCHEMA_VERSION}/{source_name}"),
        "source_name": source_name,
        "fields": [*shared_fields, *source_fields[source_name]],
        "exact_composition": [
            "source_manifest_scheduling_intersection",
            "corrected_time_provenance",
        ],
        "cross_device_waveform_alignment": "FORBIDDEN_UNTIL_SEPARATE_CONTRACT",
        "offset_lag_estimation": "FORBIDDEN_UNTIL_SEPARATE_CONTRACT",
        "interpolation_or_resampling": "FORBIDDEN",
        "derived_feature_owner": "CLOUD_RUN",
        "missing_source_policy": "NOT_DECISIONABLE",
        "imputation_policy": "NO_ZERO_OR_NULL_BIOLOGICAL_IMPUTATION",
    }
    return _hash_closed_schema(body)


def label_review_schema() -> dict[str, object]:
    """Return temporal label and independent-review truth accounting."""

    body: dict[str, object] = {
        "contract_version": LABEL_REVIEW_CONTRACT_VERSION,
        "schema_uuid": _schema_identity(f"label-review/{LABEL_REVIEW_CONTRACT_VERSION}"),
        "stage_semantics": "TEMPORAL_NOT_SEVERITY",
        "stages": {
            "1": ["pre_early"],
            "2": ["pre_late", "onset"],
            "3": ["peak"],
            "4": ["recovery_early"],
            "5": ["recovery_late", "post"],
        },
        "summary": {"before": [1, 2], "during": [3], "after": [4, 5]},
        "review_dispositions": list(REVIEW_DISPOSITIONS),
        "fpr_denominator": ["valid_non_event", "hard_negative"],
        "fnr_denominator": ["target_event"],
        "excluded_from_error_denominators": [
            "artifact_or_quality_failure",
            "correction_required",
            "novel_pattern",
            "unreviewed",
        ],
        "required_lineage": [
            "label_uuid",
            "label_revision_uuid",
            "review_uuid",
            "exact_window_id",
            "training_subject_uuid",
        ],
    }
    return _hash_closed_schema(body)


def baseline_schema() -> dict[str, object]:
    """Return the fail-closed personal baseline and decision-readiness contract."""

    body: dict[str, object] = {
        "contract_version": BASELINE_CONTRACT_VERSION,
        "schema_uuid": _schema_identity(f"baseline/{BASELINE_CONTRACT_VERSION}"),
        "time_axis": "corrected_utc_1hz",
        "minimum_eligible_seconds": 900,
        "maximum_wall_seconds": 1000,
        "decision_warmup_eligible_seconds": 1800,
        "maximum_gap_seconds": 5,
        "center": "median(past_eligible_values)",
        "scale": "1.4826*MAD(past_eligible_values)",
        "scale_fallback": ["MAD", "IQR/1.349", "BASELINE_SCALE_ZERO"],
        "quality_policy": "CONSERVATIVE_AND_FAIL_CLOSED",
        "quality_provenance": (
            "clock_trusted AND presence_active AND source_quality_sufficient "
            "for all contributing fragments"
        ),
        "window_semantics": "[t-window,t)",
        "missing_policy": "NO_INTERPOLATION_NO_HOLD_FORWARD",
    }
    return _hash_closed_schema(body)


def bigquery_training_contract() -> dict[str, object]:
    """Describe the read-only, frozen BigQuery input boundary."""

    return {
        "contract_version": BIGQUERY_CONTRACT_VERSION,
        "lookup_parameters": ["training_cohort_uuid", "cohort_digest"],
        "required_tables": [
            "committed_exact_windows",
            "feature_sets",
            "label_revisions",
            "independent_reviews",
            "training_cohort_manifests",
            "split_assignments",
        ],
        "required_views": ["authorized_or_materialized_training_view"],
        "selection_roles": ["train", "validation"],
        "view_required_columns": [
            *AUTHORIZED_ENVELOPE_FIELDS,
            *AUTHORIZED_ROW_FIELDS,
        ],
        "sequence_group_key": "training_capture_set_uuid",
        "source_set_mapping": {
            "watch": "watch_only",
            "h10": "h10_only",
            "h10+watch": "watch_h10",
        },
        "training_ready_variants": ["watch_only"],
        "non_training_ready_variants": {
            "h10_only": "PROPOSED_NOT_IMPLEMENTED",
            "watch_h10": "PROPOSED_NOT_APPROVED",
        },
        "forbidden_view_policy": "NO_DIRECT_IDENTITY_OR_GCS_CREDENTIAL_LINEAGE",
        "feature_values_policy": "EXACT_FLAT_FINITE_NUMERIC_SCHEMA_KEYS",
        "handoff_digest_policy": "RECOMPUTE_AND_REJECT_MISMATCH",
        "negative_stage_policy": "IGNORE_TEMPORAL_STAGE_USE_NO_EVENT",
        "reviewed_event_anchor_status": "BLOCKED_PENDING_CONTRACT",
        "blocked_metrics": ["event_delay", "forecast_lead_time"],
        "split_policy": {
            "selection_roles": ["train", "validation"],
            "locked_holdout": "final_champion_comparison_only",
            "minimum_purge_seconds": 1800,
            "person_group_or_chronological": True,
            "preregistered_strategies": ["person_group", "chronological_per_subject"],
        },
        "selection_query_contract": (
            "SELECT * FROM authorized_or_materialized_training_view "
            "WHERE training_cohort_uuid=@training_cohort_uuid "
            "AND cohort_digest=@cohort_digest "
            "AND split_role IN UNNEST(@selection_roles)"
        ),
        "locked_holdout_query_contract": (
            "separate final-comparison read requiring immutable evaluation_uuid "
            "after candidate and adapter freeze"
        ),
        "locked_holdout": {
            "notebook_access": "FORBIDDEN",
            "row_and_label_access": "SEALED_ONLY",
            "evaluator": "SEALED_EVALUATOR",
            "accepted_input": "final_immutable_bundle",
            "output": "aggregate_comparison_only",
            "row_level_error_return": "FORBIDDEN_UNTIL_RETIREMENT",
        },
        "gcs_is_canonical": True,
        "bigquery_is_projection": True,
        "live_auth_status": "NOT VERIFIED",
        "live_cohort_status": "NOT VERIFIED",
    }


def _authorized_row_mapping(row: object) -> dict[str, object]:
    if isinstance(row, Mapping):
        return {str(key): value for key, value in row.items()}
    items = getattr(row, "items", None)
    if not callable(items):
        raise ValueError("authorized row must be a mapping or Row-like object")
    try:
        return {str(key): value for key, value in items()}
    except (TypeError, ValueError) as error:
        raise ValueError("authorized row must be a mapping or Row-like object") from error


def _recompute_public_training_digests(
    rows: Sequence[Mapping[str, object]],
    envelope: Mapping[str, object],
) -> tuple[str, str]:
    ordered_members = sorted(
        (
            [
                row["training_subject_uuid"],
                row["training_capture_set_uuid"],
                row["exact_window_id"],
                row["split_role"],
                row["window_start_ms"],
                row["source_row_digest"],
            ]
            for row in rows
        ),
        key=lambda member: (
            str(member[0]),
            int(cast(int, member[4])),
            str(member[2]),
        ),
    )
    public_split_material = [
        [member[0], member[2], member[3]] for member in ordered_members
    ]
    public_cohort_material = [
        envelope["feature_schema_uuid"],
        envelope["feature_schema_hash"],
        envelope["split_policy"],
        envelope["purge_seconds"],
        envelope["truth_state"],
        ordered_members,
    ]
    return (
        sha256_json(public_cohort_material),
        sha256_json(public_split_material),
    )


def _validate_public_training_digests(
    rows: Sequence[Mapping[str, object]],
    envelope: Mapping[str, object],
) -> None:
    supplied_cohort = _require_sha256(
        envelope["public_cohort_digest"], "public_cohort_digest"
    )
    supplied_split = _require_sha256(
        envelope["public_split_digest"], "public_split_digest"
    )
    recomputed_cohort, recomputed_split = _recompute_public_training_digests(
        rows, envelope
    )
    if supplied_cohort != recomputed_cohort:
        raise ValueError("public_cohort_digest does not match authorized rows")
    if supplied_split != recomputed_split:
        raise ValueError("public_split_digest does not match authorized rows")


def synchronize_authorized_training_rows(rows: Iterable[object]) -> dict[str, object]:
    """Build one hash-closed model handoff from flat authorized BigQuery rows."""

    values = tuple(_authorized_row_mapping(row) for row in rows)
    if not values:
        raise ValueError("authorized handoff rows must not be empty")
    if any(set(row) != AUTHORIZED_VIEW_FIELDS for row in values):
        raise ValueError("unknown or private authorized field")
    envelope = {field: values[0][field] for field in AUTHORIZED_ENVELOPE_FIELDS}
    if any(
        any(row[field] != envelope[field] for field in AUTHORIZED_ENVELOPE_FIELDS)
        for row in values[1:]
    ):
        raise ValueError("mixed authorized handoff envelope")

    projected_rows = [
        {
            **{field: row[field] for field in AUTHORIZED_ROW_FIELDS},
            "feature_schema_uuid": envelope["feature_schema_uuid"],
            "feature_schema_hash": envelope["feature_schema_hash"],
        }
        for row in values
    ]
    projected_rows.sort(
        key=lambda row: (
            str(row["split_role"]),
            str(row["training_subject_uuid"]),
            int(cast(int, row["window_start_ms"])),
            str(row["exact_window_id"]),
        )
    )
    roles = {str(row["split_role"]) for row in projected_rows}
    if not roles.issubset({"TRAIN", "VALIDATION"}):
        raise ValueError("LOCKED or unknown split_role is forbidden")
    if roles != {"TRAIN", "VALIDATION"}:
        raise ValueError("train and validation support are required")
    exact_windows = [str(row["exact_window_id"]) for row in projected_rows]
    if len(exact_windows) != len(set(exact_windows)):
        raise ValueError("duplicate exact_window_id")
    source_digests = [str(row["source_row_digest"]) for row in projected_rows]
    if len(source_digests) != len(set(source_digests)):
        raise ValueError("duplicate source_row_digest")
    _validate_public_training_digests(values, envelope)

    body: dict[str, object] = {
        "contract_version": 1,
        **envelope,
        "sequence_group_key": "training_capture_set_uuid",
        "locked_access": False,
        "rows": projected_rows,
    }
    handoff = {**body, "handoff_digest": sha256_json(body)}
    validate_model_training_handoff(handoff)
    return handoff


def validate_model_training_handoff(handoff: Mapping[str, object]) -> dict[str, object]:
    """Validate the minimum-privilege Cloud handoff exposed to model notebooks."""

    forbidden = _find_forbidden_model_view_field(handoff)
    if forbidden is not None:
        raise ValueError(f"forbidden model-view field: {forbidden}")
    required = {
        "contract_version",
        "training_cohort_uuid",
        "cohort_digest",
        "split_digest",
        "public_cohort_digest",
        "public_split_digest",
        "feature_schema_uuid",
        "feature_schema_hash",
        "split_policy",
        "purge_seconds",
        "sequence_group_key",
        "locked_access",
        "truth_state",
        "rows",
        "handoff_digest",
    }
    missing = sorted(required.difference(handoff))
    if missing:
        raise ValueError(f"model training handoff missing {missing[0]}")
    extras = sorted(set(handoff).difference(required))
    if extras:
        raise ValueError(f"model training handoff has unknown field {extras[0]}")
    if handoff["contract_version"] != 1:
        raise ValueError("contract_version must be 1")
    handoff_digest = _require_sha256(handoff["handoff_digest"], "handoff_digest")
    digest_body = {
        str(key): value for key, value in handoff.items() if key != "handoff_digest"
    }
    if sha256_json(digest_body) != handoff_digest:
        raise ValueError("handoff_digest does not match public handoff body")
    for field in ("training_cohort_uuid", "feature_schema_uuid"):
        _require_uuid(handoff[field], field)
    for field in (
        "cohort_digest",
        "split_digest",
        "public_cohort_digest",
        "public_split_digest",
        "feature_schema_hash",
    ):
        _require_sha256(handoff[field], field)
    if handoff["sequence_group_key"] != "training_capture_set_uuid":
        raise ValueError("sequence_group_key must be training_capture_set_uuid")
    if handoff["locked_access"] is not False:
        raise ValueError("LOCKED access is forbidden in model handoff")
    split_policy = str(handoff["split_policy"])
    if split_policy not in {"PERSON_GROUP", "CHRONOLOGICAL_PER_SUBJECT"}:
        raise ValueError("unsupported split_policy")
    purge_seconds = handoff["purge_seconds"]
    if type(purge_seconds) is not int or purge_seconds < 1800:
        raise ValueError("purge_seconds must be an integer of at least 1800")
    truth_state = str(handoff["truth_state"])
    if truth_state not in {"REVIEWED_REAL", "NOT_EVALUABLE"}:
        raise ValueError("truth_state is invalid")

    schema_matches = [
        variant
        for variant in SENSOR_VARIANTS
        if (
            runtime_feature_schema_for_variant(variant)["schema_uuid"]
            == handoff["feature_schema_uuid"]
            and runtime_feature_schema_for_variant(variant)["schema_hash"]
            == handoff["feature_schema_hash"]
        )
    ]
    if len(schema_matches) != 1:
        raise ValueError("handoff feature schema is not canonical runtime identity")
    variant = schema_matches[0]
    schema = runtime_feature_schema_for_variant(variant)
    if schema["training_ready"] is not True:
        raise ValueError(
            f"feature schema is not training ready: {schema['training_status']}"
        )
    expected_features = tuple(cast(list[str], schema["ordered_feature_names"]))
    rows = handoff["rows"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("rows must be a non-empty sequence")

    row_required = {
        "training_subject_uuid",
        "training_capture_set_uuid",
        "exact_window_id",
        "source_set",
        "window_start_ms",
        "window_end_ms",
        "feature_schema_uuid",
        "feature_schema_hash",
        "feature_values",
        "label_uuid",
        "label_revision_uuid",
        "review_uuid",
        "review_disposition",
        "evaluation_class",
        "temporal_stage",
        "observation_code",
        "split_role",
        "source_row_digest",
    }
    split_counts = {"TRAIN": 0, "VALIDATION": 0}
    exact_windows: set[str] = set()
    source_digests: set[str] = set()
    subjects_by_split: dict[str, set[str]] = {"TRAIN": set(), "VALIDATION": set()}
    subjects_by_capture: dict[str, set[str]] = {}
    bounds_by_subject_split: dict[tuple[str, str], list[tuple[int, int]]] = {}
    row_order_keys: list[tuple[str, str, int, str]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"rows[{index}] must be an object")
        forbidden = _find_forbidden_model_view_field(raw, f"rows[{index}]")
        if forbidden is not None:
            raise ValueError(f"forbidden model-view field: {forbidden}")
        missing = sorted(row_required.difference(raw))
        if missing:
            raise ValueError(f"rows[{index}] missing {missing[0]}")
        extras = sorted(set(raw).difference(row_required))
        if extras:
            raise ValueError(f"rows[{index}] has unknown field {extras[0]}")
        subject = _require_uuid(raw["training_subject_uuid"], "training_subject_uuid")
        capture_set = _require_uuid(
            raw["training_capture_set_uuid"], "training_capture_set_uuid"
        )
        for field in ("label_uuid", "label_revision_uuid", "review_uuid"):
            _require_uuid(raw[field], field)
        exact_window = _require_sha256(raw["exact_window_id"], "exact_window_id")
        source_digest = _require_sha256(raw["source_row_digest"], "source_row_digest")
        if exact_window in exact_windows:
            raise ValueError("duplicate exact_window_id")
        if source_digest in source_digests:
            raise ValueError("duplicate source_row_digest")
        exact_windows.add(exact_window)
        source_digests.add(source_digest)
        source_set = raw["source_set"]
        if not isinstance(source_set, Sequence) or isinstance(source_set, (str, bytes)):
            raise ValueError("source_set must be an array")
        source_key = tuple(sorted(str(item) for item in source_set))
        if SOURCE_SET_TO_VARIANT.get(source_key) != variant:
            raise ValueError("source_set does not match feature schema variant")
        if (raw["feature_schema_uuid"], raw["feature_schema_hash"]) != (
            handoff["feature_schema_uuid"],
            handoff["feature_schema_hash"],
        ):
            raise ValueError("row feature schema does not match handoff")
        feature_values = raw["feature_values"]
        if not isinstance(feature_values, Mapping) or set(feature_values) != set(
            expected_features
        ):
            raise ValueError("feature_values must use exact canonical schema keys")
        if any(
            type(feature_values[name]) not in {int, float}
            or not math.isfinite(float(cast(float, feature_values[name])))
            for name in expected_features
        ):
            raise ValueError("feature_values must be flat finite numeric values")
        split = str(raw["split_role"])
        if split not in split_counts:
            raise ValueError("LOCKED or unknown split_role is forbidden")
        evaluation_class = str(raw["evaluation_class"])
        if evaluation_class not in {"POSITIVE", "NEGATIVE"}:
            raise ValueError("only reviewed POSITIVE or NEGATIVE rows are allowed")
        review_disposition = str(raw["review_disposition"])
        expected_evaluation = {
            "TARGET_EVENT": "POSITIVE",
            "VALID_NON_EVENT": "NEGATIVE",
            "HARD_NEGATIVE": "NEGATIVE",
        }.get(review_disposition)
        if expected_evaluation is None or evaluation_class != expected_evaluation:
            raise ValueError("review_disposition does not match evaluation_class")
        if evaluation_class == "NEGATIVE" and raw["observation_code"] != "NO_EVENT":
            raise ValueError("negative observation_code must be NO_EVENT")
        stage = raw["temporal_stage"]
        if type(stage) is not int or not 1 <= stage <= 5:
            raise ValueError("temporal_stage must be an integer from 1 to 5")
        start = raw["window_start_ms"]
        end = raw["window_end_ms"]
        if type(start) is not int or type(end) is not int or start < 0 or start >= end:
            raise ValueError("window millisecond bounds are invalid")
        split_counts[split] += 1
        row_order_keys.append((split, subject, start, exact_window))
        subjects_by_split[split].add(subject)
        subjects_by_capture.setdefault(capture_set, set()).add(subject)
        bounds_by_subject_split.setdefault((subject, split), []).append((start, end))

    if any(len(subjects) != 1 for subjects in subjects_by_capture.values()):
        raise ValueError("training capture set spans subjects")
    if row_order_keys != sorted(row_order_keys):
        raise ValueError("handoff rows are not in canonical deterministic order")
    if split_counts["TRAIN"] < 1 or split_counts["VALIDATION"] < 1:
        raise ValueError("train and validation support are required")
    if split_policy == "PERSON_GROUP":
        if subjects_by_split["TRAIN"].intersection(subjects_by_split["VALIDATION"]):
            raise ValueError("person-group split leakage")
    else:
        purge_ms = purge_seconds * 1000
        for subject in subjects_by_split["TRAIN"].intersection(
            subjects_by_split["VALIDATION"]
        ):
            train_end = max(
                end for _, end in bounds_by_subject_split[(subject, "TRAIN")]
            )
            validation_start = min(
                start for start, _ in bounds_by_subject_split[(subject, "VALIDATION")]
            )
            if train_end + purge_ms > validation_start:
                raise ValueError("chronological target purge violated")
    _validate_public_training_digests(
        cast(Sequence[Mapping[str, object]], rows), handoff
    )
    return {
        "status": (
            "VALID_SELECTION_HANDOFF"
            if truth_state == "REVIEWED_REAL"
            else "VALID_NOT_EVALUABLE_HANDOFF"
        ),
        "training_eligible": truth_state == "REVIEWED_REAL",
        "sensor_variant": variant,
        "split_counts": split_counts,
        "sequence_group_count": len(subjects_by_capture),
        "sequence_group_key": "training_capture_set_uuid",
        "negative_stage_policy": "IGNORE_TEMPORAL_STAGE_USE_NO_EVENT",
        "locked_access": False,
    }


def canonical_platform_contract() -> dict[str, object]:
    return {
        "contract_version": PLATFORM_CONTRACT_VERSION,
        "canonical_model_repository": "multisensor_ml",
        "synthetic_truth_repository": "multisensor_synth",
        "identity_scope": {
            "allowed": [
                "training_cohort_uuid",
                "training_subject_uuid",
                "source_uuid",
                "capture_session_uuid",
                "exact_window_id",
                "feature_set_uuid",
                "feature_schema_uuid",
                "label_uuid",
                "label_revision_uuid",
                "review_uuid",
                "evaluation_uuid",
                "comparison_uuid",
                "model_release_uuid",
                "adapter_uuid",
                "promotion_uuid",
            ],
            "provisional_alias": "install_scoped_legacy_alias_not_for_model_identity",
        },
        "sensor_variants": list(SENSOR_VARIANTS),
        "exact_composition": {
            "implemented": [
                "source_manifest_scheduling_intersection",
                "corrected_time_provenance",
            ],
            "forbidden_until_separate_contract": [
                "cross_device_waveform_alignment",
                "offset_or_physiological_lag_estimation",
                "interpolation",
                "resampling",
            ],
        },
        "review_dispositions": list(REVIEW_DISPOSITIONS),
        "stage_contract": {
            "ordered_truth": {
                "1": ["pre_early"],
                "2": ["pre_late", "onset"],
                "3": ["peak"],
                "4": ["recovery_early"],
                "5": ["recovery_late", "post"],
            },
            "summary": {"before": [1, 2], "during": [3], "after": [4, 5]},
            "stage_is_severity": False,
        },
        "storage_boundaries": {
            "gcs": "canonical_committed_data_lake",
            "duckdb_memory": "exact_window_ephemeral_observer",
            "bigquery": "long_term_projection_monitoring_and_frozen_training_cohort",
            "neon": "migration_only_legacy_not_runtime_source",
        },
        "bigquery_input": {
            "lookup_keys": ["training_cohort_uuid", "cohort_digest"],
            "view_kind": "materialized_or_authorized_training_view",
            "live_auth_status": "NOT VERIFIED",
        },
        "evaluation": {
            "false_positive_denominator": ["valid_non_event", "hard_negative"],
            "false_negative_denominator": ["target_event"],
            "excluded_from_fp": [
                "artifact_or_quality_failure",
                "correction_required",
                "novel_pattern",
                "unreviewed",
            ],
            "separate_rates": [
                "unreviewed_rate",
                "quality_exclusion_rate",
                "correction_backlog",
            ],
            "real_accuracy_status_before_independent_locked_holdout": "NOT VERIFIED",
        },
        "locked_holdout": {
            "notebook_row_or_label_access": False,
            "evaluator": "SEALED_EVALUATOR",
            "input": "final_immutable_bundle_only",
            "row_level_error_return": "FORBIDDEN_UNTIL_HOLDOUT_RETIREMENT",
            "post_retirement_use": "new_training_cohort_candidate_only",
        },
    }


def compute_exact_window_id(content: bytes) -> str:
    """Return the exact corrected-time window's content identity."""

    if not isinstance(content, bytes) or not content:
        raise ValueError("exact-window content must be non-empty bytes")
    return hashlib.sha256(content).hexdigest()


def _cohort_digest_payload(manifest: Mapping[str, object]) -> dict[str, object]:
    return {str(key): value for key, value in manifest.items() if key != "cohort_digest"}


def compute_cohort_digest(manifest: Mapping[str, object]) -> str:
    return sha256_json(_cohort_digest_payload(manifest))


def validate_frozen_cohort_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    """Fail closed on identity leakage, mutable lineage, or holdout selection."""

    forbidden = _find_forbidden_identity(manifest)
    if forbidden is not None:
        raise ValueError(f"forbidden identity field: {forbidden}")
    required = {
        "contract_version",
        "training_cohort_uuid",
        "cohort_digest",
        "training_view",
        "materialization_uri",
        "feature_set_uuid",
        "feature_schema_uuid",
        "feature_schema_hash",
        "sensor_variant",
        "split_strategy",
        "purge_seconds",
        "locked_holdout_read",
        "windows",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"frozen cohort manifest missing fields: {missing}")
    if manifest["contract_version"] != PLATFORM_CONTRACT_VERSION:
        raise ValueError("contract_version does not match platform contract")
    for field in UUID_FIELDS:
        _require_uuid(manifest[field], field)
    _require_sha256(manifest["feature_schema_hash"], "feature_schema_hash")
    variant = str(manifest["sensor_variant"])
    if variant not in SENSOR_VARIANTS:
        raise ValueError("sensor_variant is not canonical")
    runtime_schema = runtime_feature_schema_for_variant(variant)
    if runtime_schema["training_ready"] is not True:
        raise ValueError(
            f"feature schema is not training ready: {runtime_schema['training_status']}"
        )
    if manifest["feature_schema_uuid"] != runtime_schema["schema_uuid"]:
        raise ValueError("feature_schema_uuid does not match sensor_variant")
    if manifest["feature_schema_hash"] != runtime_schema["schema_hash"]:
        raise ValueError("feature_schema_hash does not match sensor_variant")
    if int(cast(int, manifest["purge_seconds"])) < 1800:
        raise ValueError("purge_seconds must be at least 1800")
    split_strategy = str(manifest["split_strategy"])
    if split_strategy not in {"person_group", "chronological_per_subject"}:
        raise ValueError("split_strategy must be person_group or chronological_per_subject")
    if manifest["locked_holdout_read"] is not False:
        raise ValueError("locked_holdout_read must remain false during selection")
    windows = manifest["windows"]
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)) or not windows:
        raise ValueError("windows must be a non-empty sequence")
    identities: set[str] = set()
    subjects_by_role: dict[str, set[str]] = {role: set() for role in SPLIT_ROLES}
    bounds_by_subject_role: dict[tuple[str, str], list[tuple[datetime, datetime]]] = {}
    counts = {role: 0 for role in SPLIT_ROLES}
    for index, raw in enumerate(windows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"windows[{index}] must be an object")
        for field in WINDOW_UUID_FIELDS:
            _require_uuid(raw.get(field), f"windows[{index}].{field}")
        identity = _require_sha256(raw.get("exact_window_id"), f"windows[{index}].exact_window_id")
        if identity in identities:
            raise ValueError("exact_window_id values must be unique")
        identities.add(identity)
        role = str(raw.get("split_role"))
        if role not in SPLIT_ROLES:
            raise ValueError(f"windows[{index}].split_role is invalid")
        counts[role] += 1
        subjects_by_role[role].add(str(raw["training_subject_uuid"]))
        start = str(raw.get("window_start_utc", ""))
        end = str(raw.get("window_end_utc", ""))
        if not start.endswith("Z") or not end.endswith("Z") or start >= end:
            raise ValueError(f"windows[{index}] has invalid UTC bounds")
        start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))
        bounds_by_subject_role.setdefault((str(raw["training_subject_uuid"]), role), []).append(
            (start_time, end_time)
        )
    if split_strategy == "person_group":
        for left_index, left in enumerate(SPLIT_ROLES):
            for right in SPLIT_ROLES[left_index + 1 :]:
                if subjects_by_role[left].intersection(subjects_by_role[right]):
                    raise ValueError(f"training subjects overlap across {left} and {right}")
    else:
        purge = timedelta(seconds=int(cast(int, manifest["purge_seconds"])))
        all_subjects = set().union(*subjects_by_role.values())
        for subject in all_subjects:
            for left, right in pairwise(SPLIT_ROLES):
                left_bounds = bounds_by_subject_role.get((subject, left), [])
                right_bounds = bounds_by_subject_role.get((subject, right), [])
                if left_bounds and right_bounds:
                    left_end = max(end_time for _, end_time in left_bounds)
                    right_start = min(start_time for start_time, _ in right_bounds)
                    if left_end + purge > right_start:
                        raise ValueError(f"chronological split purge violated: {left} to {right}")
    if compute_cohort_digest(manifest) != manifest["cohort_digest"]:
        raise ValueError("cohort_digest does not match manifest content")
    return {
        "status": "VALID",
        "training_cohort_uuid": str(manifest["training_cohort_uuid"]),
        "cohort_digest": str(manifest["cohort_digest"]),
        "sensor_variant": variant,
        "split_strategy": split_strategy,
        "split_window_counts": counts,
        "locked_holdout_read": False,
    }


def evaluate_error_case(
    *,
    disposition: str,
    truth_positive: bool,
    prediction: float,
    threshold: float,
    label_revision_uuid: str | None,
    review_uuid: str | None,
) -> dict[str, object]:
    """Classify one reviewed case without turning unknowns into false positives."""

    if disposition not in REVIEW_DISPOSITIONS:
        raise ValueError("unknown review disposition")
    if not math.isfinite(prediction) or not 0.0 <= prediction <= 1.0:
        raise ValueError("prediction must be finite and in [0,1]")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    evaluated_negative = disposition in {"valid_non_event", "hard_negative"}
    evaluated_positive = disposition == "target_event"
    if evaluated_negative and truth_positive:
        raise ValueError("reviewed negative cannot have positive truth")
    if evaluated_positive and not truth_positive:
        raise ValueError("target_event requires positive truth")
    if evaluated_positive or evaluated_negative:
        if label_revision_uuid is None or review_uuid is None:
            raise ValueError("evaluable cases require label_revision_uuid and review_uuid")
        _require_uuid(label_revision_uuid, "label_revision_uuid")
        _require_uuid(review_uuid, "review_uuid")
    predicted_positive = prediction >= threshold
    error_type: str | None = None
    if evaluated_negative and predicted_positive:
        error_type = "FP"
    elif evaluated_positive and not predicted_positive:
        error_type = "FN"
    return {
        "evaluable": evaluated_positive or evaluated_negative,
        "error_type": error_type,
        "predicted_positive": predicted_positive,
        "count_in_fpr_denominator": evaluated_negative,
        "count_in_fnr_denominator": evaluated_positive,
        "disposition": disposition,
    }
