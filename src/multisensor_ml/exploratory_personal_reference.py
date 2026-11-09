"""Actual feature-only exploratory personal reference fitting and readback."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from multisensor_ml.physiology_baseline_math import fit_center_scale
from multisensor_ml.physiology_compatibility import consume_descriptors, load_schema

TOP_FIELDS = {
    "schema_version",
    "purpose",
    "origin",
    "label_status",
    "reference_semantics",
    "cohort_uuid",
    "training_subject_uuid",
    "source_kind",
    "feature_schema",
    "source_snapshot_sha256",
    "source_lineage_digest",
    "scope_digest",
    "split_policy",
    "selection",
    "rows",
}
ROW_FIELDS = {
    "training_capture_set_uuid",
    "exact_window_id",
    "source_row_digest",
    "support_start_ms",
    "support_end_ms",
    "split_role",
    "feature_values",
    "quality_valid",
    "admission_state",
}
SELECTION_FIELDS = {
    "candidate_rows",
    "current_veto_count",
    "original_admission_counts",
    "selected_before_purge",
    "split_counts",
    "verified_exact_derived",
}
MODEL_FIELDS = {
    "schema_version",
    "purpose",
    "role",
    "origin",
    "label_status",
    "reference_semantics",
    "model_status",
    "model_release_uuid",
    "source_kind",
    "feature_schema_uuid",
    "feature_schema_hash",
    "feature_order",
    "training_subject_uuid",
    "cohort_uuid",
    "source_snapshot_sha256",
    "source_lineage_digest",
    "scope_digest",
    "source_dataset_sha256",
    "split_policy",
    "selection",
    "method_version",
    "random_seed",
    "uses_randomness",
    "feature_statistics",
    "parameters_digest",
    "fit_call_count",
    "fixture_fit_call_count",
    "validation_status",
    "delivery_eligible",
    "promotion_eligible",
    "stage",
    "content_sha256",
}
FORBIDDEN_KEYS = {
    "account_uuid",
    "person_uuid",
    "source_uuid",
    "capture_session_uuid",
    "raw_uri",
    "raw_path",
    "training_eligible",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field.upper()}_INVALID")
    return value


def _require_uuid(value: object, field: str, version: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field.upper()}_INVALID")
    parsed = UUID(value)
    if parsed.version != version or str(parsed) != value:
        raise ValueError(f"{field.upper()}_INVALID")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field.upper()}_INVALID")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field.upper()}_INVALID")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field.upper()}_INVALID")
    return number


def _validate_selection(selection: object, row_count: int) -> dict[str, Any]:
    if not isinstance(selection, Mapping) or set(selection) != SELECTION_FIELDS:
        raise ValueError("SELECTION_FIELDS_INVALID")
    for key in (
        "candidate_rows",
        "current_veto_count",
        "selected_before_purge",
        "verified_exact_derived",
    ):
        _nonnegative_int(selection[key], key)
    admissions = selection["original_admission_counts"]
    if not isinstance(admissions, Mapping) or not admissions:
        raise ValueError("SELECTION_ADMISSION_COUNTS_INVALID")
    if set(admissions) - {"REVIEW_REQUIRED", "EXCLUDED_QUALITY", "EXCLUDED_PATTERN_PRESENT"}:
        raise ValueError("SELECTION_ADMISSION_COUNTS_INVALID")
    if any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in admissions.items()
    ):
        raise ValueError("SELECTION_ADMISSION_COUNTS_INVALID")
    splits = selection["split_counts"]
    if not isinstance(splits, Mapping) or set(splits) != {"TRAIN", "VALIDATION", "PURGED"}:
        raise ValueError("SELECTION_SPLIT_COUNTS_INVALID")
    if any(type(value) is not int or value < 0 for value in splits.values()):
        raise ValueError("SELECTION_SPLIT_COUNTS_INVALID")
    if (
        sum(admissions.values()) != selection["candidate_rows"]
        or sum(splits.values()) != selection["selected_before_purge"]
        or splits["TRAIN"] + splits["VALIDATION"] != row_count
        or admissions.get("REVIEW_REQUIRED", 0) != selection["selected_before_purge"]
        or selection["verified_exact_derived"] != selection["selected_before_purge"]
        or selection["current_veto_count"] != 0
    ):
        raise ValueError("SELECTION_COUNTS_MISMATCH")
    return cast(dict[str, Any], json.loads(_canonical(selection)))


def _load_dataset(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    expected = _require_sha(expected_sha256, "expected_dataset_sha256")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError("DATASET_FILE_SHA256_MISMATCH")
    value = json.loads(payload)
    if not isinstance(value, dict) or set(value) != TOP_FIELDS:
        raise ValueError("DATASET_FIELDS_INVALID")
    return value, actual


def _prepare(
    dataset: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[float, ...]], list[tuple[float, ...]]]:
    if (
        dataset["schema_version"] != "exploratory_personal_dataset_v1"
        or dataset["purpose"] != "EXPLORATORY_PERSONAL_REFERENCE_V1"
        or dataset["origin"] != "REAL_OBSERVED"
        or dataset["label_status"] != "UNREVIEWED"
        or dataset["reference_semantics"] != "OBSERVED_DISTRIBUTION_NOT_CONFIRMED_REST"
    ):
        raise ValueError("DATASET_SEMANTICS_INVALID")
    cohort = _require_uuid(dataset["cohort_uuid"], "cohort_uuid", 4)
    subject = _require_uuid(dataset["training_subject_uuid"], "training_subject_uuid", 5)
    for field in ("source_snapshot_sha256", "source_lineage_digest", "scope_digest"):
        _require_sha(dataset[field], field)
    source = dataset["source_kind"]
    if not isinstance(dataset["feature_schema"], Mapping):
        raise ValueError("FEATURE_SCHEMA_INVALID")
    schema = load_schema(dataset["feature_schema"], source)
    split = dataset["split_policy"]
    if not isinstance(split, Mapping) or set(split) != {
        "method",
        "train_end_ms",
        "validation_start_ms",
        "purge_seconds",
    }:
        raise ValueError("SPLIT_POLICY_INVALID")
    if split["method"] != "CHRONOLOGICAL_FIRST_70_PERCENT":
        raise ValueError("SPLIT_METHOD_INVALID")
    train_end = _nonnegative_int(split["train_end_ms"], "train_end_ms")
    validation_start = _nonnegative_int(split["validation_start_ms"], "validation_start_ms")
    purge = _nonnegative_int(split["purge_seconds"], "purge_seconds")
    if purge != 10 or validation_start != train_end + purge * 1000:
        raise ValueError("PURGE_INVALID")
    rows = dataset["rows"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("ROWS_INVALID")
    selection = _validate_selection(dataset["selection"], len(rows))
    if any(key in FORBIDDEN_KEYS for key in dataset):
        raise ValueError("PRIVATE_OR_RECLASSIFIED_FIELD")
    specs = {descriptor["code"]: descriptor for descriptor in schema["descriptors"]}
    expected_features = tuple(schema["core_features"])
    seen_windows: set[str] = set()
    seen_rows: set[str] = set()
    capture_owners: set[str] = set()
    ordered_rows = sorted(rows, key=lambda row: (row["support_start_ms"], row["exact_window_id"]))
    train: list[tuple[float, ...]] = []
    validation: list[tuple[float, ...]] = []
    prior_end: int | None = None
    for row in ordered_rows:
        if not isinstance(row, Mapping) or set(row) != ROW_FIELDS:
            raise ValueError("ROW_FIELDS_INVALID")
        capture_owners.add(_require_uuid(row["training_capture_set_uuid"], "capture", 5))
        window_id = _require_sha(row["exact_window_id"], "exact_window_id")
        source_row = _require_sha(row["source_row_digest"], "source_row_digest")
        if window_id in seen_windows or source_row in seen_rows:
            raise ValueError("DUPLICATE_ROW")
        seen_windows.add(window_id)
        seen_rows.add(source_row)
        start = _nonnegative_int(row["support_start_ms"], "support_start_ms")
        end = _nonnegative_int(row["support_end_ms"], "support_end_ms")
        if end - start != 10000 or (prior_end is not None and start < prior_end):
            raise ValueError("SUPPORT_INVALID")
        prior_end = end
        if row["quality_valid"] is not True or row["admission_state"] != "REVIEW_REQUIRED":
            raise ValueError("QUALITY_OR_ADMISSION_INVALID")
        features = row["feature_values"]
        if not isinstance(features, Mapping) or set(features) != set(expected_features):
            raise ValueError("FEATURE_VALUES_INVALID")
        descriptors = [
            {
                "code": name,
                "unit": specs[name]["unit"],
                "value": features[name],
                "readiness": "AVAILABLE",
                "reason": None,
            }
            for name in expected_features
        ]
        vector = consume_descriptors(schema, descriptors)
        if vector is None:
            raise ValueError("CORE_FEATURE_INVALID")
        role = row["split_role"]
        if role == "TRAIN" and end <= train_end:
            train.append(vector)
        elif role == "VALIDATION" and start >= validation_start:
            validation.append(vector)
        else:
            raise ValueError("SPLIT_SUPPORT_INVALID")
    if not train or not validation:
        raise ValueError("SPLIT_EMPTY")
    if (
        len(train) != selection["split_counts"]["TRAIN"]
        or len(validation) != selection["split_counts"]["VALIDATION"]
    ):
        raise ValueError("SPLIT_COUNT_MISMATCH")
    prepared = {
        "cohort_uuid": cohort,
        "training_subject_uuid": subject,
        "source_kind": source,
        "feature_schema": schema,
        "source_snapshot_sha256": dataset["source_snapshot_sha256"],
        "source_lineage_digest": dataset["source_lineage_digest"],
        "scope_digest": dataset["scope_digest"],
        "split_policy": json.loads(_canonical(split)),
        "selection": selection,
        "capture_set_count": len(capture_owners),
    }
    return prepared, train, validation


def _model_document(
    prepared: Mapping[str, Any],
    statistics: list[dict[str, Any]],
    method: str,
    dataset_sha256: str,
    evidence_mode: str,
) -> dict[str, Any]:
    parameters = [
        {
            "feature_name": row["feature_name"],
            "center": row["center"],
            "scale": row["scale"],
            "scale_source": row["scale_source"],
        }
        for row in statistics
    ]
    parameters_digest = _digest(parameters)
    release = str(
        uuid5(
            NAMESPACE_URL,
            f"{method}:{prepared['source_kind']}:{dataset_sha256}:{parameters_digest}",
        )
    )
    actual = evidence_mode == "REAL_OBSERVED"
    body = {
        "schema_version": "exploratory_personal_reference_model_v1",
        "purpose": "EXPLORATORY_PERSONAL_REFERENCE_V1",
        "role": "EXPLORATORY_PERSONAL_REFERENCE",
        "origin": "REAL_OBSERVED" if actual else "CONTRACT_FIXTURE",
        "label_status": "UNREVIEWED",
        "reference_semantics": "OBSERVED_DISTRIBUTION_NOT_CONFIRMED_REST",
        "model_status": (
            "EXPLORATORY_PERSONAL_REFERENCE_FITTED" if actual else "CONTRACT_EVIDENCE_ONLY"
        ),
        "model_release_uuid": release,
        "source_kind": prepared["source_kind"],
        "feature_schema_uuid": prepared["feature_schema"]["schema_uuid"],
        "feature_schema_hash": _digest(prepared["feature_schema"]),
        "feature_order": list(prepared["feature_schema"]["core_features"]),
        "training_subject_uuid": prepared["training_subject_uuid"],
        "cohort_uuid": prepared["cohort_uuid"],
        "source_snapshot_sha256": prepared["source_snapshot_sha256"],
        "source_lineage_digest": prepared["source_lineage_digest"],
        "scope_digest": prepared["scope_digest"],
        "source_dataset_sha256": dataset_sha256,
        "split_policy": prepared["split_policy"],
        "selection": prepared["selection"],
        "method_version": method,
        "random_seed": 0,
        "uses_randomness": False,
        "feature_statistics": statistics,
        "parameters_digest": parameters_digest,
        "fit_call_count": 1 if actual else 0,
        "fixture_fit_call_count": 0 if actual else 1,
        "validation_status": "DESCRIPTIVE_RESIDUALS_ONLY",
        "delivery_eligible": False,
        "promotion_eligible": False,
        "stage": None,
    }
    return {**body, "content_sha256": _digest(body)}


def _validate_model(document: object) -> dict[str, Any]:
    if not isinstance(document, dict) or set(document) != MODEL_FIELDS:
        raise ValueError("MODEL_FIELDS_INVALID")
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    if document["content_sha256"] != _digest(body):
        raise ValueError("MODEL_CONTENT_DIGEST_MISMATCH")
    if (
        document["schema_version"] != "exploratory_personal_reference_model_v1"
        or document["purpose"] != "EXPLORATORY_PERSONAL_REFERENCE_V1"
        or document["role"] != "EXPLORATORY_PERSONAL_REFERENCE"
        or document["label_status"] != "UNREVIEWED"
        or document["reference_semantics"] != "OBSERVED_DISTRIBUTION_NOT_CONFIRMED_REST"
        or document["delivery_eligible"] is not False
        or document["promotion_eligible"] is not False
        or document["stage"] is not None
        or document["uses_randomness"] is not False
        or document["random_seed"] != 0
        or document["method_version"] != "TRAIN_MEDIAN_MAD_IQR_V1"
    ):
        raise ValueError("MODEL_SEMANTICS_INVALID")
    _require_uuid(document["model_release_uuid"], "model_release_uuid", 5)
    _require_uuid(document["cohort_uuid"], "cohort_uuid", 4)
    _require_uuid(document["training_subject_uuid"], "training_subject_uuid", 5)
    for field in (
        "feature_schema_hash",
        "source_snapshot_sha256",
        "source_lineage_digest",
        "scope_digest",
        "source_dataset_sha256",
        "parameters_digest",
        "content_sha256",
    ):
        _require_sha(document[field], field)
    statistics = document["feature_statistics"]
    if (
        not isinstance(statistics, list)
        or [row.get("feature_name") for row in statistics] != document["feature_order"]
    ):
        raise ValueError("MODEL_FEATURE_ORDER_INVALID")
    parameters = []
    for row in statistics:
        expected = {
            "feature_name",
            "train_count",
            "validation_count",
            "center",
            "scale",
            "scale_source",
            "validation_residuals",
            "validation_mean_residual",
            "validation_mean_absolute_residual",
            "validation_median_absolute_residual",
            "validation_p90_absolute_residual",
        }
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("MODEL_FEATURE_STATISTICS_INVALID")
        numbers = [
            row["center"],
            row["scale"],
            row["validation_mean_residual"],
            row["validation_mean_absolute_residual"],
            row["validation_median_absolute_residual"],
            row["validation_p90_absolute_residual"],
            *row["validation_residuals"],
        ]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in numbers):
            raise ValueError("MODEL_NONFINITE")
        if row["scale"] <= 0 or row["train_count"] <= 0 or row["validation_count"] <= 0:
            raise ValueError("MODEL_SCALE_OR_COUNT_INVALID")
        if len(row["validation_residuals"]) != row["validation_count"]:
            raise ValueError("MODEL_RESIDUAL_COUNT_INVALID")
        parameters.append(
            {key: row[key] for key in ("feature_name", "center", "scale", "scale_source")}
        )
    if document["parameters_digest"] != _digest(parameters):
        raise ValueError("MODEL_PARAMETERS_DIGEST_MISMATCH")
    if (
        document["origin"],
        document["model_status"],
        document["fit_call_count"],
        document["fixture_fit_call_count"],
    ) not in {
        ("REAL_OBSERVED", "EXPLORATORY_PERSONAL_REFERENCE_FITTED", 1, 0),
        ("CONTRACT_FIXTURE", "CONTRACT_EVIDENCE_ONLY", 0, 1),
    }:
        raise ValueError("MODEL_EVIDENCE_MODE_INVALID")
    return cast(dict[str, Any], json.loads(_canonical(document)))


def read_exploratory_model(path: Path) -> dict[str, Any]:
    return _validate_model(json.loads(path.read_bytes()))


def apply_exploratory_reference(
    *,
    model_path: Path,
    feature_schema_hash: str,
    feature_values: Mapping[str, object],
) -> dict[str, Any]:
    """Return current feature-wise deviations from an exploratory personal reference."""
    model = read_exploratory_model(model_path)
    if _require_sha(feature_schema_hash, "feature_schema_hash") != model["feature_schema_hash"]:
        raise ValueError("FEATURE_SCHEMA_HASH_MISMATCH")
    order = model["feature_order"]
    if set(feature_values) != set(order):
        raise ValueError("FEATURE_VALUES_INVALID")
    values = [_finite_number(feature_values[name], "feature_values") for name in order]
    residuals = [
        (float(value) - float(statistic["center"])) / float(statistic["scale"])
        for value, statistic in zip(values, model["feature_statistics"], strict=True)
    ]
    if not all(math.isfinite(value) for value in residuals):
        raise ValueError("RESIDUAL_NONFINITE")
    return {
        "schema_version": "exploratory_personal_deviation_v1",
        "role": model["role"],
        "source_kind": model["source_kind"],
        "model_release_uuid": model["model_release_uuid"],
        "feature_schema_hash": model["feature_schema_hash"],
        "feature_order": list(order),
        "residuals": residuals,
        "semantics": "CURRENT_BASELINE_DEVIATION",
        "label_status": "UNREVIEWED",
        "delivery_eligible": False,
        "promotion_eligible": False,
    }


def _write_create_only(path: Path, value: object) -> str:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def fit_exploratory_personal_reference(
    *,
    dataset_path: Path,
    expected_dataset_sha256: str,
    model_output_path: Path,
    receipt_output_path: Path,
    evidence_mode: str,
    allow_real_fit: bool = False,
) -> dict[str, Any]:
    if model_output_path.exists() or receipt_output_path.exists():
        raise FileExistsError("exploratory output already exists")
    blocked = {
        "status": "BLOCKED_INPUT_INVALID",
        "fit_call_count": 0,
        "fixture_fit_call_count": 0,
        "delivery_eligible": False,
        "promotion_eligible": False,
    }
    try:
        if evidence_mode not in {"CONTRACT_FIXTURE", "REAL_OBSERVED"}:
            raise ValueError("EVIDENCE_MODE_INVALID")
        if evidence_mode == "REAL_OBSERVED" and allow_real_fit is not True:
            raise ValueError("REAL_FIT_NOT_AUTHORIZED")
        dataset, dataset_sha256 = _load_dataset(dataset_path, expected_dataset_sha256)
        prepared, train, validation = _prepare(dataset)
        counter = "fit_call_count" if evidence_mode == "REAL_OBSERVED" else "fixture_fit_call_count"
        blocked[counter] = 1
        statistics, method = fit_center_scale(
            train, validation, prepared["feature_schema"]["core_features"]
        )
        document = _model_document(prepared, statistics, method, dataset_sha256, evidence_mode)
        _validate_model(document)
        model_sha256 = _write_create_only(model_output_path, document)
        readback = read_exploratory_model(model_output_path)
        if readback != document:
            raise ValueError("MODEL_READBACK_MISMATCH")
        result = {
            "schema_version": "exploratory_personal_reference_run_receipt_v1",
            "status": document["model_status"],
            "source_kind": document["source_kind"],
            "model_release_uuid": document["model_release_uuid"],
            "source_dataset_sha256": dataset_sha256,
            "source_snapshot_sha256": document["source_snapshot_sha256"],
            "source_lineage_digest": document["source_lineage_digest"],
            "scope_digest": document["scope_digest"],
            "feature_schema_hash": document["feature_schema_hash"],
            "feature_order": document["feature_order"],
            "parameters_digest": document["parameters_digest"],
            "model_sha256": model_sha256,
            "model_content_sha256": document["content_sha256"],
            "train_row_count": len(train),
            "validation_row_count": len(validation),
            "fit_call_count": document["fit_call_count"],
            "fixture_fit_call_count": document["fixture_fit_call_count"],
            "readback_verified": True,
            "delivery_eligible": False,
            "promotion_eligible": False,
        }
        receipt = {**result, "receipt_content_sha256": _digest(result)}
        receipt_sha256 = _write_create_only(receipt_output_path, receipt)
        return {**result, "receipt_sha256": receipt_sha256}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, OSError) as error:
        reason = str(error)
        safe = reason if reason.isupper() and reason.replace("_", "").isalpha() else "INPUT_INVALID"
        return {**blocked, "status": f"BLOCKED_{safe}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--execute-real-observed", action="store_true")
    args = parser.parse_args()
    result = fit_exploratory_personal_reference(
        dataset_path=args.dataset,
        expected_dataset_sha256=args.dataset_sha256,
        model_output_path=args.model_output,
        receipt_output_path=args.receipt_output,
        evidence_mode="REAL_OBSERVED",
        allow_real_fit=args.execute_real_observed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "EXPLORATORY_PERSONAL_REFERENCE_FITTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
