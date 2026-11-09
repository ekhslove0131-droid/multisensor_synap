import hashlib
import inspect
import json
from pathlib import Path

import pytest

from multisensor_ml.verified_exploratory_reference import (
    apply_verified_exploratory_reference,
    read_verified_exploratory_baseline,
)


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_documents(tmp_path):
    subject = "00000000-0000-5000-8000-000000000001"
    release = "00000000-0000-5000-8000-000000000002"
    feature_hash = "055bfcee108dd6e059ddd930eb682ef43760c92bc62ef68cb004953a2ae92ae2"
    statistics = [
        {
            "feature_name": "hr_median_bpm",
            "train_count": 3,
            "validation_count": 1,
            "center": 72.0,
            "scale": 2.0,
            "scale_source": "MAD",
            "validation_residuals": [2.0],
            "validation_mean_residual": 2.0,
            "validation_mean_absolute_residual": 2.0,
            "validation_median_absolute_residual": 2.0,
            "validation_p90_absolute_residual": 2.0,
        },
        {
            "feature_name": "rr_median_ms",
            "train_count": 3,
            "validation_count": 1,
            "center": 810.0,
            "scale": 10.0,
            "scale_source": "MAD",
            "validation_residuals": [2.0],
            "validation_mean_residual": 2.0,
            "validation_mean_absolute_residual": 2.0,
            "validation_median_absolute_residual": 2.0,
            "validation_p90_absolute_residual": 2.0,
        },
        {
            "feature_name": "rr_rmssd_ms",
            "train_count": 3,
            "validation_count": 1,
            "center": 12.0,
            "scale": 4.0,
            "scale_source": "MAD",
            "validation_residuals": [1.0],
            "validation_mean_residual": 1.0,
            "validation_mean_absolute_residual": 1.0,
            "validation_median_absolute_residual": 1.0,
            "validation_p90_absolute_residual": 1.0,
        },
    ]
    parameters = [
        {
            "feature_name": row["feature_name"],
            "center": row["center"],
            "scale": row["scale"],
            "scale_source": row["scale_source"],
        }
        for row in statistics
    ]
    model_body = {
        "schema_version": "exploratory_personal_reference_model_v1",
        "purpose": "EXPLORATORY_PERSONAL_REFERENCE_V1",
        "role": "EXPLORATORY_PERSONAL_REFERENCE",
        "origin": "REAL_OBSERVED",
        "label_status": "UNREVIEWED",
        "reference_semantics": "OBSERVED_DISTRIBUTION_NOT_CONFIRMED_REST",
        "model_status": "EXPLORATORY_PERSONAL_REFERENCE_FITTED",
        "model_release_uuid": release,
        "source_kind": "h10",
        "feature_schema_uuid": "c191c0cb-c5d0-45f2-bc53-69ab3fe97c1b",
        "feature_schema_hash": feature_hash,
        "feature_order": ["hr_median_bpm", "rr_median_ms", "rr_rmssd_ms"],
        "training_subject_uuid": subject,
        "cohort_uuid": "00000000-0000-4000-8000-000000000001",
        "source_snapshot_sha256": "a" * 64,
        "source_lineage_digest": "b" * 64,
        "scope_digest": "c" * 64,
        "source_dataset_sha256": "d" * 64,
        "split_policy": {
            "method": "CHRONOLOGICAL_FIRST_70_PERCENT",
            "train_end_ms": 30000,
            "validation_start_ms": 40000,
            "purge_seconds": 10,
        },
        "selection": {
            "candidate_rows": 4,
            "current_veto_count": 0,
            "original_admission_counts": {"REVIEW_REQUIRED": 4},
            "selected_before_purge": 4,
            "split_counts": {"PURGED": 0, "TRAIN": 3, "VALIDATION": 1},
            "verified_exact_derived": 4,
        },
        "method_version": "TRAIN_MEDIAN_MAD_IQR_V1",
        "random_seed": 0,
        "uses_randomness": False,
        "feature_statistics": statistics,
        "parameters_digest": digest(parameters),
        "fit_call_count": 1,
        "fixture_fit_call_count": 0,
        "validation_status": "DESCRIPTIVE_RESIDUALS_ONLY",
        "delivery_eligible": False,
        "promotion_eligible": False,
        "stage": None,
    }
    model = {**model_body, "content_sha256": digest(model_body)}
    model_path = tmp_path / "model.json"
    model_path.write_bytes(canonical(model))
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    receipt_body = {
        "schema_version": "exploratory_personal_reference_run_receipt_v1",
        "status": "EXPLORATORY_PERSONAL_REFERENCE_FITTED",
        "source_kind": "h10",
        "model_release_uuid": release,
        "source_dataset_sha256": "d" * 64,
        "source_snapshot_sha256": "a" * 64,
        "source_lineage_digest": "b" * 64,
        "scope_digest": "c" * 64,
        "feature_schema_hash": feature_hash,
        "feature_order": ["hr_median_bpm", "rr_median_ms", "rr_rmssd_ms"],
        "parameters_digest": model["parameters_digest"],
        "model_sha256": model_sha,
        "model_content_sha256": model["content_sha256"],
        "train_row_count": 3,
        "validation_row_count": 1,
        "fit_call_count": 1,
        "fixture_fit_call_count": 0,
        "readback_verified": True,
        "delivery_eligible": False,
        "promotion_eligible": False,
    }
    receipt = {**receipt_body, "receipt_content_sha256": digest(receipt_body)}
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(canonical(receipt))
    return {
        "model": model,
        "model_path": model_path,
        "model_sha": model_sha,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "subject": subject,
        "feature_hash": feature_hash,
    }


def apply(paths, **overrides):
    values = {
        "model_path": paths["model_path"],
        "expected_model_sha256": paths["model_sha"],
        "receipt_path": paths["receipt_path"],
        "expected_receipt_sha256": paths["receipt_sha"],
        "expected_training_subject_uuid": paths["subject"],
        "expected_source_kind": "h10",
        "expected_feature_schema_hash": paths["feature_hash"],
        "feature_values": {
            "hr_median_bpm": 76.0,
            "rr_median_ms": 830.0,
            "rr_rmssd_ms": 16.0,
        },
    }
    values.update(overrides)
    return apply_verified_exploratory_reference(**values)


def read_baseline(paths, **overrides):
    values = {
        "model_path": paths["model_path"],
        "expected_model_sha256": paths["model_sha"],
        "receipt_path": paths["receipt_path"],
        "expected_receipt_sha256": paths["receipt_sha"],
        "expected_training_subject_uuid": paths["subject"],
        "expected_source_kind": "h10",
        "expected_feature_schema_hash": paths["feature_hash"],
    }
    values.update(overrides)
    return read_verified_exploratory_baseline(**values)


def reseal_receipt(paths, key, value):
    receipt = dict(paths["receipt"])
    receipt[key] = value
    receipt["receipt_content_sha256"] = digest(
        {name: item for name, item in receipt.items() if name != "receipt_content_sha256"}
    )
    paths["receipt_path"].write_bytes(canonical(receipt))
    paths["receipt_sha"] = hashlib.sha256(paths["receipt_path"].read_bytes()).hexdigest()


def reseal_model_and_receipt(paths, feature_index, key, value):
    model = json.loads(canonical(paths["model"]))
    model["feature_statistics"][feature_index][key] = value
    parameters = [
        {
            name: row[name]
            for name in ("feature_name", "center", "scale", "scale_source")
        }
        for row in model["feature_statistics"]
    ]
    model["parameters_digest"] = digest(parameters)
    model["content_sha256"] = digest(
        {name: item for name, item in model.items() if name != "content_sha256"}
    )
    paths["model"] = model
    paths["model_path"].write_bytes(canonical(model))
    paths["model_sha"] = hashlib.sha256(paths["model_path"].read_bytes()).hexdigest()
    receipt = json.loads(canonical(paths["receipt"]))
    receipt["parameters_digest"] = model["parameters_digest"]
    receipt["model_sha256"] = paths["model_sha"]
    receipt["model_content_sha256"] = model["content_sha256"]
    receipt["receipt_content_sha256"] = digest(
        {name: item for name, item in receipt.items() if name != "receipt_content_sha256"}
    )
    paths["receipt"] = receipt
    paths["receipt_path"].write_bytes(canonical(receipt))
    paths["receipt_sha"] = hashlib.sha256(paths["receipt_path"].read_bytes()).hexdigest()


def test_verified_apply_returns_feature_wise_deviation_and_provenance(tmp_path):
    paths = write_documents(tmp_path)

    result = apply(paths)

    assert result == {
        "schema_version": "verified_exploratory_personal_deviation_v1",
        "role": "EXPLORATORY_PERSONAL_REFERENCE",
        "source_kind": "h10",
        "model_release_uuid": "00000000-0000-5000-8000-000000000002",
        "model_sha256": paths["model_sha"],
        "receipt_sha256": paths["receipt_sha"],
        "training_subject_uuid": "00000000-0000-5000-8000-000000000001",
        "feature_schema_uuid": "c191c0cb-c5d0-45f2-bc53-69ab3fe97c1b",
        "feature_schema_hash": paths["feature_hash"],
        "features": [
            {
                "code": "hr_median_bpm",
                "unit": "bpm",
                "current_value": 76.0,
                "baseline_center": 72.0,
                "baseline_scale": 2.0,
                "deviation_z": 2.0,
            },
            {
                "code": "rr_median_ms",
                "unit": "ms",
                "current_value": 830.0,
                "baseline_center": 810.0,
                "baseline_scale": 10.0,
                "deviation_z": 2.0,
            },
            {
                "code": "rr_rmssd_ms",
                "unit": "ms",
                "current_value": 16.0,
                "baseline_center": 12.0,
                "baseline_scale": 4.0,
                "deviation_z": 1.0,
            },
        ],
        "semantics": "CURRENT_BASELINE_DEVIATION",
        "label_status": "UNREVIEWED",
        "delivery_eligible": False,
        "promotion_eligible": False,
    }


def test_verified_baseline_returns_parameters_without_current_features(tmp_path):
    paths = write_documents(tmp_path)

    result = read_baseline(paths)

    assert result == {
        "schema_version": "verified_exploratory_personal_baseline_v1",
        "role": "EXPLORATORY_PERSONAL_REFERENCE",
        "source_kind": "h10",
        "model_release_uuid": "00000000-0000-5000-8000-000000000002",
        "model_sha256": paths["model_sha"],
        "receipt_sha256": paths["receipt_sha"],
        "training_subject_uuid": "00000000-0000-5000-8000-000000000001",
        "feature_schema_uuid": "c191c0cb-c5d0-45f2-bc53-69ab3fe97c1b",
        "feature_schema_hash": paths["feature_hash"],
        "features": [
            {
                "code": "hr_median_bpm",
                "unit": "bpm",
                "baseline_center": 72.0,
                "baseline_scale": 2.0,
            },
            {
                "code": "rr_median_ms",
                "unit": "ms",
                "baseline_center": 810.0,
                "baseline_scale": 10.0,
            },
            {
                "code": "rr_rmssd_ms",
                "unit": "ms",
                "baseline_center": 12.0,
                "baseline_scale": 4.0,
            },
        ],
        "semantics": "PERSONAL_BASELINE_PARAMETERS",
        "label_status": "UNREVIEWED",
        "delivery_eligible": False,
        "promotion_eligible": False,
    }


def test_verified_baseline_api_has_no_current_feature_argument():
    assert "feature_values" not in inspect.signature(
        read_verified_exploratory_baseline
    ).parameters


def test_verified_baseline_reads_model_and_receipt_once(tmp_path, monkeypatch):
    paths = write_documents(tmp_path)
    original = Path.read_bytes
    counts = {paths["model_path"]: 0, paths["receipt_path"]: 0}

    def counted(path):
        if path in counts:
            counts[path] += 1
            if counts[path] > 1:
                raise AssertionError("verified file was reopened after validation")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)

    read_baseline(paths)

    assert counts == {paths["model_path"]: 1, paths["receipt_path"]: 1}


@pytest.mark.parametrize(
    "feature_index,key,value,error",
    [
        (0, "center", 0.0, "BASELINE_CENTER_DOMAIN_INVALID"),
        (1, "scale", 0.0, "MODEL_SCALE_OR_COUNT_INVALID"),
    ],
)
def test_verified_baseline_rejects_invalid_source_statistics(
    tmp_path, feature_index, key, value, error
):
    paths = write_documents(tmp_path)
    reseal_model_and_receipt(paths, feature_index, key, value)

    with pytest.raises(ValueError, match=error):
        read_baseline(paths)


def test_model_and_receipt_are_each_read_once(tmp_path, monkeypatch):
    paths = write_documents(tmp_path)
    original = Path.read_bytes
    counts = {paths["model_path"]: 0, paths["receipt_path"]: 0}

    def counted(path):
        if path in counts:
            counts[path] += 1
            if counts[path] > 1:
                raise AssertionError("verified file was reopened after validation")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)

    apply(paths)

    assert counts == {paths["model_path"]: 1, paths["receipt_path"]: 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_sha256", "0" * 64),
        ("model_content_sha256", "0" * 64),
        ("source_dataset_sha256", "0" * 64),
        ("feature_schema_hash", "0" * 64),
        ("source_kind", "watch"),
        ("model_release_uuid", "00000000-0000-5000-8000-000000000003"),
        ("fit_call_count", 0),
        ("readback_verified", False),
        ("delivery_eligible", True),
        ("promotion_eligible", True),
    ],
)
def test_resealed_but_misbound_receipt_is_rejected(tmp_path, field, value):
    paths = write_documents(tmp_path)
    reseal_receipt(paths, field, value)

    with pytest.raises(ValueError, match="RECEIPT_MODEL_BINDING_MISMATCH"):
        apply(paths)


@pytest.mark.parametrize(
    "override",
    [
        {"expected_model_sha256": "0" * 64},
        {"expected_receipt_sha256": "0" * 64},
        {"expected_training_subject_uuid": "00000000-0000-5000-8000-000000000009"},
        {"expected_source_kind": "watch"},
        {"expected_feature_schema_hash": "0" * 64},
    ],
)
def test_caller_binding_mismatch_is_rejected(tmp_path, override):
    paths = write_documents(tmp_path)

    with pytest.raises(ValueError):
        apply(paths, **override)


@pytest.mark.parametrize(
    "feature_values",
    [
        {"hr_median_bpm": 0.0, "rr_median_ms": 830.0, "rr_rmssd_ms": 16.0},
        {"hr_median_bpm": 76.0, "rr_median_ms": -1.0, "rr_rmssd_ms": 16.0},
        {"hr_median_bpm": 76.0, "rr_median_ms": 830.0, "rr_rmssd_ms": -1.0},
        {"hr_median_bpm": 76.0, "rr_median_ms": 830.0, "rr_rmssd_ms": float("nan")},
    ],
)
def test_current_feature_domain_violation_is_rejected(tmp_path, feature_values):
    paths = write_documents(tmp_path)

    with pytest.raises(ValueError, match="CURRENT_FEATURE_DOMAIN_INVALID"):
        apply(paths, feature_values=feature_values)


def test_receipt_self_digest_is_required(tmp_path):
    paths = write_documents(tmp_path)
    receipt = dict(paths["receipt"])
    receipt["receipt_content_sha256"] = "0" * 64
    paths["receipt_path"].write_bytes(canonical(receipt))
    paths["receipt_sha"] = hashlib.sha256(paths["receipt_path"].read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="RECEIPT_CONTENT_DIGEST_MISMATCH"):
        apply(paths)
