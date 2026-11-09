import hashlib
import json
import stat
from pathlib import Path

import pytest

from multisensor_ml.exploratory_personal_reference import (
    apply_exploratory_reference,
    fit_exploratory_personal_reference,
    read_exploratory_model,
)

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(source="h10"):
    stem = {
        "watch": "watch_physio_core_v1",
        "h10": "h10_physio_core_v1",
        "muse": "muse_eeg_core_v1",
    }[source]
    schema = json.loads((ROOT / "contracts/physiology/v1" / f"{stem}.json").read_text())
    values = {
        "watch": [(60.0, 1.0, -0.1), (62.0, 2.0, 0.0), (64.0, 3.0, 0.1), (66.0, 4.0, 0.2)],
        "h10": [(70.0, 800.0, 10.0), (72.0, 810.0, 12.0), (74.0, 820.0, 14.0), (76.0, 830.0, 16.0)],
        "muse": [
            tuple([0.05, 0.1, 0.6, 0.2, 0.05, 0.4] * 4),
            tuple([0.06, 0.11, 0.58, 0.19, 0.06, 0.41] * 4),
            tuple([0.07, 0.12, 0.56, 0.18, 0.07, 0.42] * 4),
            tuple([0.08, 0.13, 0.54, 0.17, 0.08, 0.43] * 4),
        ],
    }[source]
    rows = []
    for index, vector in enumerate(values):
        start = (index if index < 3 else 4) * 10000
        rows.append(
            {
                "training_capture_set_uuid": "00000000-0000-5000-8000-000000000002",
                "exact_window_id": f"{index + 1}" * 64,
                "source_row_digest": f"{index + 5}" * 64,
                "support_start_ms": start,
                "support_end_ms": start + 10000,
                "split_role": "TRAIN" if index < 3 else "VALIDATION",
                "feature_values": dict(zip(schema["core_features"], vector, strict=True)),
                "quality_valid": True,
                "admission_state": "REVIEW_REQUIRED",
            }
        )
    return {
        "schema_version": "exploratory_personal_dataset_v1",
        "purpose": "EXPLORATORY_PERSONAL_REFERENCE_V1",
        "origin": "REAL_OBSERVED",
        "label_status": "UNREVIEWED",
        "reference_semantics": "OBSERVED_DISTRIBUTION_NOT_CONFIRMED_REST",
        "cohort_uuid": "00000000-0000-4000-8000-000000000001",
        "training_subject_uuid": "00000000-0000-5000-8000-000000000001",
        "source_kind": source,
        "feature_schema": schema,
        "source_snapshot_sha256": "a" * 64,
        "source_lineage_digest": "b" * 64,
        "scope_digest": "c" * 64,
        "split_policy": {
            "method": "CHRONOLOGICAL_FIRST_70_PERCENT",
            "train_end_ms": 30000,
            "validation_start_ms": 40000,
            "purge_seconds": 10,
        },
        "selection": {
            "candidate_rows": 9,
            "current_veto_count": 0,
            "original_admission_counts": {
                "EXCLUDED_QUALITY": 3,
                "EXCLUDED_PATTERN_PRESENT": 2,
                "REVIEW_REQUIRED": 4,
            },
            "selected_before_purge": 4,
            "split_counts": {"PURGED": 0, "TRAIN": 3, "VALIDATION": 1},
            "verified_exact_derived": 4,
        },
        "rows": rows,
    }


def write_dataset(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(dataset))
    return file_sha(path)


def run(tmp_path, source="h10", mode="CONTRACT_FIXTURE", dataset=None):
    dataset = dataset or fixture(source)
    path = tmp_path / f"{source}-dataset.json"
    checksum = write_dataset(path, dataset)
    result = fit_exploratory_personal_reference(
        dataset_path=path,
        expected_dataset_sha256=checksum,
        model_output_path=tmp_path / f"{source}-model.json",
        receipt_output_path=tmp_path / f"{source}-receipt.json",
        evidence_mode=mode,
        allow_real_fit=mode == "REAL_OBSERVED",
    )
    return result, dataset, path


@pytest.mark.parametrize("source,width", [("watch", 3), ("h10", 3), ("muse", 24)])
def test_each_source_fits_independently_and_readback_is_verified(tmp_path, source, width):
    result, dataset, _ = run(tmp_path, source)
    assert result["status"] == "CONTRACT_EVIDENCE_ONLY"
    assert result["fit_call_count"] == 0 and result["fixture_fit_call_count"] == 1
    assert result["readback_verified"] is True
    model = read_exploratory_model(tmp_path / f"{source}-model.json")
    assert len(model["feature_statistics"]) == width
    assert model["role"] == "EXPLORATORY_PERSONAL_REFERENCE"
    assert model["label_status"] == "UNREVIEWED"
    assert model["reference_semantics"] == "OBSERVED_DISTRIBUTION_NOT_CONFIRMED_REST"
    assert model["delivery_eligible"] is model["promotion_eligible"] is False
    assert model["source_dataset_sha256"] == file_sha(tmp_path / f"{source}-dataset.json")
    assert model["source_snapshot_sha256"] == dataset["source_snapshot_sha256"]
    assert "stress" not in json.dumps(model).lower()
    assert "accuracy" not in json.dumps(model).lower()


def test_actual_origin_records_one_real_fit_and_fixed_method(tmp_path):
    result, _, _ = run(tmp_path, mode="REAL_OBSERVED")
    assert result["status"] == "EXPLORATORY_PERSONAL_REFERENCE_FITTED"
    assert result["fit_call_count"] == 1 and result["fixture_fit_call_count"] == 0
    assert stat.S_IMODE((tmp_path / "h10-model.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "h10-receipt.json").stat().st_mode) == 0o600
    model = read_exploratory_model(tmp_path / "h10-model.json")
    assert model["method_version"] == "TRAIN_MEDIAN_MAD_IQR_V1"
    assert model["random_seed"] == 0 and model["uses_randomness"] is False
    stats = model["feature_statistics"]
    assert stats[0]["center"] == 72
    assert stats[0]["scale"] == pytest.approx(2.9652)
    assert stats[0]["validation_residuals"] == pytest.approx([4 / 2.9652])
    assert stats[0]["train_count"] == 3 and stats[0]["validation_count"] == 1
    applied = apply_exploratory_reference(
        model_path=tmp_path / "h10-model.json",
        feature_schema_hash=model["feature_schema_hash"],
        feature_values={"hr_median_bpm": 76, "rr_median_ms": 830, "rr_rmssd_ms": 16},
    )
    assert applied["feature_order"] == ["hr_median_bpm", "rr_median_ms", "rr_rmssd_ms"]
    assert applied["residuals"] == pytest.approx([4 / 2.9652, 20 / 14.826, 4 / 2.9652])
    assert applied["semantics"] == "CURRENT_BASELINE_DEVIATION"
    assert applied["delivery_eligible"] is False
    with pytest.raises(ValueError):
        apply_exploratory_reference(
            model_path=tmp_path / "h10-model.json",
            feature_schema_hash=model["feature_schema_hash"],
            feature_values={"hr_median_bpm": 76, "rr_median_ms": 830},
        )


def test_validation_does_not_change_train_center_or_scale(tmp_path):
    first, _, _ = run(tmp_path / "one")
    dataset = fixture()
    dataset["rows"][-1]["feature_values"] = dict.fromkeys(
        dataset["rows"][-1]["feature_values"], 999.0
    )
    second, _, _ = run(tmp_path / "two", dataset=dataset)
    assert first["parameters_digest"] == second["parameters_digest"]


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "private",
        "eligible",
        "label",
        "admission",
        "schema",
        "duplicate",
        "overlap",
        "purge",
        "subject",
        "selection_identity",
    ],
)
def test_invalid_or_reclassified_data_fails_before_fit(tmp_path, fault):
    dataset = fixture()
    if fault == "private":
        dataset["person_uuid"] = "private"
    elif fault == "eligible":
        dataset["rows"][0]["training_eligible"] = True
    elif fault == "label":
        dataset["label_status"] = "NO_PATTERN"
    elif fault == "admission":
        dataset["rows"][0]["admission_state"] = "ELIGIBLE_NO_PATTERN"
    elif fault == "schema":
        dataset["feature_schema"]["schema_uuid"] = "00000000-0000-4000-8000-000000000009"
    elif fault == "duplicate":
        dataset["rows"][1]["exact_window_id"] = dataset["rows"][0]["exact_window_id"]
    elif fault == "overlap":
        dataset["rows"][1].update(support_start_ms=5000, support_end_ms=15000)
    elif fault == "purge":
        dataset["split_policy"]["purge_seconds"] = 20
    elif fault == "subject":
        dataset["training_subject_uuid"] = "00000000-0000-4000-8000-000000000009"
    elif fault == "selection_identity":
        dataset["selection"]["person_uuid"] = 1
    path = tmp_path / "bad.json"
    checksum = write_dataset(path, dataset)
    if fault == "hash":
        checksum = "0" * 64
    result = fit_exploratory_personal_reference(
        dataset_path=path,
        expected_dataset_sha256=checksum,
        model_output_path=tmp_path / "model.json",
        receipt_output_path=tmp_path / "receipt.json",
        evidence_mode="CONTRACT_FIXTURE",
    )
    assert result["status"].startswith("BLOCKED")
    assert result["fit_call_count"] == result["fixture_fit_call_count"] == 0
    assert not (tmp_path / "model.json").exists()


def test_scale_zero_is_train_only_failure_and_create_only(tmp_path):
    dataset = fixture()
    for row in dataset["rows"][:3]:
        row["feature_values"] = dict.fromkeys(row["feature_values"], 1.0)
    result, _, _ = run(tmp_path / "flat", dataset=dataset)
    assert result["status"] == "BLOCKED_BASELINE_SCALE_ZERO"
    assert result["fixture_fit_call_count"] == 1
    result, _, _ = run(tmp_path / "fresh")
    with pytest.raises(FileExistsError):
        run(tmp_path / "fresh")
    assert result["model_sha256"]


def test_failure_after_fit_preserves_attempt_counter(tmp_path):
    dataset = fixture()
    dataset_path = tmp_path / "dataset.json"
    checksum = write_dataset(dataset_path, dataset)
    invalid_parent = tmp_path / "not-a-directory"
    invalid_parent.write_text("file")
    result = fit_exploratory_personal_reference(
        dataset_path=dataset_path,
        expected_dataset_sha256=checksum,
        model_output_path=tmp_path / "model.json",
        receipt_output_path=invalid_parent / "receipt.json",
        evidence_mode="CONTRACT_FIXTURE",
    )
    assert result["status"].startswith("BLOCKED")
    assert result["fit_call_count"] == 0 and result["fixture_fit_call_count"] == 1


def test_real_execution_requires_explicit_gate_before_reading(tmp_path):
    dataset = fixture()
    path = tmp_path / "dataset.json"
    checksum = write_dataset(path, dataset)
    result = fit_exploratory_personal_reference(
        dataset_path=path,
        expected_dataset_sha256=checksum,
        model_output_path=tmp_path / "model.json",
        receipt_output_path=tmp_path / "receipt.json",
        evidence_mode="REAL_OBSERVED",
    )
    assert result["status"] == "BLOCKED_REAL_FIT_NOT_AUTHORIZED"
    assert result["fit_call_count"] == 0
