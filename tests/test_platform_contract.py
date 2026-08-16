from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

import multisensor_ml.platform_contract as platform_contract
from multisensor_ml.platform_bundle import export_simple_reference_bundle
from multisensor_ml.platform_contract import (
    PLATFORM_CONTRACT_VERSION,
    REVIEW_DISPOSITIONS,
    SENSOR_VARIANTS,
    canonical_platform_contract,
    compute_cohort_digest,
    compute_exact_window_id,
    evaluate_error_case,
    feature_schema_for_variant,
    validate_frozen_cohort_manifest,
)


def _uuid(value: int) -> str:
    return str(UUID(int=value))


def _cohort_manifest() -> dict[str, object]:
    windows = [
        {
            "training_subject_uuid": _uuid(11),
            "source_uuid": _uuid(21),
            "capture_session_uuid": _uuid(31),
            "exact_window_id": hashlib.sha256(b"window-a").hexdigest(),
            "split_role": "train",
            "window_start_utc": "2026-08-01T00:00:00Z",
            "window_end_utc": "2026-08-01T00:30:00Z",
        },
        {
            "training_subject_uuid": _uuid(12),
            "source_uuid": _uuid(22),
            "capture_session_uuid": _uuid(32),
            "exact_window_id": hashlib.sha256(b"window-b").hexdigest(),
            "split_role": "validation",
            "window_start_utc": "2026-08-02T00:00:00Z",
            "window_end_utc": "2026-08-02T00:30:00Z",
        },
        {
            "training_subject_uuid": _uuid(13),
            "source_uuid": _uuid(23),
            "capture_session_uuid": _uuid(33),
            "exact_window_id": hashlib.sha256(b"window-c").hexdigest(),
            "split_role": "locked_holdout",
            "window_start_utc": "2026-08-03T00:00:00Z",
            "window_end_utc": "2026-08-03T00:30:00Z",
        },
    ]
    manifest: dict[str, object] = {
        "contract_version": PLATFORM_CONTRACT_VERSION,
        "training_cohort_uuid": _uuid(1),
        "training_view": "kidsignal_ml.authorized_training_cohort_v1",
        "materialization_uri": "gs://kidsignal-ml-cohorts/cohort-0001/manifest.parquet",
        "feature_set_uuid": _uuid(2),
        "feature_schema_uuid": "9b842d8c-8889-5259-acca-77baa0c7729d",
        "feature_schema_hash": (
            "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"
        ),
        "sensor_variant": "watch_only",
        "split_strategy": "person_group",
        "purge_seconds": 1800,
        "locked_holdout_read": False,
        "windows": windows,
    }
    manifest["cohort_digest"] = compute_cohort_digest(manifest)
    return manifest


def test_platform_contract_has_three_separate_sensor_variants_and_temporal_stage_truth() -> None:
    contract = canonical_platform_contract()

    assert contract["contract_version"] == PLATFORM_CONTRACT_VERSION
    assert tuple(contract["sensor_variants"]) == SENSOR_VARIANTS
    assert contract["stage_contract"]["ordered_truth"] == {
        "1": ["pre_early"],
        "2": ["pre_late", "onset"],
        "3": ["peak"],
        "4": ["recovery_early"],
        "5": ["recovery_late", "post"],
    }
    assert contract["stage_contract"]["stage_is_severity"] is False
    assert tuple(contract["review_dispositions"]) == REVIEW_DISPOSITIONS
    assert "account_uuid" not in json.dumps(contract)
    assert "membership_uuid" not in json.dumps(contract)
    composition = contract["exact_composition"]
    assert composition["implemented"] == [
        "source_manifest_scheduling_intersection",
        "corrected_time_provenance",
    ]
    assert "cross_device_waveform_alignment" in composition["forbidden_until_separate_contract"]
    assert contract["locked_holdout"]["notebook_row_or_label_access"] is False
    assert contract["locked_holdout"]["evaluator"] == "SEALED_EVALUATOR"


def test_watch_exact_composition_v3_locks_stream_roles_and_timing() -> None:
    contract = platform_contract.watch_exact_composition_v3_contract()

    assert contract["contract_version"] == "kidsignal-watch-exact-composition/v3"
    assert contract["logical_source"] == "watch"
    assert contract["core_streams"] == ["hr_ibi", "eda", "accelerometer"]
    assert contract["optional_streams"] == ["ppg", "skin_temperature"]
    assert contract["hr_ibi_policy"] == {
        "hr_coverage_required": True,
        "ibi_lineage_preserved": True,
        "ibi_runtime_feature": False,
        "ibi_trainer_feature": False,
        "ibi_separate_core_value_gate": False,
    }
    assert contract["interval"] == {
        "duration_ms": 10_000,
        "alignment": "FLOOR_CORRECTED_UTC",
        "non_overlapping": True,
    }
    assert contract["deadlines"] == {
        "ready_watermark_after_end_ms": 5_000,
        "terminal_reject_after_end_ms": 30_000,
    }
    assert contract["forbidden_transforms"] == [
        "interpolation",
        "resampling",
        "padding",
        "numeric_zero_substitution",
        "null_as_normal",
        "hold_forward",
    ]
    assert contract["non_v3_sources"] == {
        "h10": "SEPARATE_EXISTING_V1_NOT_PROMOTED",
        "muse_s": "NOT_RUNTIME_TRAINING_READY",
    }
    assert platform_contract.validate_watch_exact_composition_v3_contract(contract)[
        "status"
    ] == "VALID"
    assert platform_contract.watch_exact_composition_v3_interval(27_345) == {
        "interval_start_ms": 20_000,
        "interval_end_ms": 30_000,
        "ready_watermark_ms": 35_000,
        "terminal_reject_deadline_ms": 60_000,
    }
    watch_source = platform_contract.source_schema("galaxy_watch8")
    assert "exact_composition_v3" not in watch_source
    assert watch_source["sha256"] == (
        "95bea255e39fc7a46403169042b6ff8a64e100398b345bfa2f8e0bb6309e49d8"
    )
    assert hashlib.sha256(
        str(watch_source["canonical_json"]).encode("utf-8")
    ).hexdigest() == watch_source["sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        "stream_role",
        "ibi_feature",
        "ibi_trainer_feature",
        "interval",
        "ready_watermark",
        "terminal_deadline",
        "fill_policy",
    ],
)
def test_watch_exact_composition_v3_rejects_contract_drift(mutation: str) -> None:
    contract = json.loads(
        json.dumps(platform_contract.watch_exact_composition_v3_contract())
    )
    if mutation == "stream_role":
        contract["core_streams"] = ["hr", "eda", "accelerometer"]
    elif mutation == "ibi_feature":
        contract["hr_ibi_policy"]["ibi_runtime_feature"] = True
    elif mutation == "ibi_trainer_feature":
        contract["hr_ibi_policy"]["ibi_trainer_feature"] = True
    elif mutation == "interval":
        contract["interval"]["duration_ms"] = 9_999
    elif mutation == "ready_watermark":
        contract["deadlines"]["ready_watermark_after_end_ms"] = 4_999
    elif mutation == "terminal_deadline":
        contract["deadlines"]["terminal_reject_after_end_ms"] = 29_999
    elif mutation == "fill_policy":
        contract["forbidden_transforms"].remove("hold_forward")
    else:  # pragma: no cover - fixed parametrization
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="exact-composition v3 contract drift"):
        platform_contract.validate_watch_exact_composition_v3_contract(contract)


def test_watch_v3_keeps_learning_planes_and_target_leakage_gates_separate() -> None:
    from multisensor_ml.bigquery_training_preflight import (
        AUTHORIZED_TRAINING_VIEW,
        EXPECTED_VIEW_FIELDS,
    )
    from multisensor_ml.observational_contract import FEATURE_NAMES
    from multisensor_ml.standard_baseline_bigquery import (
        RUNTIME_TO_TRAINER_INDICES,
        STANDARD_BASELINE_AUTHORIZED_VIEW,
        STANDARD_BASELINE_VIEW_FIELDS,
        STANDARD_LEAKAGE_PROBE_CASE_IDS,
        STANDARD_TRAINER_FEATURE_NAMES,
        TARGET_SOURCE_FEATURE,
        project_standard_runtime_features,
    )

    assert len(STANDARD_BASELINE_VIEW_FIELDS) == 24
    assert len(EXPECTED_VIEW_FIELDS) == 26
    assert STANDARD_BASELINE_AUTHORIZED_VIEW.endswith(
        "standard_baseline_train_validation_v1"
    )
    assert AUTHORIZED_TRAINING_VIEW.endswith("training_examples_train_validation_v1")
    assert STANDARD_BASELINE_AUTHORIZED_VIEW != AUTHORIZED_TRAINING_VIEW
    assert "label_schema_digest" not in STANDARD_BASELINE_VIEW_FIELDS
    assert "label_schema_digest" not in EXPECTED_VIEW_FIELDS

    assert len(FEATURE_NAMES) == 16
    assert len(STANDARD_TRAINER_FEATURE_NAMES) == 15
    assert all("ibi" not in name.lower() for name in FEATURE_NAMES)
    assert all("ibi" not in name.lower() for name in STANDARD_TRAINER_FEATURE_NAMES)
    assert TARGET_SOURCE_FEATURE == "watch_load_median_300"
    assert TARGET_SOURCE_FEATURE not in STANDARD_TRAINER_FEATURE_NAMES
    assert RUNTIME_TO_TRAINER_INDICES == (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        13,
        14,
        15,
    )
    assert STANDARD_LEAKAGE_PROBE_CASE_IDS == (
        "target-source-base",
        "target-source-mutated",
    )
    runtime = np.arange(32, dtype=np.float32).reshape(2, 16)
    runtime[1] = runtime[0]
    runtime[1, 11] = 999.0
    projected = project_standard_runtime_features(runtime)
    assert projected.shape == (2, 15)
    assert np.array_equal(projected[0], projected[1])


@pytest.mark.parametrize("variant", SENSOR_VARIANTS)
def test_feature_schema_is_hash_closed_and_requires_only_its_source_set(variant: str) -> None:
    schema = feature_schema_for_variant(variant)
    payload = schema["canonical_json"].encode("utf-8")

    assert hashlib.sha256(payload).hexdigest() == schema["sha256"]
    assert schema["missing_source_policy"] == "NOT_DECISIONABLE"
    assert schema["biological_null_policy"] == "FORBID_ZERO_OR_NULL_IMPUTATION"
    assert schema["ordered_feature_names"]
    if variant == "watch_only":
        assert schema["required_sources"] == ["galaxy_watch8"]
    elif variant == "h10_only":
        assert schema["required_sources"] == ["polar_h10"]
        assert all("motion" not in name for name in schema["ordered_feature_names"])
    else:
        assert schema["required_sources"] == ["galaxy_watch8", "polar_h10"]
        assert schema["serving_contract_status"] == "PROPOSED_NOT_APPROVED"
        definitions = {item["name"]: item for item in schema["features"]}
        assert definitions["fused_load_raw"]["status"] == "PROPOSED_NOT_VERIFIED"
        assert definitions["source_agreement_60"]["status"] == "PROPOSED_NOT_VERIFIED"


def test_runtime_schema_identity_is_separate_from_platform_description_hash() -> None:
    description = feature_schema_for_variant("watch_only")
    runtime = platform_contract.runtime_feature_schema_for_variant("watch_only")

    assert description["sha256"] == (
        "d8314b0c329426d4b33e099a9060a3250b3954a74f4f3fd032d09ffbe3e2e43b"
    )
    assert runtime["schema_uuid"] == "9b842d8c-8889-5259-acca-77baa0c7729d"
    assert runtime["schema_version"] == "watch_standard_16_v2"
    assert runtime["schema_hash"] == (
        "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"
    )
    assert runtime["training_status"] == "CANONICAL_RUNTIME"
    assert runtime["schema_hash"] != description["sha256"]


@pytest.mark.parametrize(
    ("variant", "expected_status"),
    [
        ("h10_only", "PROPOSED_NOT_IMPLEMENTED"),
        ("watch_h10", "PROPOSED_NOT_APPROVED"),
    ],
)
def test_non_watch_runtime_schemas_are_not_training_ready(
    variant: str,
    expected_status: str,
) -> None:
    runtime = platform_contract.runtime_feature_schema_for_variant(variant)

    assert runtime["training_status"] == expected_status
    assert runtime["training_ready"] is False


def test_exact_window_identity_is_content_based_and_deterministic() -> None:
    first = compute_exact_window_id(b"same corrected-time payload")
    second = compute_exact_window_id(b"same corrected-time payload")

    assert first == second == hashlib.sha256(b"same corrected-time payload").hexdigest()
    assert compute_exact_window_id(b"different") != first


def test_frozen_cohort_rejects_legacy_or_account_identity_and_digest_changes() -> None:
    manifest = _cohort_manifest()
    assert validate_frozen_cohort_manifest(manifest)["status"] == "VALID"

    for forbidden in ("account_uuid", "membership_uuid", "person_key"):
        invalid = dict(manifest)
        invalid[forbidden] = "legacy"
        with pytest.raises(ValueError, match=forbidden):
            validate_frozen_cohort_manifest(invalid)

    changed = json.loads(json.dumps(manifest))
    changed["windows"][0]["exact_window_id"] = hashlib.sha256(b"changed").hexdigest()
    with pytest.raises(ValueError, match="cohort_digest"):
        validate_frozen_cohort_manifest(changed)

    wrong_schema_uuid = json.loads(json.dumps(manifest))
    wrong_schema_uuid["feature_schema_uuid"] = _uuid(3)
    wrong_schema_uuid["cohort_digest"] = compute_cohort_digest(wrong_schema_uuid)
    with pytest.raises(ValueError, match="feature_schema_uuid"):
        validate_frozen_cohort_manifest(wrong_schema_uuid)


def test_frozen_cohort_keeps_locked_holdout_out_of_selection() -> None:
    manifest = _cohort_manifest()
    manifest["locked_holdout_read"] = True
    manifest["cohort_digest"] = compute_cohort_digest(manifest)

    with pytest.raises(ValueError, match="locked_holdout_read"):
        validate_frozen_cohort_manifest(manifest)


def test_frozen_cohort_allows_purged_chronological_split_for_one_subject() -> None:
    manifest = _cohort_manifest()
    subject = _uuid(11)
    windows = manifest["windows"]
    assert isinstance(windows, list)
    for window in windows:
        window["training_subject_uuid"] = subject
    windows[0]["window_start_utc"] = "2026-08-01T00:00:00Z"
    windows[0]["window_end_utc"] = "2026-08-01T00:30:00Z"
    windows[1]["window_start_utc"] = "2026-08-01T01:00:00Z"
    windows[1]["window_end_utc"] = "2026-08-01T01:30:00Z"
    windows[2]["window_start_utc"] = "2026-08-01T02:00:00Z"
    windows[2]["window_end_utc"] = "2026-08-01T02:30:00Z"
    manifest["split_strategy"] = "chronological_per_subject"
    manifest["cohort_digest"] = compute_cohort_digest(manifest)

    assert validate_frozen_cohort_manifest(manifest)["status"] == "VALID"

    windows[1]["window_start_utc"] = "2026-08-01T00:59:59Z"
    manifest["cohort_digest"] = compute_cohort_digest(manifest)
    with pytest.raises(ValueError, match="purge"):
        validate_frozen_cohort_manifest(manifest)


@pytest.mark.parametrize(
    ("disposition", "truth_positive", "prediction", "expected"),
    [
        ("valid_non_event", False, 0.9, "FP"),
        ("hard_negative", False, 0.9, "FP"),
        ("target_event", True, 0.1, "FN"),
        ("target_event", True, 0.9, None),
        ("novel_pattern", False, 0.9, None),
        ("unreviewed", False, 0.9, None),
        ("artifact_or_quality_failure", False, 0.9, None),
    ],
)
def test_error_cases_only_use_independent_reviewed_truth(
    disposition: str,
    truth_positive: bool,
    prediction: float,
    expected: str | None,
) -> None:
    result = evaluate_error_case(
        disposition=disposition,
        truth_positive=truth_positive,
        prediction=prediction,
        threshold=0.5,
        label_revision_uuid=(
            _uuid(101) if disposition not in {"unreviewed", "novel_pattern"} else None
        ),
        review_uuid=_uuid(102) if disposition != "unreviewed" else None,
    )

    assert result["error_type"] == expected
    assert result["evaluable"] is (expected is not None or disposition == "target_event")
    if disposition in {"novel_pattern", "unreviewed", "artifact_or_quality_failure"}:
        assert result["count_in_fpr_denominator"] is False


def test_reference_bundle_selects_by_training_only_and_exports_hash_closed_onnx(
    tmp_path: Path,
) -> None:
    result = export_simple_reference_bundle(
        output_dir=tmp_path / "watch-reference",
        sensor_variant="watch_only",
        model_release_uuid=_uuid(201),
        training_cohort_uuid=_uuid(202),
        cohort_digest="a" * 64,
        training_mae={"persistence": 0.31, "rolling_median_300": 0.22},
        validation_mae={"persistence": 0.29, "rolling_median_300": 0.24},
        locked_holdout_read=False,
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["reference_method"] == "rolling_median_300"
    assert manifest["selection_roles_read"] == ["train"]
    assert manifest["validation_used_for_selection"] is False
    assert manifest["locked_holdout_read"] is False
    assert manifest["real_data_status"] == "NOT VERIFIED"
    assert manifest["stage"] is None
    runtime_schema = platform_contract.runtime_feature_schema_for_variant("watch_only")
    assert manifest["feature_schema_hash"] == runtime_schema["schema_hash"]
    assert manifest["feature_schema_uuid"] == runtime_schema["schema_uuid"]
    assert manifest["training_status"] == "CANONICAL_RUNTIME"
    assert manifest["training_ready"] is True
    assert result.model_path.suffix == ".onnx"
    assert hashlib.sha256(result.model_path.read_bytes()).hexdigest() == manifest["model_sha256"]

    import onnxruntime as ort

    session = ort.InferenceSession(str(result.model_path), providers=["CPUExecutionProvider"])
    features = np.zeros((1, len(manifest["feature_names"])), dtype="float32")
    index = manifest["reference_feature_index"]
    features[0, index] = 1.25
    prediction = session.run(None, {"features": features})[0]
    assert float(prediction.reshape(-1)[0]) == pytest.approx(1.25)


def test_reference_bundle_refuses_holdout_selection_and_non_uuid_release(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="locked_holdout"):
        export_simple_reference_bundle(
            output_dir=tmp_path / "invalid",
            sensor_variant="h10_only",
            model_release_uuid=_uuid(301),
            training_cohort_uuid=_uuid(302),
            cohort_digest="b" * 64,
            training_mae={"persistence": 0.4, "rolling_median_300": 0.3},
            validation_mae={},
            locked_holdout_read=True,
        )

    with pytest.raises(ValueError, match="model_release_uuid"):
        export_simple_reference_bundle(
            output_dir=tmp_path / "invalid-uuid",
            sensor_variant="watch_h10",
            model_release_uuid="not-a-uuid",
            training_cohort_uuid=_uuid(302),
            cohort_digest="c" * 64,
            training_mae={"persistence": 0.4, "rolling_median_300": 0.3},
            validation_mae={},
            locked_holdout_read=False,
        )
