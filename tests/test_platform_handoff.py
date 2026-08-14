from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np
import onnxruntime as ort
import pytest

import multisensor_ml.platform_contract as platform_contract
from multisensor_ml.platform_contract import (
    SENSOR_VARIANTS,
    baseline_schema,
    bigquery_training_contract,
    canonical_json,
    feature_schema_for_variant,
    label_review_schema,
    source_schema,
    validate_model_training_handoff,
)
from multisensor_ml.platform_handoff import (
    build_comparison_manifest,
    build_evaluation_manifest,
    export_contract_fixture_handoff,
)


def _assert_hash_closed(payload: dict[str, object]) -> None:
    body = {key: value for key, value in payload.items() if key not in {"sha256", "canonical_json"}}
    assert payload["canonical_json"] == canonical_json(body)
    assert payload["sha256"] == hashlib.sha256(
        str(payload["canonical_json"]).encode("utf-8")
    ).hexdigest()
    UUID(str(payload["schema_uuid"]))


@pytest.mark.parametrize("source_name", ["galaxy_watch8", "polar_h10"])
def test_source_schema_is_uuid_and_hash_closed(source_name: str) -> None:
    payload = source_schema(source_name)
    _assert_hash_closed(payload)
    fields = {str(item["name"]) for item in payload["fields"]}
    assert {"corrected_utc", "source_uuid", "capture_session_uuid"}.issubset(fields)
    assert payload["cross_device_waveform_alignment"] == "FORBIDDEN_UNTIL_SEPARATE_CONTRACT"
    if source_name == "polar_h10":
        assert {
            "payload_schema_version",
            "device_profile",
            "ecg_samples",
            "hr_observations",
            "expected_sample_count",
            "observed_sample_count",
            "gap_count",
            "timestamp_regression_count",
            "is_sufficient",
        }.issubset(fields)
        assert not any("acc" in name for name in fields)


def test_label_review_schema_keeps_temporal_stage_and_review_denominators() -> None:
    payload = label_review_schema()
    _assert_hash_closed(payload)
    assert payload["stage_semantics"] == "TEMPORAL_NOT_SEVERITY"
    assert payload["fpr_denominator"] == ["valid_non_event", "hard_negative"]
    assert payload["fnr_denominator"] == ["target_event"]
    assert "novel_pattern" in payload["excluded_from_error_denominators"]


def test_bigquery_contract_is_frozen_uuid_only_and_has_no_live_claim() -> None:
    payload = bigquery_training_contract()
    text = json.dumps(payload, sort_keys=True)
    assert payload["lookup_parameters"] == ["training_cohort_uuid", "cohort_digest"]
    assert payload["live_auth_status"] == "NOT VERIFIED"
    assert "authorized_or_materialized_training_view" in payload["required_views"]
    assert "account_uuid" not in text
    assert "membership_uuid" not in text
    assert "person_key" not in text
    assert "locked_holdout" in payload["split_policy"]
    assert "split_role IN UNNEST(@selection_roles)" in payload["selection_query_contract"]
    assert payload["selection_roles"] == ["train", "validation"]
    assert payload["locked_holdout"]["notebook_access"] == "FORBIDDEN"
    assert payload["locked_holdout"]["evaluator"] == "SEALED_EVALUATOR"
    assert payload["split_policy"]["preregistered_strategies"] == [
        "person_group",
        "chronological_per_subject",
    ]
    assert payload["sequence_group_key"] == "training_capture_set_uuid"
    assert "training_capture_set_uuid" in payload["view_required_columns"]
    assert "source_row_digest" in payload["view_required_columns"]
    assert "review_disposition" in payload["view_required_columns"]
    assert payload["handoff_digest_policy"] == "RECOMPUTE_AND_REJECT_MISMATCH"
    assert payload["view_required_columns"] == [
        *platform_contract.AUTHORIZED_ENVELOPE_FIELDS,
        *platform_contract.AUTHORIZED_ROW_FIELDS,
    ]
    assert payload["reviewed_event_anchor_status"] == "BLOCKED_PENDING_CONTRACT"
    assert payload["blocked_metrics"] == ["event_delay", "forecast_lead_time"]
    assert payload["training_ready_variants"] == ["watch_only"]
    assert payload["non_training_ready_variants"] == {
        "h10_only": "PROPOSED_NOT_IMPLEMENTED",
        "watch_h10": "PROPOSED_NOT_APPROVED",
    }
    assert "source_uuid" not in payload["view_required_columns"]
    assert "capture_session_uuid" not in payload["view_required_columns"]
    assert "artifact_uri" not in payload["view_required_columns"]


def _seal_handoff(handoff: dict[str, object]) -> dict[str, object]:
    handoff.pop("handoff_digest", None)
    handoff["handoff_digest"] = hashlib.sha256(
        json.dumps(
            handoff,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return handoff


def _public_digests(handoff: dict[str, object]) -> tuple[str, str]:
    members = sorted(
        [
            [
                row["training_subject_uuid"],
                row["training_capture_set_uuid"],
                row["exact_window_id"],
                row["split_role"],
                row["window_start_ms"],
                row["source_row_digest"],
            ]
            for row in handoff["rows"]
        ],
        key=lambda member: (member[0], member[4], member[2]),
    )
    split_material = [[member[0], member[2], member[3]] for member in members]
    cohort_material = [
        handoff["feature_schema_uuid"],
        handoff["feature_schema_hash"],
        handoff["split_policy"],
        handoff["purge_seconds"],
        handoff["truth_state"],
        members,
    ]
    return (
        hashlib.sha256(canonical_json(cohort_material).encode("utf-8")).hexdigest(),
        hashlib.sha256(canonical_json(split_material).encode("utf-8")).hexdigest(),
    )


def _cloud_handoff() -> dict[str, object]:
    schema = feature_schema_for_variant("watch_only")
    names = list(schema["ordered_feature_names"])
    features = {name: float(index + 1) for index, name in enumerate(names)}
    handoff: dict[str, object] = {
        "contract_version": 1,
        "training_cohort_uuid": "10000000-0000-4000-8000-000000000001",
        "cohort_digest": "a" * 64,
        "split_digest": "b" * 64,
        "feature_schema_uuid": schema["schema_uuid"],
        "feature_schema_hash": (
            "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"
        ),
        "split_policy": "CHRONOLOGICAL_PER_SUBJECT",
        "purge_seconds": 1800,
        "sequence_group_key": "training_capture_set_uuid",
        "locked_access": False,
        "truth_state": "REVIEWED_REAL",
        "rows": [
            {
                "training_subject_uuid": "20000000-0000-4000-8000-000000000001",
                "training_capture_set_uuid": "30000000-0000-4000-8000-000000000001",
                "exact_window_id": "c" * 64,
                "source_set": ["watch"],
                "window_start_ms": 1_000,
                "window_end_ms": 2_000,
                "feature_schema_uuid": schema["schema_uuid"],
                "feature_schema_hash": (
                    "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"
                ),
                "feature_values": features,
                "label_uuid": "40000000-0000-4000-8000-000000000001",
                "label_revision_uuid": "50000000-0000-4000-8000-000000000001",
                "review_uuid": "60000000-0000-4000-8000-000000000001",
                "review_disposition": "TARGET_EVENT",
                "evaluation_class": "POSITIVE",
                "temporal_stage": 1,
                "observation_code": "EARLY_PATTERN",
                "split_role": "TRAIN",
                "source_row_digest": "d" * 64,
            },
            {
                "training_subject_uuid": "20000000-0000-4000-8000-000000000001",
                "training_capture_set_uuid": "30000000-0000-4000-8000-000000000001",
                "exact_window_id": "e" * 64,
                "source_set": ["watch"],
                "window_start_ms": 4_000_000,
                "window_end_ms": 4_001_000,
                "feature_schema_uuid": schema["schema_uuid"],
                "feature_schema_hash": (
                    "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"
                ),
                "feature_values": features,
                "label_uuid": "40000000-0000-4000-8000-000000000002",
                "label_revision_uuid": "50000000-0000-4000-8000-000000000002",
                "review_uuid": "60000000-0000-4000-8000-000000000002",
                "review_disposition": "VALID_NON_EVENT",
                "evaluation_class": "NEGATIVE",
                "temporal_stage": 5,
                "observation_code": "NO_EVENT",
                "split_role": "VALIDATION",
                "source_row_digest": "f" * 64,
            },
        ],
    }
    public_cohort_digest, public_split_digest = _public_digests(handoff)
    handoff["public_cohort_digest"] = public_cohort_digest
    handoff["public_split_digest"] = public_split_digest
    return _seal_handoff(handoff)


def _authorized_rows() -> list[dict[str, object]]:
    handoff = _cloud_handoff()
    envelope_fields = (
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
    envelope = {field: handoff[field] for field in envelope_fields}
    return [{**envelope, **row} for row in handoff["rows"]]


class _BigQueryRowLike:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def items(self):
        return self._values.items()


def test_model_handoff_accepts_cohort_scoped_capture_sequence_group() -> None:
    result = validate_model_training_handoff(_cloud_handoff())

    assert result["status"] == "VALID_SELECTION_HANDOFF"
    assert result["sensor_variant"] == "watch_only"
    assert result["split_counts"] == {"TRAIN": 1, "VALIDATION": 1}
    assert result["sequence_group_count"] == 1
    assert result["negative_stage_policy"] == "IGNORE_TEMPORAL_STAGE_USE_NO_EVENT"


def test_model_handoff_recomputes_and_rejects_mismatched_digest() -> None:
    handoff = _cloud_handoff()
    handoff["split_digest"] = "9" * 64

    with pytest.raises(ValueError, match="handoff_digest"):
        validate_model_training_handoff(handoff)


def test_model_handoff_rejects_platform_description_hash_as_runtime_hash() -> None:
    handoff = _cloud_handoff()
    description_hash = feature_schema_for_variant("watch_only")["sha256"]
    handoff["feature_schema_hash"] = description_hash
    for row in handoff["rows"]:
        row["feature_schema_hash"] = description_hash
    _seal_handoff(handoff)

    with pytest.raises(ValueError, match=r"runtime|canonical"):
        validate_model_training_handoff(handoff)


@pytest.mark.parametrize("variant", ["h10_only", "watch_h10"])
def test_model_handoff_rejects_non_runtime_implemented_variants(variant: str) -> None:
    handoff = _cloud_handoff()
    runtime_schema = platform_contract.runtime_feature_schema_for_variant(variant)
    runtime = {
        "h10_only": {
            "source_set": ["h10"],
        },
        "watch_h10": {
            "source_set": ["h10", "watch"],
        },
    }[variant]
    features = {
        name: float(index + 1)
        for index, name in enumerate(runtime_schema["ordered_feature_names"])
    }
    handoff["feature_schema_uuid"] = runtime_schema["schema_uuid"]
    handoff["feature_schema_hash"] = runtime_schema["schema_hash"]
    for row in handoff["rows"]:
        row["feature_schema_uuid"] = runtime_schema["schema_uuid"]
        row["feature_schema_hash"] = runtime_schema["schema_hash"]
        row["source_set"] = runtime["source_set"]
        row["feature_values"] = features
    _seal_handoff(handoff)

    with pytest.raises(ValueError, match="not training ready"):
        validate_model_training_handoff(handoff)


def test_authorized_rows_build_one_order_independent_handoff() -> None:
    rows = _authorized_rows()

    from_dicts = platform_contract.synchronize_authorized_training_rows(
        tuple(reversed(rows))
    )
    from_row_like = platform_contract.synchronize_authorized_training_rows(
        tuple(_BigQueryRowLike(row) for row in rows)
    )

    assert from_dicts == _cloud_handoff()
    assert from_row_like == from_dicts
    assert validate_model_training_handoff(from_dicts)["status"] == (
        "VALID_SELECTION_HANDOFF"
    )
    assert from_dicts["public_cohort_digest"] == _cloud_handoff()[
        "public_cohort_digest"
    ]
    assert from_dicts["public_split_digest"] == _cloud_handoff()[
        "public_split_digest"
    ]


def test_authorized_rows_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        platform_contract.synchronize_authorized_training_rows(())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("mixed_envelope", "mixed authorized handoff envelope"),
        ("locked", "LOCKED or unknown"),
        ("missing_validation", "train and validation"),
        ("duplicate_window", "duplicate exact_window_id"),
        ("duplicate_source", "duplicate source_row_digest"),
        ("null_public_digest", "public_cohort_digest"),
        ("wrong_public_digest", "public_cohort_digest does not match"),
        ("private_field", "unknown or private authorized field"),
    ],
)
def test_authorized_rows_fail_closed(
    mutation: str,
    message: str,
) -> None:
    rows = copy.deepcopy(_authorized_rows())
    if mutation == "mixed_envelope":
        rows[1]["cohort_digest"] = "9" * 64
    elif mutation == "locked":
        rows[1]["split_role"] = "LOCKED"
    elif mutation == "missing_validation":
        rows = rows[:1]
    elif mutation == "duplicate_window":
        rows[1]["exact_window_id"] = rows[0]["exact_window_id"]
    elif mutation == "duplicate_source":
        rows[1]["source_row_digest"] = rows[0]["source_row_digest"]
    elif mutation == "null_public_digest":
        rows[0]["public_cohort_digest"] = None
        rows[1]["public_cohort_digest"] = None
    elif mutation == "wrong_public_digest":
        rows[0]["public_cohort_digest"] = "9" * 64
        rows[1]["public_cohort_digest"] = "9" * 64
    elif mutation == "private_field":
        rows[0]["person_uuid"] = "private"
    else:  # pragma: no cover - the parameter table is closed above
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        platform_contract.synchronize_authorized_training_rows(rows)


@pytest.mark.parametrize(
    "field",
    ["person_uuid", "source_uuid", "capture_session_uuid", "account_uuid", "membership_uuid"],
)
def test_model_handoff_rejects_direct_identity_fields(field: str) -> None:
    handoff = _cloud_handoff()
    handoff["rows"][0][field] = "forbidden"
    _seal_handoff(handoff)
    with pytest.raises(ValueError, match="forbidden model-view field"):
        validate_model_training_handoff(handoff)


def test_model_handoff_rejects_locked_rows_and_cross_subject_capture_groups() -> None:
    locked = _cloud_handoff()
    locked["rows"][0]["split_role"] = "LOCKED"
    _seal_handoff(locked)
    with pytest.raises(ValueError, match="LOCKED"):
        validate_model_training_handoff(locked)

    crossed = _cloud_handoff()
    crossed["rows"][1]["training_subject_uuid"] = (
        "20000000-0000-4000-8000-000000000002"
    )
    _seal_handoff(crossed)
    with pytest.raises(ValueError, match="capture set spans subjects"):
        validate_model_training_handoff(crossed)


def test_model_handoff_requires_purge_and_exact_flat_numeric_features() -> None:
    no_purge = _cloud_handoff()
    no_purge.pop("purge_seconds")
    _seal_handoff(no_purge)
    with pytest.raises(ValueError, match="purge_seconds"):
        validate_model_training_handoff(no_purge)

    nested = _cloud_handoff()
    nested["rows"][0]["feature_values"]["watch_eda_z"] = [1.0]
    _seal_handoff(nested)
    with pytest.raises(ValueError, match="flat finite numeric"):
        validate_model_training_handoff(nested)


@pytest.mark.parametrize(
    ("review_disposition", "evaluation_class", "observation_code"),
    [
        ("TARGET_EVENT", "NEGATIVE", "NO_EVENT"),
        ("HARD_NEGATIVE", "POSITIVE", "EARLY_PATTERN"),
        ("VALID_NON_EVENT", "NEGATIVE", "BASELINE_NON_EVENT"),
    ],
)
def test_model_handoff_rejects_invalid_review_truth_pair(
    review_disposition: str,
    evaluation_class: str,
    observation_code: str,
) -> None:
    handoff = _cloud_handoff()
    row = handoff["rows"][1]
    row["review_disposition"] = review_disposition
    row["evaluation_class"] = evaluation_class
    row["observation_code"] = observation_code
    _seal_handoff(handoff)

    with pytest.raises(ValueError, match=r"review|NO_EVENT"):
        validate_model_training_handoff(handoff)


def test_baseline_contract_is_hash_closed_and_fail_closed() -> None:
    payload = baseline_schema()
    _assert_hash_closed(payload)
    assert payload["minimum_eligible_seconds"] == 900
    assert payload["maximum_wall_seconds"] == 1000
    assert payload["decision_warmup_eligible_seconds"] == 1800
    assert payload["scale_fallback"] == ["MAD", "IQR/1.349", "BASELINE_SCALE_ZERO"]
    assert payload["quality_policy"] == "CONSERVATIVE_AND_FAIL_CLOSED"


def test_evaluation_manifest_does_not_invent_cases_without_independent_truth() -> None:
    manifest = build_evaluation_manifest(
        evaluation_uuid="30000000-0000-4000-8000-000000000001",
        model_release_uuid="40000000-0000-4000-8000-000000000001",
        training_cohort_uuid="10000000-0000-4000-8000-000000000001",
        cohort_digest="a" * 64,
        sensor_variant="watch_only",
        evaluated_cases=[],
        locked_holdout_read=False,
    )
    assert manifest["status"] == "NOT EVALUABLE"
    assert manifest["real_accuracy_status"] == "NOT VERIFIED"
    assert manifest["fp_cases"] == []
    assert manifest["fn_cases"] == []
    assert manifest["required_next_evidence"][0] == "independent_target_event_truth"


def test_evaluation_manifest_requires_case_lineage() -> None:
    with pytest.raises(ValueError, match="label_revision_uuid"):
        build_evaluation_manifest(
            evaluation_uuid="30000000-0000-4000-8000-000000000001",
            model_release_uuid="40000000-0000-4000-8000-000000000001",
            training_cohort_uuid="10000000-0000-4000-8000-000000000001",
            cohort_digest="a" * 64,
            sensor_variant="watch_only",
            evaluated_cases=[
                {
                    "training_subject_uuid": "20000000-0000-4000-8000-000000000001",
                    "source_set": ["galaxy_watch8"],
                    "exact_window_id": "b" * 64,
                    "event_time_utc": "2026-08-12T00:00:00Z",
                    "review_uuid": "50000000-0000-4000-8000-000000000001",
                    "prediction": 0.9,
                    "threshold": 0.5,
                    "error_type": "FP",
                    "quality_status": "DECISIONABLE",
                    "artifact_uri": "gs://fixture/evidence.parquet",
                    "artifact_sha256": "c" * 64,
                }
            ],
            locked_holdout_read=False,
        )


def test_locked_holdout_row_level_cases_are_not_returned_to_training() -> None:
    case = {
        "training_subject_uuid": "20000000-0000-4000-8000-000000000001",
        "source_set": ["galaxy_watch8"],
        "exact_window_id": "b" * 64,
        "event_time_utc": "2026-08-12T00:00:00Z",
        "label_revision_uuid": "70000000-0000-4000-8000-000000000001",
        "review_uuid": "50000000-0000-4000-8000-000000000001",
        "prediction": 0.9,
        "threshold": 0.5,
        "error_type": "FP",
        "quality_status": "DECISIONABLE",
        "artifact_uri": "gs://fixture/evidence.parquet",
        "artifact_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="holdout retirement"):
        build_evaluation_manifest(
            evaluation_uuid="30000000-0000-4000-8000-000000000001",
            model_release_uuid="40000000-0000-4000-8000-000000000001",
            training_cohort_uuid="10000000-0000-4000-8000-000000000001",
            cohort_digest="a" * 64,
            sensor_variant="watch_only",
            evaluated_cases=[case],
            locked_holdout_read=True,
        )

    operational = build_evaluation_manifest(
        evaluation_uuid="30000000-0000-4000-8000-000000000001",
        model_release_uuid="40000000-0000-4000-8000-000000000001",
        training_cohort_uuid="10000000-0000-4000-8000-000000000001",
        cohort_digest="a" * 64,
        sensor_variant="watch_only",
        evaluated_cases=[case],
        locked_holdout_read=False,
    )
    assert operational["error_slice_scope"] == "OPERATIONAL_ADJUDICATED"
    assert operational["locked_holdout_row_level_return"] == "FORBIDDEN_UNTIL_RETIREMENT"


def test_comparison_requires_same_frozen_evaluation_input() -> None:
    manifest = build_comparison_manifest(
        comparison_uuid="60000000-0000-4000-8000-000000000001",
        training_cohort_uuid="10000000-0000-4000-8000-000000000001",
        cohort_digest="a" * 64,
        evaluation_input_digest="b" * 64,
        active_model_release_uuid="40000000-0000-4000-8000-000000000001",
        candidate_model_release_uuid="40000000-0000-4000-8000-000000000002",
        active_evaluation_input_digest="b" * 64,
        candidate_evaluation_input_digest="b" * 64,
        reviewed_metric_rows=[],
    )
    assert manifest["status"] == "NOT EVALUABLE"
    assert manifest["same_window_comparison"] is True
    assert manifest["promotion_eligible"] is False

    with pytest.raises(ValueError, match="same frozen evaluation input"):
        build_comparison_manifest(
            comparison_uuid="60000000-0000-4000-8000-000000000001",
            training_cohort_uuid="10000000-0000-4000-8000-000000000001",
            cohort_digest="a" * 64,
            evaluation_input_digest="b" * 64,
            active_model_release_uuid="40000000-0000-4000-8000-000000000001",
            candidate_model_release_uuid="40000000-0000-4000-8000-000000000002",
            active_evaluation_input_digest="b" * 64,
            candidate_evaluation_input_digest="c" * 64,
            reviewed_metric_rows=[],
        )


def test_contract_fixture_handoff_exports_three_non_deployable_reference_bundles(
    tmp_path: Path,
) -> None:
    result = export_contract_fixture_handoff(tmp_path / "handoff")
    root_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert root_manifest["evidence_scope"] == "LOCAL_CONTRACT_FIXTURE"
    assert root_manifest["live_bigquery_status"] == "NOT VERIFIED"
    assert root_manifest["deployment_eligible"] is False
    assert root_manifest["sensor_variants"] == list(SENSOR_VARIANTS)
    assert root_manifest["candidate_adapter_status"] == "BLOCKED_NO_FROZEN_REAL_COHORT"
    assert root_manifest["training_ready_variants"] == ["watch_only"]
    assert root_manifest["non_training_ready_variants"] == {
        "h10_only": "PROPOSED_NOT_IMPLEMENTED",
        "watch_h10": "PROPOSED_NOT_APPROVED",
    }

    for variant in SENSOR_VARIANTS:
        bundle = result.reference_bundles[variant]
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        assert manifest["evidence_scope"] == "LOCAL_CONTRACT_FIXTURE"
        assert manifest["delivery_eligible"] is False
        if variant == "watch_h10":
            assert manifest["serving_contract_status"] == "PROPOSED_NOT_APPROVED"
            assert manifest["reference_status"] == "PROPOSED_NOT_APPROVED"
            assert manifest["training_ready"] is False
        elif variant == "h10_only":
            assert manifest["training_status"] == "PROPOSED_NOT_IMPLEMENTED"
            assert manifest["training_ready"] is False
        else:
            assert manifest["feature_schema_hash"] == (
                "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"
            )
            assert manifest["training_status"] == "CANONICAL_RUNTIME"
        session = ort.InferenceSession(
            bundle.model_path.read_bytes(), providers=["CPUExecutionProvider"]
        )
        count = len(manifest["feature_names"])
        row = np.arange(count, dtype=np.float32)[None, :]
        actual = session.run(["prediction"], {"features": row})[0]
        expected = row[:, [manifest["reference_feature_index"]]]
        np.testing.assert_array_equal(actual, expected)

    checksums = json.loads(result.checksums_path.read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        path = result.root / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected

    second = export_contract_fixture_handoff(tmp_path / "handoff-second")
    assert second.checksums_path.read_bytes() == result.checksums_path.read_bytes()
