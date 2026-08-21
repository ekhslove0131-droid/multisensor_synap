from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess
from uuid import NAMESPACE_URL, uuid5

import pytest

from multisensor_ml.bigquery_training_preflight import EXPECTED_VIEW_FIELDS
from multisensor_ml.observational_contract import FEATURE_NAMES
from multisensor_ml.standard_baseline_bigquery import (
    EXPECTED_MODEL_READER,
    STANDARD_BASELINE_AUTHORIZED_VIEW,
    STANDARD_BASELINE_UPSTREAM_COMMIT,
    STANDARD_BASELINE_VIEW_FIELDS,
    build_standard_cohort_readiness_receipt,
    run_read_only_standard_cohort_reader,
    synchronize_standard_baseline_rows,
    validate_standard_baseline_handoff,
)

STANDARD_COHORT_UUID = "00000000-0000-4000-8000-000000000099"
WATCH_SCHEMA_UUID = "9b842d8c-8889-5259-acca-77baa0c7729d"
WATCH_SCHEMA_HASH = "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _attach_public_digests(rows: list[dict[str, object]]) -> None:
    members = sorted(
        [
            [
                row["training_subject_uuid"],
                row["training_capture_set_uuid"],
                row["standard_hour_uuid"],
                row["split_role"],
                row["hour_start_ms"],
                row["source_row_digest"],
            ]
            for row in rows
        ]
    )
    first = rows[0]
    cohort_material = [
        first["stable_standard_version"],
        first["feature_schema_uuid"],
        first["feature_schema_hash"],
        first["split_policy"],
        first["purge_seconds"],
        first["eligibility_policy"],
        members,
    ]
    split_material = [[member[0], member[2], member[3]] for member in members]
    for row in rows:
        row["public_cohort_digest"] = _canonical_sha256(cohort_material)
        row["public_split_digest"] = _canonical_sha256(split_material)


def _rows() -> list[dict[str, object]]:
    subject = str(uuid5(NAMESPACE_URL, "standard-subject"))
    capture = str(uuid5(NAMESPACE_URL, "standard-capture"))
    rows: list[dict[str, object]] = []
    for index, split in enumerate(("TRAIN", "VALIDATION")):
        start = index * 3 * 3_600_000
        target = 0.25 + index * 0.5
        features = {
            name: float(position + 1) / 10.0
            for position, name in enumerate(FEATURE_NAMES)
        }
        features["watch_load_median_300"] = target
        rows.append(
            {
                "standard_cohort_uuid": STANDARD_COHORT_UUID,
                "cohort_digest": "a" * 64,
                "split_digest": "b" * 64,
                "public_cohort_digest": "",
                "public_split_digest": "",
                "stable_standard_version": "stable-stress-standard-v1",
                "feature_schema_uuid": WATCH_SCHEMA_UUID,
                "feature_schema_hash": WATCH_SCHEMA_HASH,
                "split_policy": "CHRONOLOGICAL_PER_SUBJECT",
                "purge_seconds": 1800,
                "eligibility_policy": "ELIGIBLE_NO_PATTERN_REAL",
                "training_subject_uuid": subject,
                "training_capture_set_uuid": capture,
                "standard_hour_uuid": str(
                    uuid5(NAMESPACE_URL, f"standard-hour-{index}")
                ),
                "source_set": ["watch"],
                "hour_start_ms": start,
                "hour_end_ms": start + 3_600_000,
                "feature_values": features,
                "target_name": "no_pattern_median",
                "target_value": target,
                "target_unit": "positive_robust_z",
                "valid_coverage_seconds": 3000,
                "split_role": split,
                "source_row_digest": f"{index + 1:064x}",
            }
        )
    _attach_public_digests(rows)
    return rows


def _expected_public_digests(
    rows: list[dict[str, object]] | None = None,
) -> dict[str, str]:
    values = rows or _rows()
    return {
        "expected_public_cohort_digest": str(values[0]["public_cohort_digest"]),
        "expected_public_split_digest": str(values[0]["public_split_digest"]),
    }


class _RowLike:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def items(self):
        return self._values.items()


def test_standard_baseline_reader_is_separate_from_behavior_reader() -> None:
    assert importlib.util.find_spec(
        "multisensor_ml.standard_baseline_bigquery"
    ) is not None

    assert len(STANDARD_BASELINE_VIEW_FIELDS) == 24
    assert len(EXPECTED_VIEW_FIELDS) == 26
    assert STANDARD_BASELINE_VIEW_FIELDS != EXPECTED_VIEW_FIELDS
    assert STANDARD_BASELINE_AUTHORIZED_VIEW.endswith(
        ".standard_baseline_train_validation_v1"
    )
    assert STANDARD_BASELINE_UPSTREAM_COMMIT == (
        "bce5b0f8c54266e28dde1b7780c2a228bf50622c"
    )


def test_exact_24_field_order_is_required() -> None:
    assert STANDARD_BASELINE_VIEW_FIELDS == (
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
    shuffled = (*STANDARD_BASELINE_VIEW_FIELDS[1:], STANDARD_BASELINE_VIEW_FIELDS[0])
    with pytest.raises(ValueError, match="field order"):
        build_standard_cohort_readiness_receipt(
            rows=(),
            schema_fields=shuffled,
            requested_standard_cohort_uuid=STANDARD_COHORT_UUID,
            **_expected_public_digests(),
            observed_at_utc="2026-08-15T00:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
        )


def test_rows_build_order_independent_hash_closed_standard_handoff() -> None:
    rows = _rows()
    handoff = synchronize_standard_baseline_rows(tuple(reversed(rows)))
    row_like = synchronize_standard_baseline_rows(tuple(_RowLike(row) for row in rows))

    assert handoff == row_like
    assert handoff["public_cohort_digest"] == rows[0]["public_cohort_digest"]
    assert handoff["public_split_digest"] == rows[0]["public_split_digest"]
    assert [row["split_role"] for row in handoff["rows"]] == [
        "TRAIN",
        "VALIDATION",
    ]
    result = validate_standard_baseline_handoff(handoff)
    assert result == {
        "status": "VALID_STANDARD_BASELINE_HANDOFF",
        "training_eligible": True,
        "split_counts": {"TRAIN": 1, "VALIDATION": 1},
        "sequence_group_count": 1,
        "sequence_group_key": "training_capture_set_uuid",
        "target_name": "no_pattern_median",
        "target_unit": "positive_robust_z",
        "locked_access": False,
    }


def test_public_digest_sorts_the_complete_six_element_member_tuple() -> None:
    rows = _rows()
    rows[0]["training_capture_set_uuid"] = (
        "00000000-0000-5000-8000-0000000000ff"
    )
    rows[1]["training_capture_set_uuid"] = (
        "00000000-0000-5000-8000-000000000001"
    )
    _attach_public_digests(rows)

    handoff = synchronize_standard_baseline_rows(tuple(reversed(rows)))

    assert handoff["public_cohort_digest"] == rows[0]["public_cohort_digest"]
    assert handoff["public_split_digest"] == rows[0]["public_split_digest"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_private", "unknown or private"),
        ("missing", "unknown or private"),
        ("locked", "LOCKED or unknown"),
        ("duplicate_hour", "duplicate standard_hour_uuid"),
        ("duplicate_source", "duplicate source_row_digest"),
        ("mixed_cohort", "mixed standard handoff envelope"),
        ("null_public", "public_cohort_digest"),
        ("wrong_public", "public_cohort_digest does not match"),
        ("train_only", "train and validation"),
        ("target_mismatch", "target_value"),
        ("negative_target", "nonnegative"),
        ("nonfinite", "finite numeric"),
        ("bad_eligibility", "eligibility_policy"),
    ],
)
def test_standard_rows_fail_closed(mutation: str, message: str) -> None:
    rows = copy.deepcopy(_rows())
    if mutation == "extra_private":
        rows[0]["person_uuid"] = "private"
    elif mutation == "missing":
        rows[0].pop("target_unit")
    elif mutation == "locked":
        rows[1]["split_role"] = "LOCKED"
    elif mutation == "duplicate_hour":
        rows[1]["standard_hour_uuid"] = rows[0]["standard_hour_uuid"]
    elif mutation == "duplicate_source":
        rows[1]["source_row_digest"] = rows[0]["source_row_digest"]
    elif mutation == "mixed_cohort":
        rows[1]["standard_cohort_uuid"] = str(uuid5(NAMESPACE_URL, "other"))
    elif mutation == "null_public":
        for row in rows:
            row["public_cohort_digest"] = None
    elif mutation == "wrong_public":
        for row in rows:
            row["public_cohort_digest"] = "f" * 64
    elif mutation == "train_only":
        rows = rows[:1]
        _attach_public_digests(rows)
    elif mutation == "target_mismatch":
        rows[0]["target_value"] = 99.0
    elif mutation == "negative_target":
        rows[0]["target_value"] = -0.25
        rows[0]["feature_values"]["watch_load_median_300"] = -0.25
    elif mutation == "nonfinite":
        rows[0]["feature_values"][FEATURE_NAMES[0]] = math.nan
    elif mutation == "bad_eligibility":
        for row in rows:
            row["eligibility_policy"] = "REVIEWED_REAL"
        _attach_public_digests(rows)
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        synchronize_standard_baseline_rows(rows)


def test_zero_real_rows_are_blocked_without_fit() -> None:
    receipt = build_standard_cohort_readiness_receipt(
        rows=(),
        schema_fields=STANDARD_BASELINE_VIEW_FIELDS,
        requested_standard_cohort_uuid=STANDARD_COHORT_UUID,
        **_expected_public_digests(),
        observed_at_utc="2026-08-15T00:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )

    assert receipt["status"] == "BLOCKED_NO_REAL_COHORT"
    assert receipt["training_ready"] is False
    assert receipt["row_count"] == 0
    assert receipt["fit_call_count"] == 0
    assert receipt["training_status"] == "NOT STARTED"
    assert receipt["evaluation_status"] == "NOT EVALUABLE"
    assert receipt["upstream_contract_commit"] == STANDARD_BASELINE_UPSTREAM_COMMIT


def test_valid_rows_produce_read_only_receipt_without_fit() -> None:
    rows = _rows()
    receipt = build_standard_cohort_readiness_receipt(
        rows=rows,
        schema_fields=STANDARD_BASELINE_VIEW_FIELDS,
        requested_standard_cohort_uuid=STANDARD_COHORT_UUID,
        expected_public_cohort_digest=str(rows[0]["public_cohort_digest"]),
        expected_public_split_digest=str(rows[0]["public_split_digest"]),
        observed_at_utc="2026-08-15T00:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )

    assert receipt["status"] == "READY_FOR_SYNC_NOT_TRAINED"
    assert receipt["training_ready"] is True
    assert receipt["split_counts"] == {"TRAIN": 1, "VALIDATION": 1}
    assert receipt["target_name"] == "no_pattern_median"
    assert receipt["target_unit"] == "positive_robust_z"
    assert receipt["fit_call_count"] == 0
    assert receipt["public_digest_policy"] == (
        "EXPECTED_PUBLIC_COHORT_AND_SPLIT_DIGEST_PIN"
    )
    assert receipt["canonical_handoff_digest_policy"] == (
        "RECOMPUTED_AFTER_EXPECTED_PUBLIC_DIGEST_MATCH"
    )
    assert receipt["expected_public_digests_match"] is True


def test_expected_public_digest_mismatch_blocks_standard_dataset_use() -> None:
    rows = _rows()
    receipt = build_standard_cohort_readiness_receipt(
        rows=rows,
        schema_fields=STANDARD_BASELINE_VIEW_FIELDS,
        requested_standard_cohort_uuid=STANDARD_COHORT_UUID,
        expected_public_cohort_digest=str(rows[0]["public_cohort_digest"]),
        expected_public_split_digest="f" * 64,
        observed_at_utc="2026-08-15T00:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )

    assert receipt["status"] == "BLOCKED_EXPECTED_PUBLIC_DIGEST_MISMATCH"
    assert receipt["training_ready"] is False
    assert receipt["fit_call_count"] == 0
    assert receipt["canonical_handoff_digest"] is None
    assert receipt["canonical_handoff_digest_policy"] == (
        "RECOMPUTED_AFTER_EXPECTED_PUBLIC_DIGEST_MATCH"
    )
    assert "rows" not in receipt


def test_validated_standard_rows_prepare_typed_trainer_inputs_without_fit() -> None:
    import multisensor_ml.standard_baseline_bigquery as module

    rows = _rows()
    receipt = build_standard_cohort_readiness_receipt(
        rows=rows,
        schema_fields=STANDARD_BASELINE_VIEW_FIELDS,
        requested_standard_cohort_uuid=STANDARD_COHORT_UUID,
        **_expected_public_digests(rows),
        observed_at_utc="2026-08-15T00:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )
    prepared = module.prepare_standard_dataset_handoff(
        rows=rows,
        readiness_receipt=receipt,
        **_expected_public_digests(rows),
    )

    assert prepared.task == "stable_standard_regression"
    assert prepared.feature_names == tuple(FEATURE_NAMES)
    assert prepared.runtime_train_features.shape == (1, 16)
    assert prepared.runtime_validation_features.shape == (1, 16)
    assert prepared.train_features.shape == (1, 15)
    assert prepared.validation_features.shape == (1, 15)
    assert prepared.train_features.dtype.name == "float32"
    assert prepared.train_targets.dtype.name == "float32"
    assert prepared.train_targets.tolist() == [0.25]
    assert prepared.validation_targets.tolist() == [0.75]
    assert prepared.target_source_feature == "watch_load_median_300"
    assert prepared.target_source_runtime_index == 11
    assert prepared.trainer_feature_names == (
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
        "watch_load_ema_1800",
        "watch_load_ema_21600",
        "quality_confidence",
        "watch_ineligible_fraction_60",
    )
    assert prepared.runtime_to_trainer_indices == (
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
    assert prepared.runtime_train_features[0, 11] == prepared.train_targets[0]
    assert (
        prepared.runtime_validation_features[0, 11]
        == prepared.validation_targets[0]
    )
    assert prepared.target_leakage_status == "DIRECT_TARGET_SOURCE_EXCLUDED"
    assert prepared.train_subject_groups == prepared.validation_subject_groups
    assert prepared.public_cohort_digest == rows[0]["public_cohort_digest"]
    assert prepared.public_split_digest == rows[0]["public_split_digest"]
    assert prepared.cohort_digest == "a" * 64
    assert prepared.split_digest == "b" * 64
    assert prepared.feature_schema_uuid == WATCH_SCHEMA_UUID
    assert prepared.feature_schema_hash == WATCH_SCHEMA_HASH
    assert prepared.split_policy == "CHRONOLOGICAL_PER_SUBJECT"
    assert prepared.purge_seconds == 1800
    assert prepared.trainer_entrypoint == "stable_standard_hourly_regression"
    assert len(prepared.canonical_handoff_digest) == 64
    assert prepared.fit_call_count == 0

    lineage = module.build_standard_candidate_bundle_lineage(prepared)
    assert lineage == {
        "schema_version": "kidsignal-standard-candidate-lineage/v1",
        "task": "stable_standard_regression",
        "trainer_entrypoint": "stable_standard_hourly_regression",
        "standard_cohort_uuid": STANDARD_COHORT_UUID,
        "cohort_digest": "a" * 64,
        "split_digest": "b" * 64,
        "public_cohort_digest": rows[0]["public_cohort_digest"],
        "public_split_digest": rows[0]["public_split_digest"],
        "canonical_handoff_digest": prepared.canonical_handoff_digest,
        "stable_standard_version": "stable-stress-standard-v1",
        "feature_schema_uuid": WATCH_SCHEMA_UUID,
        "feature_schema_hash": WATCH_SCHEMA_HASH,
        "feature_names": list(FEATURE_NAMES),
        "runtime_input_shape": [None, 16],
        "trainer_feature_names": list(prepared.trainer_feature_names),
        "trainer_input_shape": [None, 15],
        "runtime_to_trainer_indices": list(prepared.runtime_to_trainer_indices),
        "target_source_feature": "watch_load_median_300",
        "target_source_runtime_index": 11,
        "target_leakage_policy": "exclude_exact_target_source_v1",
        "target_leakage_status": "DIRECT_TARGET_SOURCE_EXCLUDED",
        "split_policy": "CHRONOLOGICAL_PER_SUBJECT",
        "purge_seconds": 1800,
        "eligibility_policy": "ELIGIBLE_NO_PATTERN_REAL",
        "target_name": "no_pattern_median",
        "target_unit": "positive_robust_z",
        "source_variant": "watch_only",
        "source_set": ["watch"],
        "onnx_execution_provider": "CPUExecutionProvider",
        "locked_access": False,
        "delivery_eligible": False,
        "promotion_eligible": False,
        "model_status": "NOT_TRAINED",
        "model_artifact_created": False,
        "artifact_creation_gate": "REAL_FROZEN_COHORT_REQUIRED",
        "evaluation_status": "NOT EVALUABLE",
        "fit_call_count": 0,
    }

    leakage_audit = module.audit_standard_target_leakage(prepared)
    assert leakage_audit == {
        "schema_version": "kidsignal-standard-target-leakage-audit/v1",
        "target_name": "no_pattern_median",
        "target_source_feature": "watch_load_median_300",
        "target_source_runtime_index": 11,
        "runtime_feature_count": 16,
        "trainer_feature_count": 15,
        "source_equals_target_in_train": True,
        "source_equals_target_in_validation": True,
        "source_present_in_runtime_contract": True,
        "source_present_in_trainer_input": False,
        "policy": "exclude_exact_target_source_v1",
        "status": "DIRECT_TARGET_SOURCE_EXCLUDED",
        "fit_call_count": 0,
    }


def test_prepared_standard_dataset_requires_verified_reader_receipt() -> None:
    import multisensor_ml.standard_baseline_bigquery as module

    rows = _rows()
    receipt = build_standard_cohort_readiness_receipt(
        rows=rows,
        schema_fields=STANDARD_BASELINE_VIEW_FIELDS,
        requested_standard_cohort_uuid=STANDARD_COHORT_UUID,
        **_expected_public_digests(rows),
        observed_at_utc="2026-08-15T00:00:00Z",
        observed_principal="wrong-principal@example.invalid",
    )

    with pytest.raises(ValueError, match="model-reader identity"):
        module.prepare_standard_dataset_handoff(
            rows=rows,
            readiness_receipt=receipt,
            **_expected_public_digests(rows),
        )


def test_target_leakage_audit_rejects_reintroduced_target_source_feature() -> None:
    import multisensor_ml.standard_baseline_bigquery as module

    rows = _rows()
    receipt = build_standard_cohort_readiness_receipt(
        rows=rows,
        schema_fields=STANDARD_BASELINE_VIEW_FIELDS,
        requested_standard_cohort_uuid=STANDARD_COHORT_UUID,
        **_expected_public_digests(rows),
        observed_at_utc="2026-08-15T00:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )
    prepared = module.prepare_standard_dataset_handoff(
        rows=rows,
        readiness_receipt=receipt,
        **_expected_public_digests(rows),
    )
    unsafe = replace(
        prepared,
        trainer_feature_names=tuple(FEATURE_NAMES),
        runtime_to_trainer_indices=tuple(range(16)),
        train_features=prepared.runtime_train_features,
        validation_features=prepared.runtime_validation_features,
    )

    with pytest.raises(ValueError, match="direct target leakage feature"):
        module.audit_standard_target_leakage(unsafe)


def test_standard_training_projection_matches_cloud_bundle_contract() -> None:
    import multisensor_ml.standard_baseline_bigquery as module

    rows = _rows()
    receipt = build_standard_cohort_readiness_receipt(
        rows=rows,
        schema_fields=STANDARD_BASELINE_VIEW_FIELDS,
        requested_standard_cohort_uuid=STANDARD_COHORT_UUID,
        **_expected_public_digests(rows),
        observed_at_utc="2026-08-15T00:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )
    prepared = module.prepare_standard_dataset_handoff(
        rows=rows,
        readiness_receipt=receipt,
        **_expected_public_digests(rows),
    )

    assert module.build_standard_training_projection(prepared) == {
        "schema_version": "kidsignal-standard-training-projection/v1",
        "task": "stable_standard_regression",
        "trainer_entrypoint": "stable_standard_hourly_regression",
        "feature_schema_uuid": WATCH_SCHEMA_UUID,
        "feature_schema_hash": WATCH_SCHEMA_HASH,
        "feature_names": list(FEATURE_NAMES),
        "runtime_input_shape": ["N", 16],
        "trainer_feature_names": list(prepared.trainer_feature_names),
        "trainer_input_shape": ["N", 15],
        "runtime_to_trainer_indices": [
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
        ],
        "target_name": "no_pattern_median",
        "target_unit": "positive_robust_z",
        "target_source_feature": "watch_load_median_300",
        "target_source_runtime_index": 11,
        "target_leakage_policy": "exclude_exact_target_source_v1",
        "target_leakage_status": "DIRECT_TARGET_SOURCE_EXCLUDED",
        "onnx_input_shape": ["N", 16],
        "projection_embedded_in_onnx": True,
        "leakage_probe_case_ids": [
            "target-source-base",
            "target-source-mutated",
        ],
    }


def test_standard_runtime_projection_and_golden_probe_ignore_target_source() -> None:
    import multisensor_ml.standard_baseline_bigquery as module

    base_values = {
        name: float(index + 1) / 10.0
        for index, name in enumerate(FEATURE_NAMES)
    }
    mutated_values = dict(base_values)
    mutated_values["watch_load_median_300"] = 9.99
    runtime_rows = [
        [base_values[name] for name in FEATURE_NAMES],
        [mutated_values[name] for name in FEATURE_NAMES],
    ]

    projected = module.project_standard_runtime_features(runtime_rows)
    assert projected.shape == (2, 15)
    assert projected[0].tolist() == projected[1].tolist()

    fixture_cases = module.build_standard_leakage_probe_fixture_cases(
        base_values,
        mutated_target_value=9.99,
    )
    assert [case["case_id"] for case in fixture_cases] == [
        "target-source-base",
        "target-source-mutated",
    ]
    assert all(case["expected_runtime_action"] == "INFER" for case in fixture_cases)
    changed = [
        name
        for name in FEATURE_NAMES
        if fixture_cases[0]["feature_values"][name]
        != fixture_cases[1]["feature_values"][name]
    ]
    assert changed == ["watch_load_median_300"]

    output_cases = module.build_standard_leakage_probe_output_cases(
        base_prediction=0.42,
        mutated_prediction=0.42,
    )
    assert output_cases[0]["expected_output"] == output_cases[1]["expected_output"]
    with pytest.raises(ValueError, match="identical prediction"):
        module.build_standard_leakage_probe_output_cases(
            base_prediction=0.42,
            mutated_prediction=0.43,
        )


def test_reader_uses_bound_uuid_and_only_standard_authorized_view(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        if "show" in command:
            payload = [
                {"name": field, "type": "STRING"}
                for field in STANDARD_BASELINE_VIEW_FIELDS
            ]
        else:
            payload = _rows()
        return CompletedProcess(command, 0, json.dumps(payload), "")

    receipt = run_read_only_standard_cohort_reader(
        tmp_path / "standard-receipt.json",
        standard_cohort_uuid=STANDARD_COHORT_UUID,
        **_expected_public_digests(),
        observed_at_utc="2026-08-15T00:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
        runner=runner,
    )

    assert receipt["status"] == "READY_FOR_SYNC_NOT_TRAINED"
    assert receipt["fit_call_count"] == 0
    assert len(calls) == 2
    query = calls[1][-1]
    assert "standard_cohort_uuid=@standard_cohort_uuid" in query
    assert STANDARD_BASELINE_AUTHORIZED_VIEW in query
    assert "training_examples_train_validation_v1" not in query
    assert "kidsignal_training_private" not in query
    assert all(
        token not in query.upper()
        for token in ("LOCKED", "INSERT", "UPDATE", "DELETE", "MERGE")
    )
    assert (
        f"--parameter=standard_cohort_uuid:STRING:{STANDARD_COHORT_UUID}"
        in calls[1]
    )


def test_standard_query_contract_is_transport_neutral_for_kaggle() -> None:
    import multisensor_ml.standard_baseline_bigquery as module

    contract = module.standard_cohort_query_contract(STANDARD_COHORT_UUID)

    assert contract["authorized_view"] == (
        "multi-app-kidsignal-260801.kidsignal_model_training."
        "standard_baseline_train_validation_v1"
    )
    assert contract["parameter"] == {
        "name": "standard_cohort_uuid",
        "type": "STRING",
        "value": STANDARD_COHORT_UUID,
    }
    assert contract["field_count"] == 24
    assert contract["locked_access"] is False
    assert contract["allowed_transports"] == ["bq_cli", "google_cloud_bigquery_sdk"]
    query = str(contract["query"])
    assert "standard_cohort_uuid=@standard_cohort_uuid" in query
    assert "ORDER BY split_role, training_subject_uuid, hour_start_ms" in query
    assert "training_examples_train_validation_v1" not in query
    assert "kidsignal_training_private" not in query
