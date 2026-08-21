import hashlib
import json
from pathlib import Path
from subprocess import CompletedProcess
from uuid import NAMESPACE_URL, uuid5

import pytest

from multisensor_ml.bigquery_training_preflight import (
    AUTHORIZED_TRAINING_VIEW,
    EXPECTED_MODEL_READER,
    EXPECTED_VIEW_FIELDS,
    build_cohort_readiness_receipt,
    build_preflight_receipt,
    run_read_only_cohort_reader,
    run_read_only_preflight,
)
from multisensor_ml.observational_contract import FEATURE_NAMES
from multisensor_ml.platform_contract import synchronize_authorized_training_rows

COHORT_UUID = "10000000-0000-4000-8000-000000000001"
WATCH_SCHEMA_UUID = "9b842d8c-8889-5259-acca-77baa0c7729d"
WATCH_SCHEMA_HASH = "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"
GOLDEN_PUBLIC_COHORT_DIGEST = (
    "e2947f1db7bab1dc1598e2eae69bc109fcd7607004452a57d3b929bd9fe7705f"
)
GOLDEN_PUBLIC_SPLIT_DIGEST = (
    "7f6067b1a2a98d543a3a086a75a67a78b20c560aaffb31b58225de8fcdcf311a"
)


def _uuid(index: int) -> str:
    return f"20000000-0000-4000-8000-{index:012d}"


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
                row["exact_window_id"],
                row["split_role"],
                row["window_start_ms"],
                row["source_row_digest"],
            ]
            for row in rows
        ],
        key=lambda member: (member[0], member[4], member[2]),
    )
    public_split_digest = _canonical_sha256(
        [[member[0], member[2], member[3]] for member in members]
    )
    first = rows[0]
    public_cohort_digest = _canonical_sha256(
        [
            first["feature_schema_uuid"],
            first["feature_schema_hash"],
            first["split_policy"],
            first["purge_seconds"],
            first["truth_state"],
            members,
        ]
    )
    for row in rows:
        row["public_cohort_digest"] = public_cohort_digest
        row["public_split_digest"] = public_split_digest


def _authorized_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index = 0
    for split in ("TRAIN", "VALIDATION"):
        for evaluation, disposition in (
            ("POSITIVE", "TARGET_EVENT"),
            ("NEGATIVE", "VALID_NON_EVENT"),
        ):
            index += 1
            rows.append(
                {
                    "training_cohort_uuid": COHORT_UUID,
                    "cohort_digest": "a" * 64,
                    "split_digest": "b" * 64,
                    "feature_schema_uuid": WATCH_SCHEMA_UUID,
                    "feature_schema_hash": WATCH_SCHEMA_HASH,
                    "split_policy": "PERSON_GROUP",
                    "purge_seconds": 1800,
                    "truth_state": "REVIEWED_REAL",
                    "training_subject_uuid": _uuid(1 if split == "TRAIN" else 2),
                    "training_capture_set_uuid": _uuid(10 + index),
                    "exact_window_id": f"{index:064x}",
                    "source_set": ["watch"],
                    "window_start_ms": index * 2_000_000,
                    "window_end_ms": index * 2_000_000 + 1_000,
                    "feature_values": {
                        name: float(position + 1)
                        for position, name in enumerate(FEATURE_NAMES)
                    },
                    "label_uuid": _uuid(100 + index),
                    "label_revision_uuid": _uuid(200 + index),
                    "review_uuid": _uuid(300 + index),
                    "review_disposition": disposition,
                    "evaluation_class": evaluation,
                    "temporal_stage": 1,
                    "observation_code": (
                        "TARGET_EVENT" if evaluation == "POSITIVE" else "NO_EVENT"
                    ),
                    "split_role": split,
                    "source_row_digest": f"{1000 + index:064x}",
                }
            )
    _attach_public_digests(rows)
    return rows


def _expected_public_digests(
    rows: list[dict[str, object]] | None = None,
) -> dict[str, str]:
    values = rows or _authorized_rows()
    return {
        "expected_public_cohort_digest": str(values[0]["public_cohort_digest"]),
        "expected_public_split_digest": str(values[0]["public_split_digest"]),
    }


def _cloud_authorized_handoff_golden_rows() -> list[dict[str, object]]:
    envelope: dict[str, object] = {
        "training_cohort_uuid": "00000000-0000-4000-8000-000000000100",
        "cohort_digest": "7910d86a46230a04b77d7ece1b6979996c484ceb0872653dc37e4a63c238241b",
        "split_digest": "2a66192d9a770a3483d724b570a36f8726297dcba2e4e2659444a93402b457f3",
        "public_cohort_digest": GOLDEN_PUBLIC_COHORT_DIGEST,
        "public_split_digest": GOLDEN_PUBLIC_SPLIT_DIGEST,
        "feature_schema_uuid": WATCH_SCHEMA_UUID,
        "feature_schema_hash": WATCH_SCHEMA_HASH,
        "split_policy": "PERSON_GROUP",
        "purge_seconds": 1800,
        "truth_state": "REVIEWED_REAL",
    }
    features = {
        name: float(position + 1) for position, name in enumerate(FEATURE_NAMES)
    }
    return [
        {
            **envelope,
            "training_subject_uuid": "560496c2-8216-5533-be6f-884ee3cfe498",
            "training_capture_set_uuid": "138fbfaf-2913-5f21-a42e-8d9e8b80dc20",
            "exact_window_id": "1" * 64,
            "source_set": ["watch"],
            "window_start_ms": 1000,
            "window_end_ms": 1500,
            "feature_values": features,
            "label_uuid": "00000000-0000-4000-8000-000000000020",
            "label_revision_uuid": "00000000-0000-4000-8000-000000000021",
            "review_uuid": "00000000-0000-4000-8000-000000000022",
            "review_disposition": "TARGET_EVENT",
            "evaluation_class": "POSITIVE",
            "temporal_stage": 1,
            "observation_code": "EARLY_PATTERN",
            "split_role": "TRAIN",
            "source_row_digest": (
                "3eaa448aa500d12718a99c0ef3cd5edd522bd33f3ce6dfa064af15c97622e5a8"
            ),
        },
        {
            **envelope,
            "training_subject_uuid": "fe7c9047-2c95-5369-9a5a-08927fc5d967",
            "training_capture_set_uuid": "51c6a435-95e8-54ea-85f2-69bce98feb80",
            "exact_window_id": "2" * 64,
            "source_set": ["watch"],
            "window_start_ms": 1000,
            "window_end_ms": 1500,
            "feature_values": features,
            "label_uuid": "00000000-0000-4000-8000-000000000030",
            "label_revision_uuid": "00000000-0000-4000-8000-000000000031",
            "review_uuid": "00000000-0000-4000-8000-000000000032",
            "review_disposition": "VALID_NON_EVENT",
            "evaluation_class": "NEGATIVE",
            "temporal_stage": 5,
            "observation_code": "NO_EVENT",
            "split_role": "VALIDATION",
            "source_row_digest": (
                "3adcd7a02f4f438148102902d266603f3f1a7a0c6cf8fe69eb881b34e7ae311f"
            ),
        },
    ]


def test_zero_row_authorized_view_is_blocked_without_starting_training() -> None:
    receipt = build_preflight_receipt(
        schema_fields=EXPECTED_VIEW_FIELDS,
        row_count=0,
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )

    assert receipt["schema_version"] == "kidsignal-bigquery-training-preflight/v2"
    assert receipt["mode"] == "READ_ONLY"
    assert receipt["authorized_view"] == AUTHORIZED_TRAINING_VIEW
    assert receipt["authorized_view_is_only_training_input"] is True
    assert receipt["observed_row_count"] == 0
    assert receipt["status"] == "BLOCKED_NO_REAL_COHORT"
    assert receipt["sync_status"] == "NOT STARTED"
    assert receipt["training_status"] == "NOT STARTED"
    assert receipt["evaluation_status"] == "NOT EVALUABLE"
    assert receipt["real_performance_status"] == "NOT VERIFIED"
    assert receipt["locked_access"] is False
    assert receipt["kaggle_started"] is False
    assert receipt["h10_only_status"] == "PROPOSED_NOT_IMPLEMENTED"
    assert receipt["watch_h10_status"] == "PROPOSED_NOT_APPROVED"
    assert len(EXPECTED_VIEW_FIELDS) == 26


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "account_uuid",
        "membership_uuid",
        "provider_subject",
        "person_uuid",
        "source_uuid",
        "capture_session_uuid",
        "raw_payload",
        "artifact_uri",
        "locked_label",
    ],
)
def test_preflight_rejects_private_raw_or_locked_schema_fields(
    forbidden_field: str,
) -> None:
    fields = (*EXPECTED_VIEW_FIELDS, forbidden_field)

    with pytest.raises(ValueError, match="forbidden or unknown field"):
        build_preflight_receipt(
            schema_fields=fields,
            row_count=0,
            observed_at_utc="2026-08-13T12:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
        )


def test_preflight_rejects_missing_public_contract_field() -> None:
    fields = tuple(field for field in EXPECTED_VIEW_FIELDS if field != "review_uuid")

    with pytest.raises(ValueError, match="missing authorized field: review_uuid"):
        build_preflight_receipt(
            schema_fields=fields,
            row_count=0,
            observed_at_utc="2026-08-13T12:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
        )


def test_live_24_field_schema_is_blocked_until_public_digest_migration() -> None:
    legacy_fields = tuple(
        field
        for field in EXPECTED_VIEW_FIELDS
        if field not in {"public_cohort_digest", "public_split_digest"}
    )

    with pytest.raises(ValueError, match="missing authorized field: public_cohort_digest"):
        build_preflight_receipt(
            schema_fields=legacy_fields,
            row_count=0,
            observed_at_utc="2026-08-14T00:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
        )


def test_nonempty_view_still_requires_explicit_cohort_key_before_sync() -> None:
    receipt = build_preflight_receipt(
        schema_fields=EXPECTED_VIEW_FIELDS,
        row_count=4,
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )

    assert receipt["status"] == "AWAITING_EXPLICIT_FROZEN_COHORT_KEY"
    assert receipt["required_next_input"][:3] == [
        "training_cohort_uuid",
        "expected_public_cohort_digest",
        "expected_public_split_digest",
    ]
    assert receipt["sync_status"] == "NOT STARTED"
    assert receipt["training_status"] == "NOT STARTED"


def test_nonempty_view_cannot_sync_under_a_broader_local_principal() -> None:
    receipt = build_preflight_receipt(
        schema_fields=EXPECTED_VIEW_FIELDS,
        row_count=4,
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal="bjcoding2017@gmail.com",
    )

    assert receipt["status"] == "BLOCKED_MODEL_READER_IDENTITY_NOT_VERIFIED"
    assert receipt["model_reader_identity_verified"] is False
    assert receipt["sync_status"] == "NOT STARTED"


def test_read_only_runner_uses_only_exact_view_and_writes_blocked_receipt(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        if "show" in command:
            payload = [{"name": field, "type": "STRING"} for field in EXPECTED_VIEW_FIELDS]
        else:
            payload = [{"row_count": "0"}]
        return CompletedProcess(command, 0, json.dumps(payload), "")

    receipt_path = tmp_path / "preflight.json"
    receipt = run_read_only_preflight(
        receipt_path,
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
        runner=runner,
    )

    assert receipt["status"] == "BLOCKED_NO_REAL_COHORT"
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert len(calls) == 2
    show_target = AUTHORIZED_TRAINING_VIEW.replace(".", ":", 1)
    assert show_target in " ".join(calls[0])
    assert AUTHORIZED_TRAINING_VIEW in " ".join(calls[1])
    query = " ".join(calls[1])
    assert "SELECT COUNT(*)" in query
    assert all(
        keyword not in query.upper()
        for keyword in ("INSERT", "UPDATE", "DELETE", "MERGE")
    )
    assert "kidsignal_training_private" not in query


def test_preflight_receipt_is_create_only(tmp_path: Path) -> None:
    receipt_path = tmp_path / "preflight.json"
    receipt_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_read_only_preflight(
            receipt_path,
            observed_at_utc="2026-08-13T12:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
            runner=lambda command: CompletedProcess(command, 0, "[]", ""),
        )


def test_contract_valid_cohort_is_ready_using_recomputed_public_digests() -> None:
    rows = _authorized_rows()
    receipt = build_cohort_readiness_receipt(
        rows=rows,
        schema_fields=EXPECTED_VIEW_FIELDS,
        requested_training_cohort_uuid=COHORT_UUID,
        expected_public_cohort_digest=str(rows[0]["public_cohort_digest"]),
        expected_public_split_digest=str(rows[0]["public_split_digest"]),
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )

    assert receipt["status"] == "READY_FOR_SYNC_NOT_TRAINED"
    assert receipt["training_ready"] is True
    assert receipt["selection_contract_valid"] is True
    assert receipt["row_count"] == 4
    assert receipt["split_class_counts"] == {
        "TRAIN": {"POSITIVE": 1, "NEGATIVE": 1},
        "VALIDATION": {"POSITIVE": 1, "NEGATIVE": 1},
    }
    assert receipt["training_cohort_uuid"] == COHORT_UUID
    assert receipt["cohort_digest"] == "a" * 64
    assert receipt["split_digest"] == "b" * 64
    assert receipt["public_cohort_digest"] == _authorized_rows()[0][
        "public_cohort_digest"
    ]
    assert receipt["public_split_digest"] == _authorized_rows()[0][
        "public_split_digest"
    ]
    assert receipt["feature_schema_uuid"] == WATCH_SCHEMA_UUID
    assert receipt["feature_schema_hash"] == WATCH_SCHEMA_HASH
    assert receipt["model_reader_identity_verified"] is True
    assert receipt["locked_access"] is False
    assert len(receipt["canonical_handoff_digest"]) == 64
    assert receipt["public_digest_policy"] == (
        "EXPECTED_PUBLIC_COHORT_AND_SPLIT_DIGEST_PIN"
    )
    assert receipt["canonical_handoff_digest_policy"] == (
        "RECOMPUTED_AFTER_EXPECTED_PUBLIC_DIGEST_MATCH"
    )
    assert receipt["expected_public_digests_match"] is True
    assert "private_digest_verification" not in receipt
    assert receipt["fit_call_count"] == 0


def test_expected_public_digest_mismatch_blocks_before_dataset_use() -> None:
    rows = _authorized_rows()
    receipt = build_cohort_readiness_receipt(
        rows=rows,
        schema_fields=EXPECTED_VIEW_FIELDS,
        requested_training_cohort_uuid=COHORT_UUID,
        expected_public_cohort_digest="f" * 64,
        expected_public_split_digest=str(rows[0]["public_split_digest"]),
        observed_at_utc="2026-08-13T12:00:00Z",
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


def test_validated_behavior_rows_prepare_typed_trainer_inputs_without_fit() -> None:
    import multisensor_ml.bigquery_training_preflight as module

    rows = _authorized_rows()
    prepared = module.prepare_behavior_dataset_handoff(
        rows=rows, **_expected_public_digests(rows)
    )

    assert prepared.task == "reviewed_behavior_binary"
    assert prepared.feature_names == tuple(FEATURE_NAMES)
    assert prepared.train_features.shape == (2, len(FEATURE_NAMES))
    assert prepared.validation_features.shape == (2, len(FEATURE_NAMES))
    assert prepared.train_features.dtype.name == "float32"
    assert prepared.train_targets.tolist() == [1, 0]
    assert prepared.validation_targets.tolist() == [1, 0]
    assert len(prepared.train_subject_groups) == 2
    assert len(prepared.train_sequence_groups) == 2
    assert prepared.public_cohort_digest == rows[0]["public_cohort_digest"]
    assert prepared.public_split_digest == rows[0]["public_split_digest"]
    assert len(prepared.canonical_handoff_digest) == 64
    assert prepared.fit_call_count == 0


def test_behavior_query_contract_is_transport_neutral_for_kaggle() -> None:
    import multisensor_ml.bigquery_training_preflight as module

    contract = module.behavior_cohort_query_contract(COHORT_UUID)

    assert contract["authorized_view"] == (
        "multi-app-kidsignal-260801.kidsignal_model_training."
        "training_examples_train_validation_v1"
    )
    assert contract["parameter"] == {
        "name": "training_cohort_uuid",
        "type": "STRING",
        "value": COHORT_UUID,
    }
    assert contract["field_count"] == 26
    assert contract["locked_access"] is False
    assert contract["allowed_transports"] == ["bq_cli", "google_cloud_bigquery_sdk"]
    query = str(contract["query"])
    assert "training_cohort_uuid=@training_cohort_uuid" in query
    assert "ORDER BY split_role, training_subject_uuid, window_start_ms" in query
    assert "standard_baseline_train_validation_v1" not in query
    assert "kidsignal_training_private" not in query


def test_behavior_query_contract_rejects_uuid5_before_any_read(tmp_path: Path) -> None:
    import multisensor_ml.bigquery_training_preflight as module

    cohort_uuid_v5 = str(uuid5(NAMESPACE_URL, "kidsignal-behavior-cohort"))
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="unsupported UUID version"):
        module.behavior_cohort_query_contract(cohort_uuid_v5)
    with pytest.raises(ValueError, match="unsupported UUID version"):
        run_read_only_cohort_reader(
            tmp_path / "uuid5-blocked.json",
            training_cohort_uuid=cohort_uuid_v5,
            **_expected_public_digests(),
            observed_at_utc="2026-08-13T12:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
            runner=lambda command: calls.append(command),
        )

    assert calls == []


def test_not_evaluable_rows_cannot_prepare_trainer_inputs() -> None:
    import multisensor_ml.bigquery_training_preflight as module

    rows = _authorized_rows()
    for row in rows:
        row["truth_state"] = "NOT_EVALUABLE"
    _attach_public_digests(rows)

    with pytest.raises(ValueError, match="REVIEWED_REAL"):
        module.prepare_behavior_dataset_handoff(
            rows=rows, **_expected_public_digests(rows)
        )


def test_model_canonicalization_matches_cloud_authorized_handoff_golden() -> None:
    handoff = synchronize_authorized_training_rows(
        tuple(reversed(_cloud_authorized_handoff_golden_rows()))
    )

    assert handoff["handoff_digest"] == (
        "ee97620fbe4dd243bdd259ba0cdf4e644ce2e0d037b8117fce2b7df04e250400"
    )
    assert handoff["public_cohort_digest"] == GOLDEN_PUBLIC_COHORT_DIGEST
    assert handoff["public_split_digest"] == GOLDEN_PUBLIC_SPLIT_DIGEST
    assert handoff["locked_access"] is False
    assert [row["split_role"] for row in handoff["rows"]] == [
        "TRAIN",
        "VALIDATION",
    ]


def test_zero_row_cohort_is_blocked_without_a_handoff_digest() -> None:
    receipt = build_cohort_readiness_receipt(
        rows=[],
        schema_fields=EXPECTED_VIEW_FIELDS,
        requested_training_cohort_uuid=COHORT_UUID,
        **_expected_public_digests(),
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )

    assert receipt["status"] == "BLOCKED_NO_REAL_COHORT"
    assert receipt["training_ready"] is False
    assert receipt["row_count"] == 0
    assert receipt["canonical_handoff_digest"] is None
    assert receipt["public_cohort_digest"] is None
    assert receipt["public_split_digest"] is None
    assert receipt["fit_call_count"] == 0


def test_missing_class_support_is_blocked_with_counts_only() -> None:
    rows = [row for row in _authorized_rows() if row["evaluation_class"] == "POSITIVE"]
    _attach_public_digests(rows)

    receipt = build_cohort_readiness_receipt(
        rows=rows,
        schema_fields=EXPECTED_VIEW_FIELDS,
        requested_training_cohort_uuid=COHORT_UUID,
        **_expected_public_digests(rows),
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )

    assert receipt["status"] == "BLOCKED_INSUFFICIENT_CLASS_SUPPORT"
    assert receipt["training_ready"] is False
    assert receipt["split_class_counts"]["TRAIN"] == {
        "POSITIVE": 1,
        "NEGATIVE": 0,
    }
    assert "rows" not in receipt
    assert receipt["fit_call_count"] == 0


def test_not_evaluable_truth_never_becomes_training_ready() -> None:
    rows = _authorized_rows()
    for row in rows:
        row["truth_state"] = "NOT_EVALUABLE"
    _attach_public_digests(rows)

    receipt = build_cohort_readiness_receipt(
        rows=rows,
        schema_fields=EXPECTED_VIEW_FIELDS,
        requested_training_cohort_uuid=COHORT_UUID,
        **_expected_public_digests(rows),
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )

    assert receipt["status"] == "BLOCKED_TRUTH_NOT_REVIEWED_REAL"
    assert receipt["training_ready"] is False
    assert receipt["fit_call_count"] == 0


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        ("mixed_cohort", "BLOCKED_MIXED_OR_INVALID_COHORT"),
        ("duplicate_window", "BLOCKED_MIXED_OR_INVALID_COHORT"),
        ("duplicate_source", "BLOCKED_MIXED_OR_INVALID_COHORT"),
        ("nonfinite", "BLOCKED_MIXED_OR_INVALID_COHORT"),
        ("locked", "BLOCKED_MIXED_OR_INVALID_COHORT"),
        ("h10", "BLOCKED_MIXED_OR_INVALID_COHORT"),
    ],
)
def test_invalid_cohort_rows_fail_closed_without_exposing_rows(
    mutation: str,
    expected_status: str,
) -> None:
    rows = _authorized_rows()
    if mutation == "mixed_cohort":
        rows[-1]["training_cohort_uuid"] = _uuid(999)
    elif mutation == "duplicate_window":
        rows[-1]["exact_window_id"] = rows[0]["exact_window_id"]
    elif mutation == "duplicate_source":
        rows[-1]["source_row_digest"] = rows[0]["source_row_digest"]
    elif mutation == "nonfinite":
        rows[-1]["feature_values"][FEATURE_NAMES[0]] = float("nan")
    elif mutation == "locked":
        rows[-1]["split_role"] = "LOCKED"
    elif mutation == "h10":
        rows[-1]["source_set"] = ["h10"]
    else:  # pragma: no cover
        raise AssertionError(mutation)

    receipt = build_cohort_readiness_receipt(
        rows=rows,
        schema_fields=EXPECTED_VIEW_FIELDS,
        requested_training_cohort_uuid=COHORT_UUID,
        **_expected_public_digests(rows),
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
    )

    assert receipt["status"] == expected_status
    assert receipt["training_ready"] is False
    assert "rows" not in receipt
    assert receipt["fit_call_count"] == 0


def test_cohort_reader_uses_bound_parameter_and_only_authorized_view(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        if "show" in command:
            payload = [{"name": field, "type": "STRING"} for field in EXPECTED_VIEW_FIELDS]
        else:
            payload = _authorized_rows()
        return CompletedProcess(command, 0, json.dumps(payload), "")

    receipt = run_read_only_cohort_reader(
        tmp_path / "cohort-receipt.json",
        training_cohort_uuid=COHORT_UUID,
        **_expected_public_digests(),
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
        runner=runner,
    )

    assert receipt["status"] == "READY_FOR_SYNC_NOT_TRAINED"
    assert receipt["expected_public_digests_match"] is True
    assert len(calls) == 2
    query_command = calls[1]
    assert (
        f"--parameter=training_cohort_uuid:STRING:{COHORT_UUID}" in query_command
    )
    query = query_command[-1]
    assert "training_cohort_uuid=@training_cohort_uuid" in query
    assert AUTHORIZED_TRAINING_VIEW in query
    assert "kidsignal_training_private" not in query
    assert all(
        word not in query.upper()
        for word in ("LOCKED", "INSERT", "UPDATE", "DELETE", "MERGE")
    )


def test_cohort_reader_rejects_noncanonical_uuid_before_query(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    with pytest.raises(ValueError, match="canonical UUID"):
        run_read_only_cohort_reader(
            tmp_path / "receipt.json",
            training_cohort_uuid="NOT-A-UUID",
            **_expected_public_digests(),
            observed_at_utc="2026-08-13T12:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
            runner=lambda command: calls.append(command),
        )

    assert calls == []


def test_cohort_reader_blocks_schema_mismatch_before_reading_rows(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str]) -> CompletedProcess[str]:
        calls.append(command)
        payload = [
            {"name": field, "type": "STRING"}
            for field in (*EXPECTED_VIEW_FIELDS, "person_uuid")
        ]
        return CompletedProcess(command, 0, json.dumps(payload), "")

    receipt = run_read_only_cohort_reader(
        tmp_path / "schema-blocked.json",
        training_cohort_uuid=COHORT_UUID,
        **_expected_public_digests(),
        observed_at_utc="2026-08-13T12:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
        runner=runner,
    )

    assert receipt["status"] == "BLOCKED_SCHEMA_CONTRACT_MISMATCH"
    assert receipt["row_count"] == 0
    assert receipt["training_ready"] is False
    assert receipt["fit_call_count"] == 0
    assert len(calls) == 1
